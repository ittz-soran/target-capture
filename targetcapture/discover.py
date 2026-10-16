"""Find the DVRs / NVRs (and IP cameras) on this PC's local network.

Uses the same "who is there?" broadcasts as the makers' own tools, on the local network only:
  * Hikvision / HiLook SADP (UDP 37020, what the SADP tool shows),
  * Dahua / Imou DHDiscover (UDP 37810, what ConfigTool shows),
  * ONVIF WS-Discovery (UDP 3702), which most other makes answer.
Nothing is logged in to: a device found here still needs its password to list its cameras.
"""
from __future__ import annotations

import json
import re
import select
import socket
import struct
import time
import uuid

GROUP = "239.255.255.250"
SADP_PORT, ONVIF_PORT = 37020, 3702
DH_GROUP, DH_PORT = "239.255.255.251", 37810
HIK, DAHUA = "Hikvision / HiLook", "Dahua / Imou"


def local_ips() -> list[str]:
    """This PC's IPv4 addresses on local networks (not loopback, link-local or Tailscale)."""
    ips = set()
    try:
        ips.update(a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    try:  # the address of the network the default route uses (no packet is sent)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            ips.add(s.getsockname()[0])
    except OSError:
        pass

    def local(ip):
        a, b = map(int, ip.split(".")[:2])
        return not (a == 127 or (a, b) == (169, 254) or (a == 100 and 64 <= b < 128) or a == 0)

    return sorted(ip for ip in ips if local(ip))


def _tag(xml: str, tag: str) -> str:
    m = re.search(rf"<(?:\w+:)?{tag}\b[^>]*>([^<]*)</(?:\w+:)?{tag}>", xml)
    return m.group(1).strip() if m else ""


def kind_of(model: str, channels: int = 0, device_class: str = "") -> str:
    """'DVR / NVR' or 'camera', from the model name, channel count or Dahua device class."""
    m = (model or "").upper()
    if re.search(r"(^|[^A-Z])(I?DS-[79]|DVR|NVR|XVR|HVR|HCVR|NVR)", m) or "VR" in device_class.upper():
        return "DVR / NVR"
    if channels > 1:
        return "DVR / NVR"
    return "camera"


def parse_sadp(data: bytes, sender: str) -> dict | None:
    xml = data.decode("utf-8", "replace")
    if "<ProbeMatch" not in xml:
        return None
    model = _tag(xml, "DeviceDescription") or _tag(xml, "DeviceType")
    ch = sum(int(_tag(xml, t) or 0) for t in ("AnalogChannelNum", "DigitalChannelNum"))
    return {"ip": _tag(xml, "IPv4Address") or sender, "make": HIK, "model": model,
            "serial": _tag(xml, "DeviceSN"), "mac": _tag(xml, "MAC"), "http_port": int(_tag(xml, "HttpPort") or 80),
            "firmware": _tag(xml, "SoftwareVersion"), "channels": ch, "kind": kind_of(model, ch),
            "note": "not activated (set a password with SADP first)" if _tag(xml, "Activated") == "false" else "",
            "via": "SADP"}


def parse_dhip(data: bytes, sender: str) -> dict | None:
    i = data.find(b"{")
    if i < 0 or b"DHIP" not in data[:16]:
        return None
    try:
        info = json.loads(data[i:data.rfind(b"}") + 1].decode("utf-8", "replace"))["params"]["deviceInfo"]
    except (ValueError, KeyError, TypeError):
        return None
    model = info.get("DeviceType", "")
    ip = (info.get("IPv4Address") or {}).get("IPAddress") or sender
    return {"ip": ip, "make": DAHUA, "model": model, "serial": info.get("SerialNo", ""), "mac": info.get("Mac", ""),
            "http_port": int(info.get("HttpPort") or 80), "firmware": info.get("Version", ""), "channels": 0,
            "kind": kind_of(model, 0, info.get("DeviceClass", "")), "note": "", "via": "DHDiscover"}


def parse_onvif(data: bytes, sender: str) -> dict | None:
    xml = data.decode("utf-8", "replace")
    if "ProbeMatch" not in xml:
        return None
    scopes = _tag(xml, "Scopes").split()

    def scope(key):
        from urllib.parse import unquote

        vals = [unquote(s.split(f"/{key}/", 1)[1]) for s in scopes if f"/{key}/" in s]
        return vals[0] if vals else ""

    addrs = _tag(xml, "XAddrs").split()
    m = re.match(r"https?://([^/:\]]+)(?::(\d+))?", addrs[0]) if addrs else None
    model = scope("hardware") or scope("name")
    name = (scope("name") + " " + model).lower()
    make = HIK if "hik" in name or model.upper().startswith(("DS-", "IDS-")) else (
        DAHUA if "dahua" in name or model.upper().startswith(("IPC-", "NVR", "XVR", "DH-")) else "other (ONVIF)")
    return {"ip": m.group(1) if m else sender, "make": make, "model": model, "serial": "", "mac": "",
            "http_port": int(m.group(2) or 80) if m else 80, "firmware": "", "channels": 0,
            "kind": kind_of(model), "note": "", "via": "ONVIF"}


def _onvif_probe() -> bytes:
    return (f'<?xml version="1.0" encoding="UTF-8"?><e:Envelope xmlns:e="http://www.w3.org/2003/05/soap-envelope" '
            'xmlns:w="http://schemas.xmlsoap.org/ws/2004/08/addressing" '
            'xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery" '
            'xmlns:dn="http://www.onvif.org/ver10/network/wsdl"><e:Header>'
            f"<w:MessageID>uuid:{uuid.uuid4()}</w:MessageID><w:To>urn:schemas-xmlsoap-org:ws:2005:04:discovery</w:To>"
            "<w:Action>http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</w:Action></e:Header><e:Body><d:Probe>"
            "<d:Types>dn:NetworkVideoTransmitter</d:Types></d:Probe></e:Body></e:Envelope>").encode()


def _sadp_probe() -> bytes:
    return (f'<?xml version="1.0" encoding="utf-8"?><Probe><Uuid>{str(uuid.uuid4()).upper()}</Uuid>'
            "<Types>inquiry</Types></Probe>").encode()


def _dhip_probe() -> bytes:
    body = json.dumps({"method": "DHDiscover.search", "params": {"mac": "", "uni": 1}}).encode()
    return b"\x20\x00\x00\x00DHIP" + struct.pack("<IIIIII", 0, 0, len(body), 0, len(body), 0) + body


def _listener(port: int, group: str, ifaces: list[str]):
    """A socket on the port the devices answer to by multicast (SADP / DHDiscover do)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("", port))
        for ip in ifaces:
            try:
                s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                             socket.inet_aton(group) + socket.inet_aton(ip))
            except OSError:
                pass
    except OSError:
        s.close()
        return None
    return s


def search(seconds: float = 3.0, ifaces: list[str] | None = None) -> list[dict]:
    """Devices that answered, one per IP address, DVRs / NVRs first."""
    ifaces = local_ips() if ifaces is None else ifaces
    socks, parsers = [], {}
    for port, group, parse in ((SADP_PORT, GROUP, parse_sadp), (DH_PORT, DH_GROUP, parse_dhip)):
        lst = _listener(port, group, ifaces)
        if lst:
            socks.append(lst)
            parsers[lst] = parse
    probes = ((_sadp_probe(), GROUP, SADP_PORT, parse_sadp), (_onvif_probe(), GROUP, ONVIF_PORT, parse_onvif),
              (_dhip_probe(), DH_GROUP, DH_PORT, parse_dhip))
    for ip in ifaces or ["0.0.0.0"]:
        for msg, group, port, parse in probes:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            try:
                s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)  # never leaves the local network
                if ip != "0.0.0.0":
                    s.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(ip))
                s.bind((ip, 0))
                s.sendto(msg, (group, port))
            except OSError:
                s.close()
                continue
            socks.append(s)
            parsers[s] = parse
    found: dict[str, dict] = {}
    end = time.time() + seconds
    while socks and time.time() < end:
        ready, _, _ = select.select(socks, [], [], max(0.0, end - time.time()))
        for s in ready:
            try:
                data, (sender, _) = s.recvfrom(65535)
            except OSError:
                continue
            for parse in (parsers[s], parse_sadp, parse_dhip, parse_onvif):
                d = parse(data, sender)
                if d:
                    old = found.get(d["ip"])
                    if old:  # SADP / DHDiscover know more than ONVIF: keep their fields
                        d = {**d, **{k: v for k, v in old.items() if v}} if d["via"] == "ONVIF" else \
                            {**old, **{k: v for k, v in d.items() if v}}
                    found[d["ip"]] = d
                    break
    for s in socks:
        s.close()
    return sorted(found.values(), key=lambda d: (d["kind"] != "DVR / NVR", tuple(map(int, d["ip"].split(".")))
                                                 if re.fullmatch(r"[\d.]+", d["ip"]) else (999,)))
