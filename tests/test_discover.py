"""Answers from DVRs / NVRs / cameras on the network are read (Hikvision SADP, Dahua, ONVIF)."""
import json
import struct

from targetcapture import discover as F

SADP = b"""<?xml version="1.0" encoding="UTF-8"?><ProbeMatch><Uuid>X</Uuid><Types>inquiry</Types>
<DeviceType>139298</DeviceType><DeviceDescription>DS-7208HGHI-K1</DeviceDescription>
<DeviceSN>DS-7208HGHI-K10820200101CCWRE12345678WCVU</DeviceSN><CommandPort>8000</CommandPort>
<HttpPort>80</HttpPort><MAC>a4-14-37-00-00-01</MAC><IPv4Address>192.168.1.230</IPv4Address>
<SoftwareVersion>V4.30.005</SoftwareVersion><AnalogChannelNum>8</AnalogChannelNum>
<DigitalChannelNum>2</DigitalChannelNum><Activated>true</Activated></ProbeMatch>"""

ONVIF = b"""<?xml version="1.0"?><s:Envelope xmlns:s="x" xmlns:d="y"><s:Body><d:ProbeMatches><d:ProbeMatch>
<d:Scopes>onvif://www.onvif.org/type/video_encoder onvif://www.onvif.org/name/HIKVISION%20DS-2CD1163G2-LIU
onvif://www.onvif.org/hardware/DS-2CD1163G2-LIU</d:Scopes>
<d:XAddrs>http://192.168.1.101/onvif/device_service</d:XAddrs></d:ProbeMatch></d:ProbeMatches></s:Body></s:Envelope>"""


def dhip(info):
    body = json.dumps({"method": "client.notifyDevInfo", "params": {"deviceInfo": info}}).encode()
    return b"\x20\x00\x00\x00DHIP" + struct.pack("<IIIIII", 0, 0, len(body), 0, len(body), 0) + body


def test_hikvision_sadp_answer():
    d = F.parse_sadp(SADP, "192.168.1.230")
    assert (d["ip"], d["make"], d["model"], d["channels"], d["kind"]) == (
        "192.168.1.230", F.HIK, "DS-7208HGHI-K1", 10, "DVR / NVR")


def test_onvif_answer_is_a_camera():
    d = F.parse_onvif(ONVIF, "192.168.1.101")
    assert (d["ip"], d["make"], d["model"], d["kind"]) == ("192.168.1.101", F.HIK, "DS-2CD1163G2-LIU", "camera")


def test_dahua_answer():
    d = F.parse_dhip(dhip({"DeviceType": "NVR4108HS-4KS2", "DeviceClass": "NVR", "SerialNo": "6J0",
                           "IPv4Address": {"IPAddress": "192.168.1.108"}, "HttpPort": 80}), "192.168.1.108")
    assert (d["ip"], d["make"], d["kind"]) == ("192.168.1.108", F.DAHUA, "DVR / NVR")
    assert F.parse_dhip(b"junk", "x") is None and F.parse_sadp(b"junk", "x") is None


def test_probes_are_well_formed():
    assert F._dhip_probe()[4:8] == b"DHIP" and b"inquiry" in F._sadp_probe()
    assert all(not ip.startswith("127.") for ip in F.local_ips())
