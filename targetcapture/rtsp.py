"""A small RTSP reader for DVR streams ffmpeg cannot start decoding.

Some DVRs (seen on Hikvision Turbo HD, e.g. iDS-7204HUHI) send H.265 whose setup data (VPS / SPS /
PPS) is only in the stream description, written in a form ffmpeg does not read
("sprop-parameter-sets", the H.264 name), and not in the video itself. ffmpeg then never decodes a
picture ("PPS id out of range", "Skipping invalid undecodable NALU"). This reader talks RTSP over
TCP itself, takes the setup data from wherever the DVR puts it, and hands ffmpeg a plain H.265 /
H.264 stream (Annex B) that starts with it."""
from __future__ import annotations

import base64
import hashlib
import re
import socket
import threading
import time
from urllib.parse import unquote, urlparse

START = b"\x00\x00\x00\x01"


def _fmtp_params(fmtp: str) -> dict:
    out = {}
    for part in fmtp.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip().lower()] = v.strip()
    return out


def _b64_nals(value: str) -> list[bytes]:
    nals = []
    for s in (value or "").split(","):
        s = s.strip()
        if s:
            try:
                nals.append(base64.b64decode(s + "=" * (-len(s) % 4)))
            except ValueError:
                pass
    return [n for n in nals if n]


def parse_sdp(sdp: str) -> dict:
    """The video part of a stream description: codec ("hevc" / "h264"), control address and the
    setup NAL units given in it."""
    video, cur = None, None
    for line in sdp.splitlines():
        line = line.strip()
        if line.startswith("m="):
            cur = {"media": line[2:].split()[0], "pt": (line.split() + ["", "", "", ""])[3], "control": "",
                   "codec": "", "fmtp": ""}
            if cur["media"] == "video" and video is None:
                video = cur
        elif cur is not None and cur is video:
            if line.startswith("a=control:"):
                cur["control"] = line[len("a=control:"):]
            elif line.startswith("a=rtpmap:"):
                name = line.split(None, 1)[1].split("/")[0].upper() if " " in line else ""
                cur["codec"] = {"H265": "hevc", "HEVC": "hevc", "H264": "h264"}.get(name, name.lower())
            elif line.startswith("a=fmtp:"):
                cur["fmtp"] = line.split(None, 1)[1] if " " in line else ""
    if video is None:
        return {"codec": "", "control": "", "params": [], "fmtp": ""}
    p = _fmtp_params(video["fmtp"])
    params = []
    for key in ("sprop-vps", "sprop-sps", "sprop-pps", "sprop-sei", "sprop-parameter-sets"):
        params += _b64_nals(p.get(key, ""))
    return {"codec": video["codec"], "control": video["control"], "params": params, "fmtp": video["fmtp"]}


def nal_type(nal: bytes, codec: str) -> int:
    if not nal:
        return -1
    return (nal[0] >> 1) & 0x3F if codec == "hevc" else nal[0] & 0x1F


def _unescape(nal: bytes) -> bytes:
    return re.sub(b"\x00\x00\x03", b"\x00\x00", nal)


def _escape(rbsp: bytes) -> bytes:
    out, zeros = bytearray(), 0
    for b in rbsp:
        if zeros >= 2 and b <= 3:
            out.append(3)
            zeros = 0
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return bytes(out)


class _Bits:
    def __init__(self, data: bytes):
        self.data, self.pos = data, 0

    def u(self, n: int) -> int:
        v = 0
        for _ in range(n):
            byte = self.data[self.pos >> 3] if (self.pos >> 3) < len(self.data) else 0
            v = (v << 1) | ((byte >> (7 - (self.pos & 7))) & 1)
            self.pos += 1
        return v

    def ue(self) -> int:
        zeros = 0
        while self.u(1) == 0:
            zeros += 1
            if zeros > 31:
                raise ValueError("bad number")
        return (1 << zeros) - 1 + self.u(zeros)


class _BitWriter:
    def __init__(self):
        self.bits: list[int] = []

    def u(self, n: int, v: int):
        self.bits += [(v >> (n - 1 - i)) & 1 for i in range(n)]

    def ue(self, v: int):
        v += 1
        n = v.bit_length()
        self.u(n - 1, 0)
        self.u(n, v)

    def copy(self, r: _Bits, n: int):
        for _ in range(n):
            self.bits.append(r.u(1))

    def done(self) -> bytes:
        self.bits.append(1)  # rbsp stop bit
        while len(self.bits) % 8:
            self.bits.append(0)
        return bytes(int("".join(map(str, self.bits[i:i + 8])), 2) for i in range(0, len(self.bits), 8))


def _ptl_bits(r: _Bits, max_sub_layers_minus1: int) -> int:
    """How many bits the profile_tier_level() at r takes (H.265 7.3.3)."""
    start = r.pos
    r.u(88)  # general profile, tier, flags
    r.u(8)   # general level
    present = []
    for _ in range(max_sub_layers_minus1):
        present.append((r.u(1), r.u(1)))
    if max_sub_layers_minus1 > 0:
        r.u(2 * (8 - max_sub_layers_minus1))
    for prof, lev in present:
        if prof:
            r.u(88)
        if lev:
            r.u(8)
    n = r.pos - start
    r.pos = start
    return n


def make_vps(sps: bytes) -> bytes | None:
    """A video parameter set (VPS) that fits this H.265 SPS, for DVRs that never send one: the
    decoder needs it before it accepts the SPS ("Skipping invalid undecodable NALU: 33")."""
    try:
        r = _Bits(_unescape(sps[2:]))
        vps_id, sub, nesting = r.u(4), r.u(3), r.u(1)
        ptl_start, ptl_len = r.pos, _ptl_bits(r, sub)
        r.pos = ptl_start + ptl_len
        r.ue()  # sps id
        chroma = r.ue()
        if chroma == 3:
            r.u(1)
        r.ue(), r.ue()  # width, height
        if r.u(1):
            for _ in range(4):
                r.ue()
        r.ue(), r.ue()  # bit depths
        r.ue()  # log2_max_pic_order_cnt_lsb_minus4
        info_all = r.u(1)
        order = [(r.ue(), r.ue(), r.ue()) for _ in range(0 if info_all else sub, sub + 1)]
        w = _BitWriter()
        w.u(4, vps_id)
        w.u(1, 1), w.u(1, 1)  # base layer internal / available
        w.u(6, 0)  # max layers - 1
        w.u(3, sub), w.u(1, nesting)
        w.u(16, 0xFFFF)
        r.pos = ptl_start
        w.copy(r, ptl_len)
        w.u(1, 1 if info_all else 0)
        for a, b, c in order:
            w.ue(a), w.ue(b), w.ue(c)
        w.u(6, 0)   # max layer id
        w.ue(0)     # one layer set
        w.u(1, 0)   # no timing info
        w.u(1, 0)   # no extension
        return bytes([32 << 1, sps[1]]) + _escape(w.done())
    except (IndexError, ValueError):
        return None


class Depacketizer:
    """RTP payloads -> whole NAL units (RFC 7798 for H.265, RFC 6184 for H.264)."""

    def __init__(self, codec: str):
        self.codec, self.frag = codec, None

    def push(self, payload: bytes) -> list[bytes]:
        if len(payload) < 2:
            return []
        out = []
        if self.codec == "hevc":
            t = (payload[0] >> 1) & 0x3F
            if t == 48:  # aggregation packet
                i = 2
                while i + 2 <= len(payload):
                    n = int.from_bytes(payload[i:i + 2], "big")
                    out.append(payload[i + 2:i + 2 + n])
                    i += 2 + n
            elif t == 49:  # fragment
                if len(payload) < 3:
                    return []
                fu, ft = payload[2], payload[2] & 0x3F
                if fu & 0x80:
                    self.frag = bytearray([(payload[0] & 0x81) | (ft << 1), payload[1]]) + payload[3:]
                elif self.frag is not None:
                    self.frag += payload[3:]
                if fu & 0x40 and self.frag is not None:
                    out.append(bytes(self.frag))
                    self.frag = None
            else:
                out.append(payload)
        else:
            t = payload[0] & 0x1F
            if t == 24:  # STAP-A
                i = 1
                while i + 2 <= len(payload):
                    n = int.from_bytes(payload[i:i + 2], "big")
                    out.append(payload[i + 2:i + 2 + n])
                    i += 2 + n
            elif t == 28:  # FU-A
                fu = payload[1]
                if fu & 0x80:
                    self.frag = bytearray([(payload[0] & 0xE0) | (fu & 0x1F)]) + payload[2:]
                elif self.frag is not None:
                    self.frag += payload[2:]
                if fu & 0x40 and self.frag is not None:
                    out.append(bytes(self.frag))
                    self.frag = None
            else:
                out.append(payload)
        return [n for n in out if n]


class RtspFeed:
    """Reads one RTSP stream over TCP and gives its video as Annex B bytes (read_chunk)."""

    def __init__(self, url: str, timeout: float = 10):
        u = urlparse(url)
        self.host, self.port = u.hostname or "", u.port or 554
        self.user, self.password = unquote(u.username or ""), unquote(u.password or "")
        self.url = f"rtsp://{self.host}:{self.port}{u.path or '/'}" + (f"?{u.query}" if u.query else "")
        self.timeout = timeout
        self.sock: socket.socket | None = None
        self.buf = b""
        self.cseq = 0
        self.session = ""
        self.auth: dict | None = None
        self.codec = ""
        self.params: list[bytes] = []
        self.seen: dict[int, int] = {}  # NAL type -> how many came (for the details shown on failure)
        self.sent_params = False
        self.has_vps = False
        self.made_vps = False
        self._ka = 0.0
        self.lock = threading.Lock()

    # --- RTSP requests -----------------------------------------------------------------------
    def _auth_header(self, method: str, uri: str) -> str:
        a = self.auth
        if not a:
            return ""
        if a["kind"] == "digest":
            h = lambda x: hashlib.md5(x.encode()).hexdigest()  # noqa: E731
            realm, nonce = a["realm"], a["nonce"]  # (no quotes inside f-strings: Python before 3.12)
            resp = h(f"{h(f'{self.user}:{realm}:{self.password}')}:{nonce}:{h(f'{method}:{uri}')}")
            return (f'Authorization: Digest username="{self.user}", realm="{a["realm"]}", nonce="{a["nonce"]}", '
                    f'uri="{uri}", response="{resp}"\r\n')
        return "Authorization: Basic " + base64.b64encode(f"{self.user}:{self.password}".encode()).decode() + "\r\n"

    def _send(self, method: str, uri: str, extra: str = ""):
        self.cseq += 1
        req = f"{method} {uri} RTSP/1.0\r\nCSeq: {self.cseq}\r\nUser-Agent: TargetCapture\r\n"
        if self.session:
            req += f"Session: {self.session}\r\n"
        req += self._auth_header(method, uri) + extra + "\r\n"
        with self.lock:
            self.sock.sendall(req.encode())

    def _recv_more(self):
        data = self.sock.recv(65536)
        if not data:
            raise ConnectionError("the DVR closed the connection")
        self.buf += data

    def _read_response(self) -> tuple[int, str, bytes]:
        """The next RTSP answer (video packets arriving before it are dropped)."""
        while True:
            while len(self.buf) < 4:
                self._recv_more()
            if self.buf[0:1] == b"$":
                while len(self.buf) < 4:
                    self._recv_more()
                n = int.from_bytes(self.buf[2:4], "big")
                while len(self.buf) < 4 + n:
                    self._recv_more()
                self.buf = self.buf[4 + n:]
                continue
            while b"\r\n\r\n" not in self.buf:
                self._recv_more()
            head, self.buf = self.buf.split(b"\r\n\r\n", 1)
            text = head.decode(errors="replace")
            m = re.search(r"Content-Length:\s*(\d+)", text, re.I)
            n = int(m.group(1)) if m else 0
            while len(self.buf) < n:
                self._recv_more()
            body, self.buf = self.buf[:n], self.buf[n:]
            code = re.match(r"RTSP/\d\.\d (\d+)", text)
            return (int(code.group(1)) if code else 0), text, body

    def _request(self, method: str, uri: str, extra: str = "") -> tuple[int, str, bytes]:
        self._send(method, uri, extra)
        code, head, body = self._read_response()
        if code == 401 and self.auth is None:
            auths = re.findall(r"WWW-Authenticate:\s*(.+)", head, re.I)
            digest = next((a for a in auths if a.lower().startswith("digest")), None)
            if digest:
                f = dict(re.findall(r'(\w+)="?([^",]*)"?', digest))
                self.auth = {"kind": "digest", "realm": f.get("realm", ""), "nonce": f.get("nonce", "")}
            else:
                self.auth = {"kind": "basic"}
            self._send(method, uri, extra)
            code, head, body = self._read_response()
        return code, head, body

    def open(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)
        code, head, body = self._request("DESCRIBE", self.url, "Accept: application/sdp\r\n")
        if code == 401:
            raise PermissionError("the DVR refuses this user name and password")
        if code != 200:
            raise ConnectionError(f"the DVR answered {code or 'nothing'} to the stream address")
        sdp = parse_sdp(body.decode(errors="replace"))
        if sdp["codec"] not in ("hevc", "h264"):
            raise ConnectionError(f"video type {sdp['codec'] or 'unknown'} is not H.265 or H.264")
        self.codec, self.params = sdp["codec"], sdp["params"]
        base = re.search(r"Content-Base:\s*(\S+)", head, re.I)
        base = base.group(1) if base else self.url
        ctl = sdp["control"]
        if not ctl or ctl == "*":
            track = base
        elif ctl.lower().startswith("rtsp://"):
            track = ctl
        else:
            track = base.rstrip("/") + "/" + ctl.lstrip("/")
        code, head, _ = self._request("SETUP", track, "Transport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n")
        if code != 200:
            raise ConnectionError(f"the DVR answered {code} to SETUP")
        m = re.search(r"Session:\s*([^;\r\n]+)", head, re.I)
        self.session = m.group(1).strip() if m else ""
        code, _, _ = self._request("PLAY", base, "Range: npt=0.000-\r\n")
        if code != 200:
            raise ConnectionError(f"the DVR answered {code} to PLAY")
        self._ka = time.time()
        return self

    def _keep_alive(self):
        if time.time() - self._ka > 25:
            self._ka = time.time()
            self._send("GET_PARAMETER", self.url)  # its answer is skipped by the reader

    # --- video -------------------------------------------------------------------------------
    def chunks(self):
        """Annex B video, a piece at a time, starting with the setup data from the description."""
        dep = Depacketizer(self.codec)
        while True:
            self._keep_alive()
            while len(self.buf) < 4:
                self._recv_more()
            if self.buf[0:1] != b"$":  # an answer (e.g. to GET_PARAMETER) in the middle of the video
                if self.buf.startswith(b"RTSP/"):
                    self._read_response()
                else:  # lost our place: find the next packet
                    i = self.buf.find(b"$", 1)
                    self.buf = self.buf[i:] if i > 0 else b""
                continue
            ch, n = self.buf[1], int.from_bytes(self.buf[2:4], "big")
            while len(self.buf) < 4 + n:
                self._recv_more()
            pkt, self.buf = self.buf[4:4 + n], self.buf[4 + n:]
            if ch != 0 or len(pkt) < 12:
                continue
            cc, ext = pkt[0] & 0x0F, pkt[0] & 0x10
            i = 12 + 4 * cc
            if ext and len(pkt) >= i + 4:
                i += 4 + 4 * int.from_bytes(pkt[i + 2:i + 4], "big")
            payload = pkt[i:]
            if pkt[0] & 0x20 and payload:  # padding
                payload = payload[:-payload[-1]]
            out = []
            for nal in dep.push(payload):
                t = nal_type(nal, self.codec)
                self.seen[t] = self.seen.get(t, 0) + 1
                if not self.sent_params:
                    if self.params:
                        sps = next((p for p in self.params if nal_type(p, self.codec) == 33), None)
                        if self.codec == "hevc" and sps and not any(nal_type(p, self.codec) == 32
                                                                    for p in self.params):
                            vps = make_vps(sps)
                            if vps:
                                self.params = [vps] + self.params
                                self.made_vps = True
                        out += [START + p for p in self.params]
                    self.sent_params = True
                    self.has_vps = any(nal_type(p, self.codec) == 32 for p in self.params)
                if self.codec == "hevc" and not self.has_vps:
                    if t == 32:
                        self.has_vps = True
                    elif t == 33:  # an SPS with no VPS before it: make one
                        vps = make_vps(nal)
                        if vps:
                            out.append(START + vps)
                            self.made_vps = True
                out.append(START + nal)
            if out:
                yield b"".join(out)

    def details(self) -> str:
        """What came, in words: helps tell why a picture could not be made."""
        if self.codec == "hevc":
            names = {32: "VPS", 33: "SPS", 34: "PPS", 19: "key frame", 20: "key frame", 21: "key frame", 1: "frame"}
        else:
            names = {7: "SPS", 8: "PPS", 5: "key frame", 1: "frame"}
        got: dict[str, int] = {}
        for t, n in self.seen.items():
            k = names.get(t, "other")
            got[k] = got.get(k, 0) + n
        what = ", ".join(f"{k} {v}" for k, v in sorted(got.items())) or "no video"
        return (f"{'H.265' if self.codec == 'hevc' else 'H.264'}; setup data in the description: "
                f"{len(self.params)}; received: {what}" + ("; made the missing VPS" if self.made_vps else ""))

    def close(self):
        try:
            if self.sock is not None and self.session:
                self._send("TEARDOWN", self.url)
        except OSError:
            pass
        try:
            if self.sock is not None:
                self.sock.close()
        except OSError:
            pass


def pump(feed: RtspFeed, dst, stop) -> str:
    """Copy the feed's video into dst (ffmpeg's stdin) until stop() or an error. Returns the error."""
    try:
        for chunk in feed.chunks():
            if stop():
                break
            view = memoryview(chunk)
            while view:  # a pipe without a buffer can take part of it
                n = dst.write(view)
                view = view[n if n else len(view):]
            dst.flush()
    except (OSError, ValueError, ConnectionError) as e:
        return str(e)
    finally:
        feed.close()
        try:
            dst.close()
        except OSError:
            pass
    return ""
