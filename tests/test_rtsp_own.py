"""Our own RTSP reader: a pretend DVR that, like the iDS-7204HUHI, sends H.265 with its setup data
(VPS / SPS / PPS) only in the stream description, as "sprop-parameter-sets"."""
import base64
import hashlib
import re
import shutil
import socket
import subprocess
import threading
import time

import pytest

from targetcapture.rtsp import Depacketizer, parse_sdp

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")


def _hevc_nals(tmp_path):
    """A short H.265 clip, as NAL units (key frame every 10 frames)."""
    if b"libx265" not in subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True).stdout:
        pytest.skip("ffmpeg has no H.265 encoder")
    out = tmp_path / "clip.h265"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10", "-t", "6",
                    "-c:v", "libx265", "-x265-params", "keyint=10:min-keyint=10:log-level=none:repeat-headers=1",
                    "-pix_fmt", "yuv420p", "-f", "hevc", str(out)], check=True)
    data = out.read_bytes()
    parts = re.split(b"\x00\x00\x00\x01|\x00\x00\x01", data)
    return [p.rstrip(b"\x00") for p in parts if p]


class FakeDvr:
    """Answers DESCRIBE (digest login), SETUP, PLAY, then plays the clip in a loop over the RTSP
    connection. The setup NAL units are in the description only, never in the video."""

    def __init__(self, nals, user="admin", password="pa@ss"):
        self.params = []
        for t in (32, 33, 34):  # the first VPS, SPS and PPS
            self.params.append(next(n for n in nals if (n[0] >> 1) & 0x3F == t))
        self.video = [n for n in nals if (n[0] >> 1) & 0x3F not in (32, 33, 34, 35, 39, 40)]
        self.user, self.password = user, password
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(4)
        self.port = self.srv.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            threading.Thread(target=self._client, args=(c,), daemon=True).start()

    def _client(self, c):
        buf, realm, nonce = b"", "IP Camera", "abc123"
        try:
            while True:
                while b"\r\n\r\n" not in buf:
                    d = c.recv(4096)
                    if not d:
                        return
                    buf += d
                head, buf = buf.split(b"\r\n\r\n", 1)
                req = head.decode()
                method, uri = req.split()[0], req.split()[1]
                cseq = re.search(r"CSeq: (\d+)", req).group(1)
                m = re.search(r'response="([0-9a-f]+)"', req)
                h = lambda x: hashlib.md5(x.encode()).hexdigest()  # noqa: E731
                want = h(f"{h(f'{self.user}:{realm}:{self.password}')}:{nonce}:{h(f'{method}:{uri}')}")
                if not m or m.group(1) != want:
                    c.sendall(f'RTSP/1.0 401 Unauthorized\r\nCSeq: {cseq}\r\nWWW-Authenticate: Digest '
                              f'realm="{realm}", nonce="{nonce}", stale="FALSE"\r\n\r\n'.encode())
                    continue
                if method == "DESCRIBE":
                    sprop = ",".join(base64.b64encode(p).decode() for p in self.params)
                    sdp = ("v=0\r\no=- 1 1 IN IP4 127.0.0.1\r\ns=Media Presentation\r\nt=0 0\r\n"
                           "m=video 0 RTP/AVP 96\r\na=rtpmap:96 H265/90000\r\n"
                           f"a=fmtp:96 profile-level-id=420029; packetization-mode=1; sprop-parameter-sets={sprop}\r\n"
                           "a=control:rtsp://127.0.0.1/Streaming/Channels/101/trackID=1\r\n")
                    c.sendall(f"RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nContent-Type: application/sdp\r\n"
                              f"Content-Base: {uri}/\r\nContent-Length: {len(sdp)}\r\n\r\n{sdp}".encode())
                elif method == "SETUP":
                    c.sendall(f"RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nSession: 7777;timeout=60\r\n"
                              f"Transport: RTP/AVP/TCP;unicast;interleaved=0-1\r\n\r\n".encode())
                elif method == "PLAY":
                    c.sendall(f"RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\nSession: 7777\r\n\r\n".encode())
                    self._play(c)
                    return
                else:
                    c.sendall(f"RTSP/1.0 200 OK\r\nCSeq: {cseq}\r\n\r\n".encode())
        except OSError:
            return
        finally:
            c.close()

    def _play(self, c):
        seq, ts = 0, 0
        while True:
            for nal in self.video:
                if len(nal) <= 1200:
                    pls = [nal]
                else:  # fragment units (type 49)
                    t, body, pls = (nal[0] >> 1) & 0x3F, nal[2:], []
                    for i in range(0, len(body), 1200):
                        fu = t | (0x80 if i == 0 else 0) | (0x40 if i + 1200 >= len(body) else 0)
                        pls.append(bytes([(nal[0] & 0x81) | (49 << 1), nal[1], fu]) + body[i:i + 1200])
                for pl in pls:
                    rtp = bytes([0x80, 96]) + seq.to_bytes(2, "big") + ts.to_bytes(4, "big") + b"\0\0\0\1" + pl
                    c.sendall(b"$\x00" + len(rtp).to_bytes(2, "big") + rtp)
                    seq = (seq + 1) & 0xFFFF
                if (nal[0] >> 1) & 0x3F < 32:
                    ts = (ts + 9000) & 0xFFFFFFFF
                    time.sleep(0.02)

    def close(self):
        self.srv.close()


@pytest.fixture
def dvr(tmp_path):
    d = FakeDvr(_hevc_nals(tmp_path))
    yield d
    d.close()


def test_sdp_with_h264_style_parameter_sets():
    sdp = ("m=audio 0 RTP/AVP 0\r\na=control:trackID=2\r\nm=video 0 RTP/AVP 96\r\na=rtpmap:96 H265/90000\r\n"
           "a=fmtp:96 sprop-parameter-sets=QAE=,QgE=,RAE=\r\na=control:trackID=1\r\n")
    v = parse_sdp(sdp)
    assert v["codec"] == "hevc" and v["control"] == "trackID=1"
    assert v["params"] == [b"\x40\x01", b"\x42\x01", b"\x44\x01"]


def test_depacketizer_joins_fragments():
    nal = bytes([0x26, 0x01]) + bytes(range(200)) * 20  # a key-frame slice
    body, parts = nal[2:], []
    for i in range(0, len(body), 1000):
        fu = 19 | (0x80 if i == 0 else 0) | (0x40 if i + 1000 >= len(body) else 0)
        parts.append(bytes([(nal[0] & 0x81) | (49 << 1), nal[1], fu]) + body[i:i + 1000])
    d, got = Depacketizer("hevc"), []
    for p in parts:
        got += d.push(p)
    assert got == [nal]


def test_ffmpeg_alone_cannot_start_but_our_reader_can(dvr):
    from targetcapture.live import test_camera

    url = f"rtsp://admin:pa%40ss@127.0.0.1:{dvr.port}/Streaming/Channels/101"
    ok, msg, _ = test_camera(url, timeout=8, http_port=1)  # what Soran's log showed
    assert not ok
    ok, msg, jpg = test_camera(url, timeout=15, transport="own", http_port=1)
    assert ok, msg
    assert jpg[:2] == b"\xff\xd8"


def test_live_reader_switches_to_our_reader(dvr, monkeypatch):
    from targetcapture import live

    url = f"rtsp://admin:pa%40ss@127.0.0.1:{dvr.port}/Streaming/Channels/101"
    r = live.StreamReader(url, hw=False, fps=5)
    r.http_port = 1
    r.transport = "own"  # the tcp and udp tries are skipped to keep the test short
    r.start()
    try:
        t0 = time.time()
        got = None
        while time.time() - t0 < 20 and got is None:
            got = r.take()
            time.sleep(0.1)
        assert got is not None, r.status
        assert got[0].shape == (240, 320, 3)
        assert r.status == "live"
    finally:
        r.stop()
