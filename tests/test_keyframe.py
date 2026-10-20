"""H.265 / H.265+ streams: a key frame is asked of a Hikvision DVR while connecting."""
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from targetcapture.live import request_keyframe


def test_request_keyframe_asks_the_right_channel():
    asked = []

    class H(BaseHTTPRequestHandler):
        def do_PUT(self):
            asked.append(self.path)
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        port = srv.server_address[1]
        url = "rtsp://admin:p%40ss@127.0.0.1:554/Streaming/Channels/202"
        assert request_keyframe(url, port)
        assert asked == ["/ISAPI/Streaming/channels/202/requestKeyFrame"]
        # not a Hikvision address: nothing is asked
        assert not request_keyframe("rtsp://admin:x@127.0.0.1:554/cam/realmonitor?channel=1&subtype=0", port)
        assert len(asked) == 1
    finally:
        srv.shutdown()
