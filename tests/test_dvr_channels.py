"""The DVR's own camera list is read (fake Hikvision and Dahua web pages)."""
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import unquote

from targetcapture import dvr

HIK = {
    "/ISAPI/System/deviceInfo": "<DeviceInfo><model>DS-7608NI-K2</model></DeviceInfo>",
    "/ISAPI/System/Video/inputs/channels": (
        "<VideoInputChannelList><VideoInputChannel><id>1</id><inputPort>1</inputPort>"
        "<videoInputEnabled>true</videoInputEnabled><name>Gate</name><videoFormat>PAL</videoFormat>"
        "<resDesc>1080P25</resDesc></VideoInputChannel><VideoInputChannel><id>2</id>"
        "<videoInputEnabled>true</videoInputEnabled><name>Back</name><resDesc>NO VIDEO</resDesc>"
        "</VideoInputChannel></VideoInputChannelList>"),
    "/ISAPI/ContentMgmt/InputProxy/channels": (
        "<InputProxyChannelList><InputProxyChannel><id>3</id><name>Street</name><sourceInputPortDescriptor>"
        "<proxyProtocol>HIKVISION</proxyProtocol><addressingFormatType>ipaddress</addressingFormatType>"
        "<ipAddress>192.168.1.101</ipAddress><managePortNo>8000</managePortNo>"
        "<model>DS-2CD1163G2-LIU</model></sourceInputPortDescriptor></InputProxyChannel></InputProxyChannelList>"),
    "/ISAPI/ContentMgmt/InputProxy/channels/status": (
        "<InputProxyChannelStatusList><InputProxyChannelStatus><id>3</id><online>false</online>"
        "<chanDetectResult>netUnreachable</chanDetectResult></InputProxyChannelStatus>"
        "</InputProxyChannelStatusList>"),
}
DAHUA = {
    "/cgi-bin/magicBox.cgi?action=getDeviceType": "type=NVR4108HS\r\n",
    "/cgi-bin/configManager.cgi?action=getConfig&name=ChannelTitle":
        "table.ChannelTitle[0].Name=Gate\r\ntable.ChannelTitle[1].Name=Shop\r\n",
    "/cgi-bin/LogicDeviceManager.cgi?action=getCameraAll":
        "camera[0].Channel=1\r\ncamera[0].DeviceInfo.Address=192.168.1.108\r\ncamera[0].DeviceInfo.Port=37777\r\n"
        "camera[0].DeviceInfo.DeviceType=IPC-HFW1230S\r\ncamera[0].DeviceInfo.ProtocolType=Dahua2\r\n",
    "/cgi-bin/LogicDeviceManager.cgi?action=getCameraState&uniqueChannels[0]=-1":
        "states[0].channel=1\r\nstates[0].connectionState=Connected\r\n",
}


def serve(pages):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = pages.get(unquote(self.path))
            self.send_response(200 if body else 404)
            self.end_headers()
            self.wfile.write((body or "").encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_hikvision_channels():
    srv = serve(HIK)
    try:
        chans, model = dvr.list_channels("Hikvision / HiLook", "127.0.0.1", "admin", "x", srv.server_port)
    finally:
        srv.shutdown()
    assert model == "DS-7608NI-K2"
    assert [(c["channel"], c["name"], c["kind"], c["status"]) for c in chans] == [
        (1, "Gate", "analog", "online"), (2, "Back", "analog", "no video"),
        (3, "Street", "IP", "offline: netUnreachable")]
    assert chans[2]["ip"] == "192.168.1.101" and chans[2]["model"] == "DS-2CD1163G2-LIU"


def test_dahua_channels():
    srv = serve(DAHUA)
    try:
        chans, model = dvr.list_channels("Dahua / Imou", "127.0.0.1", "admin", "x", srv.server_port)
    finally:
        srv.shutdown()
    assert model == "NVR4108HS"
    assert chans[1]["name"] == "Shop" and chans[1]["ip"] == "192.168.1.108" and chans[1]["status"] == "online"
    assert chans[0]["kind"] == "analog" and chans[1]["port"] == "37777"


def test_no_dvr():
    chans, why = dvr.list_channels("Hikvision / HiLook", "127.0.0.1", "admin", "x", 1)
    assert chans is None and "web port" in why


def test_page_lists_dvrs_then_cameras(tmp_path, monkeypatch):
    import os

    import pytest

    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    from targetcapture.store import Store

    srv = serve(HIK)
    monkeypatch.chdir(tmp_path)
    Store("targetcapture.db").save_dvr("Home DVR", "Hikvision / HiLook", "127.0.0.1", "admin", "x",
                                       srv.server_port, 554)
    app = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
    try:
        at = AppTest.from_file(app, default_timeout=120).run()
    finally:
        srv.shutdown()
    assert not at.exception
    heads = [h.value for h in at.subheader]
    assert "Your DVRs / NVRs" in heads and "Cameras on Home DVR (DS-7608NI-K2)" in heads
    assert [m for m in at.multiselect if m.label == "Channels"][0].options[:3] == ["1 · Gate", "2 · Back", "3 · Street"]
