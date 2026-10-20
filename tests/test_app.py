"""The web page starts and every tab is drawn (catches a tab lost by an edit)."""
import os

import pytest

pytest.importorskip("streamlit")


def test_app_renders_all_tabs(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    app = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
    monkeypatch.chdir(tmp_path)  # fresh index, settings and results
    at = AppTest.from_file(app, default_timeout=120).run()
    assert not at.exception
    labels = [w.label for w in at.text_input]
    # The Search tab opens on "Plate", so its plate-number box renders by default.
    for want in ("Index database", "Plate number", "Filter by vehicle", "Index database file", "IP address"):
        assert any(lbl.startswith(want) for lbl in labels), want
    assert len(at.tabs) == 10

    # The Search tab's other two modes must also render without error.
    at.radio(key="search_kind").set_value("Vehicle").run()
    assert not at.exception
    assert any(w.label.startswith("Make, model") for w in at.text_input)
    at.radio(key="search_kind").set_value("Person").run()
    assert not at.exception
    assert any(w.label.startswith("Match threshold") for w in at.slider)


def test_live_cameras_dvr_picker(tmp_path, monkeypatch):
    """Live cameras: a saved DVR is listed, its cameras are listed, and the start button waits for a pick."""
    from streamlit.testing.v1 import AppTest

    from targetcapture.store import Store

    app = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
    monkeypatch.chdir(tmp_path)
    Store(str(tmp_path / "targetcapture.db")).save_dvr("Shop NVR", "Hikvision / HiLook", "192.168.1.230",
                                                      "admin", "secret", 80, 554)
    cam = dict(status="online", kind="IP", ip="192.168.1.64", port=8000, model="DS-2CD1163G2", protocol="HIKVISION",
               resolution="", serial="", firmware="")
    at = AppTest.from_file(app, default_timeout=120)
    at.session_state["dvr_found"] = []  # no network search in the test
    at.session_state["dvr_cams"] = {"192.168.1.230:80": ([dict(cam, channel=1, name="Gate"),
                                                          dict(cam, channel=2, name="Street")], "DS-7608NI")}
    at.run()
    assert not at.exception
    assert any(s.value == "Watch cameras of a DVR / NVR" for s in at.subheader)
    go = at.button(key="live_dvr_go")
    assert go.disabled  # nothing selected yet
    assert any("Shop NVR" in c.value for c in at.caption)


def _live_picker_app(tmp_path, monkeypatch, picture):
    """App with a saved DVR (2 cameras, channel 2 selected), its network calls stubbed. picture(url,
    transport) says whether that stream gives a picture. Returns (app, store, live processes started)."""
    import subprocess

    import cv2
    import numpy as np
    from streamlit.testing.v1 import AppTest

    import targetcapture.live as LV
    from targetcapture.store import Store

    app = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
    monkeypatch.chdir(tmp_path)
    # the DVR answers only to the Hikvision main-stream address form
    monkeypatch.setattr(LV, "check_login", lambda *a: (True, ""))
    monkeypatch.setattr(LV, "hik_rtsp_port", lambda *a: None)
    monkeypatch.setattr(LV, "rtsp_describe", lambda ip, port, path, u, p, timeout=6:
                        (200, "") if path.startswith("/Streaming/Channels/") else (404, ""))
    jpg = cv2.imencode(".jpg", np.zeros((90, 160, 3), np.uint8))[1].tobytes()
    monkeypatch.setattr(LV, "test_camera", lambda url, timeout=15, transport="tcp", **k:
                        (True, "connected", jpg) if picture(url, transport) else (False, "no picture within 20 s", None))
    started, real_popen = [], subprocess.Popen

    class FakePopen(real_popen):  # the live monitoring process is not really started
        def __init__(self, args, *a, **k):
            if isinstance(args, list) and "live" in args:
                started.append(args)
                self.args, self.pid, self.returncode = args, 0, None
                return
            super().__init__(args, *a, **k)

    monkeypatch.setattr(subprocess, "Popen", FakePopen)
    store = Store(str(tmp_path / "targetcapture.db"))
    store.save_dvr("Shop NVR", "Hikvision / HiLook", "192.168.1.230", "admin", "secret", 80, 554)
    cam = dict(status="online", kind="IP", ip="", port="", model="", protocol="", resolution="", serial="", firmware="")
    at = AppTest.from_file(app, default_timeout=120)
    at.session_state["dvr_found"] = []
    at.session_state["dvr_cams"] = {"192.168.1.230:80": ([dict(cam, channel=1, name="Gate"),
                                                          dict(cam, channel=2, name="Street")], "DS-7608NI")}
    at.session_state["live_cam_table_192.168.1.230:80"] = {"selection": {"rows": [1], "columns": []}}
    at.run()
    return at, store, started


def test_live_cameras_dvr_picker_uses_the_stream_that_gives_a_picture(tmp_path, monkeypatch):
    """No picture on the main stream (e.g. H.265+), one on the sub stream: the camera is added with the
    sub-stream address, and live monitoring starts."""
    at, store, started = _live_picker_app(tmp_path, monkeypatch, lambda url, tr: url.endswith("/202"))
    assert at.text_input(key="live_vp_192.168.1.230:80").value == "secret"  # the saved login is shown
    at.button(key="live_dvr_go").click().run()
    assert not at.exception
    assert [(c["name"], c["url"], c["enabled"]) for c in store.cameras()] == [
        ("Shop NVR ch2", "rtsp://admin:secret@192.168.1.230:554/Streaming/Channels/202", 1)]
    assert len(started) == 1
    assert any(m.value.startswith("Watching 1 camera(s) of Shop NVR live, using the sub stream") for m in at.success)


def test_live_cameras_dvr_picker_adds_nothing_without_a_picture(tmp_path, monkeypatch):
    at, store, started = _live_picker_app(tmp_path, monkeypatch, lambda url, tr: False)
    at.button(key="live_dvr_go").click().run()
    assert not at.exception
    assert store.cameras() == [] and started == []
    assert any("sent no picture" in e.value for e in at.error)
