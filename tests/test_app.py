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
