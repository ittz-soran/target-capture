import datetime as dt
import json
import os

from targetcapture import dvr


def test_stale_job_can_be_cleared(tmp_path):
    job = tmp_path / "1.json"
    job.write_text('{"password": "x"}')
    status = tmp_path / "1.status.json"
    old = (dt.datetime.now() - dt.timedelta(minutes=10)).isoformat(timespec="seconds")
    status.write_text(json.dumps({"state": "running", "updated": old}))
    assert dvr.is_stale(dvr.read_status(str(status)))
    assert not dvr.is_stale({"state": "running", "updated": dt.datetime.now().isoformat()})
    assert not dvr.is_stale({"state": "finished", "updated": old})
    dvr.remove_job(str(status))
    assert not status.exists() and not job.exists() and (tmp_path / "1.stop").exists()


def test_failed_job_shows_reason(tmp_path):
    job = tmp_path / "2.json"
    job.write_text('{"password": "x"}')
    dvr.fail_job(str(job), OSError("disk full"))
    s = dvr.read_status(str(tmp_path / "2.status.json"))
    assert s["state"] == "failed: OSError: disk full" and not job.exists()
    assert not dvr.is_stale(s)


def test_status_write_survives_a_busy_file(tmp_path, monkeypatch):
    """Windows: the page reading the status file makes the swap fail for a moment."""
    real, calls = os.replace, []

    def busy(a, b):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError(5, "Access is denied")
        real(a, b)

    monkeypatch.setattr(dvr.os, "replace", busy)
    path = str(tmp_path / "1.status.json")
    assert dvr.write_status(path, {"state": "running"})
    assert dvr.read_status(path) == {"state": "running"} and len(calls) == 3
    monkeypatch.setattr(dvr.os, "replace", lambda a, b: (_ for _ in ()).throw(PermissionError(5, "denied")))
    assert not dvr.write_status(path, {"state": "x"}, tries=2)  # skipped, not crashed
    assert os.listdir(tmp_path) == ["1.status.json"]
