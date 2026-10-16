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
