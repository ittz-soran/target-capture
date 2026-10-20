"""Indexing speed changes (no models needed): the per-vehicle look throttle, and the indexing
pipeline still keeping frame pictures and signatures now that the saving thread makes them."""
import os

import cv2
import numpy as np

from targetcapture.plates import PlateIndexer


def _indexer():
    ix = object.__new__(PlateIndexer)  # no models: only the look bookkeeping is tested
    ix._looks = {}
    return ix


def test_second_look_throttle():
    ix = _indexer()
    read, nothing = [{"text": "12345"}], []
    # a passing car: looked at every frame until its plate has been read 3 times
    for t in (0.0, 0.4, 0.8):
        assert ix._look_due(7, t, 10000)
        ix._looked(7, t, 10000, read)
    assert not ix._look_due(7, 1.2, 10000)      # then about once a second
    assert ix._look_due(7, 1.9, 10000)
    assert ix._look_due(7, 1.2, 13500)          # ... or sooner when it comes clearly closer
    # a parked car with no readable plate: after 2 empty looks, every 1.5 s
    for t in (0.0, 0.4):
        assert ix._look_due(9, t, 8000)
        ix._looked(9, t, 8000, nothing)
    assert not ix._look_due(9, 0.8, 8000)
    assert not ix._look_due(9, 1.6, 8000)
    assert ix._look_due(9, 1.9, 8000)
    # a vehicle never seen before is always looked at
    assert ix._look_due(42, 5.0, 100)


class _FakePlates:
    """Stands in for PlateIndexer: one plate read on every 4th sampled frame."""

    timing: dict = {}

    def process(self, frame):
        if frame.index % 4:
            return []
        return [dict(text="12ABC345", conf=0.9, region=None, x1=10, y1=10, x2=60, y2=30, frame=frame.index,
                     t=frame.t, vehicle_type="car", vehicle_color="white", vx1=None, vy1=None, vx2=None, vy2=None)]

    def drain_vehicles(self):
        return []


def test_pipeline_keeps_pictures_and_signatures(tmp_path, monkeypatch):
    from targetcapture.pipeline import index_passes
    from targetcapture.store import Store
    from targetcapture.video import probe

    monkeypatch.chdir(tmp_path)  # fresh settings (keep_frames is on by default)
    path = str(tmp_path / "clip.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 8, (320, 180))
    for i in range(48):
        f = np.zeros((180, 320, 3), np.uint8)
        cv2.putText(f, str(i), (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
        w.write(f)
    w.release()

    store = Store(str(tmp_path / "t.db"))
    info = probe(path)
    mid = store.upsert_media(info, None)
    res = index_passes(store, mid, info, {"plates": _FakePlates()}, {"plates": 8.0}, motion_skip=False)
    assert res["plates"] == 12  # frames 0, 4, ... 44

    kept = sorted(os.listdir(store.kept_dir(mid)))
    assert kept == [f"{i:08d}.jpg" for i in range(0, 48, 4)]
    assert cv2.imread(os.path.join(store.kept_dir(mid), kept[0])).shape == (180, 320, 3)
    n_sigs = store.db.execute("SELECT COUNT(*) FROM frame_sigs WHERE media_id=?", (mid,)).fetchone()[0]
    assert n_sigs == 12
