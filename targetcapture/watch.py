"""Watch list: wanted plates and faces, checked automatically on every indexed video and on
live cameras. A match becomes an alert (shown at the top of the page until marked seen)."""
from __future__ import annotations

import cv2
import numpy as np

from . import persons as P
from . import plates as L


def add_plate(store, plate: str, note: str = "") -> int:
    plate = plate.strip()
    if not L.fold(plate.replace("*", "").replace("?", "")):
        raise ValueError("type a plate number")
    return store.add_watch("plate", plate, note)


def add_face(store, photo_path: str, face_model, note: str = "") -> int:
    """Wanted person from a photo (the largest face in it)."""
    import os

    emb = face_model.query_embedding(photo_path)
    img = cv2.imread(photo_path)
    faces = face_model.faces(img) if img is not None else []
    thumb = img
    if faces:
        (x1, y1, x2, y2), _, _ = max(faces, key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1]))
        w, h = x2 - x1, y2 - y1
        thumb = img[max(0, int(y1 - h * .3)):int(y2 + h * .3), max(0, int(x1 - w * .3)):int(x2 + w * .3)]
    if thumb is not None and thumb.shape[1] > 200:
        thumb = cv2.resize(thumb, (200, int(thumb.shape[0] * 200 / thumb.shape[1])))
    jpg = cv2.imencode(".jpg", thumb)[1].tobytes() if thumb is not None and thumb.size else None
    return store.add_watch("face", os.path.basename(photo_path), note, emb=emb, photo=jpg)


def check(store, media_id: int | None = None, plate_max_dist: int = 1, face_threshold: float = 0.45) -> list[dict]:
    """Look for every watch-list entry in the index (one video, or all). Returns new alerts."""
    items = store.watchlist(active_only=True)
    new = []
    plates = [w for w in items if w["kind"] == "plate"]
    if plates:
        reads = store.plate_reads(media_id) if media_id is not None else store.plate_reads()
        sightings = L.group_sightings(reads)
        for w in plates:
            for s in sightings:
                hits = [L.match_score(w["value"], v, plate_max_dist) for v in s["variants"]]
                ds = [d for ok, d in hits if ok]
                if not ds:
                    continue
                b = s["best"]
                a = {"watch_id": w["id"], "media_id": s["media_id"], "start": s["start"], "end": s["end"],
                     "text": s["text"], "score": 1.0 - min(ds) / max(1, len(L.fold(s["text"]))),
                     "frame": b["frame"], "x1": b["x1"], "y1": b["y1"], "x2": b["x2"], "y2": b["y2"]}
                if store.add_alert(a):
                    new.append({**a, "kind": "plate", "value": w["value"], "note": w["note"]})
    for w in (w for w in items if w["kind"] == "face"):
        emb = np.frombuffer(w["emb"], np.float32)
        found, _ = P.search(store, [emb], threshold=face_threshold)
        for s in found:
            if media_id is not None and s["media_id"] != media_id:
                continue
            b = s["best"]
            a = {"watch_id": w["id"], "media_id": s["media_id"], "start": s["start"], "end": s["end"],
                 "text": f"face {s['sim']:.0%} alike", "score": float(s["sim"]),
                 "frame": b["frame"], "x1": b["fx1"], "y1": b["fy1"], "x2": b["fx2"], "y2": b["fy2"]}
            if store.add_alert(a):
                new.append({**a, "kind": "face", "value": w["value"], "note": w["note"]})
    return new
