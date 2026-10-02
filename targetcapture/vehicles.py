"""Every vehicle seen, also those whose plate could not be read: search by type, colour,
make / model and time ("white Camry between 08:00 and 09:00")."""
from __future__ import annotations

from collections import Counter, defaultdict

from .plates import group_sightings
from .store import clock_time


def sightings(store, media_id: int | None = None) -> list[dict]:
    """One entry per vehicle pass (tracked vehicle), with the plate if one was read on it."""
    by_track: dict[tuple, list] = defaultdict(list)
    for r in store.vehicle_rows(media_id):
        by_track[(r["media_id"], r["track_id"])].append(dict(r))
    # plates read on a vehicle: match reads to vehicle rows by frame and box
    plates_on: dict[tuple, Counter] = defaultdict(Counter)
    reads = store.plate_reads(media_id) if media_id is not None else store.plate_reads()
    for s in group_sightings(reads):
        b = s["best"]
        if b.get("vx1") is None:
            continue
        # the vehicle track whose box at about the same time overlaps the plate's vehicle box
        for (mid, tid), rows in by_track.items():
            if mid != s["media_id"] or rows[0]["t"] > s["end"] + 1 or rows[-1]["t"] < s["start"] - 1:
                continue
            near = min(rows, key=lambda r: abs(r["t"] - b["t"]))
            if abs(near["t"] - b["t"]) <= 1.5 and _overlap(near, b) > 0.3:
                plates_on[(mid, tid)][s["text"]] += s["reads"]
    out = []
    for (mid, tid), rows in by_track.items():
        models = Counter()
        for r in rows:
            if r["vehicle_model"]:
                models[r["vehicle_model"]] += r["model_conf"] or 0
        best = max(rows, key=lambda r: (r["x2"] - r["x1"]) * (r["y2"] - r["y1"]))
        plate = plates_on[(mid, tid)].most_common(1)[0][0] if plates_on[(mid, tid)] else None
        out.append({
            "media_id": mid, "track_id": tid, "start": rows[0]["t"], "end": rows[-1]["t"],
            "vehicle_type": Counter(r["vehicle_type"] for r in rows).most_common(1)[0][0],
            "vehicle_color": Counter(r["vehicle_color"] for r in rows if r["vehicle_color"]).most_common(1)[0][0]
            if any(r["vehicle_color"] for r in rows) else None,
            "vehicle_model": models.most_common(1)[0][0] if models else None,
            "plate": plate, "best": best, "detections": len(rows),
        })
    return sorted(out, key=lambda v: (v["media_id"], v["start"]))


def _overlap(a, plate_read) -> float:
    b = {"x1": plate_read["vx1"], "y1": plate_read["vy1"], "x2": plate_read["vx2"], "y2": plate_read["vy2"]}
    ix = max(0, min(a["x2"], b["x2"]) - max(a["x1"], b["x1"]))
    iy = max(0, min(a["y2"], b["y2"]) - max(a["y1"], b["y1"]))
    inter = ix * iy
    union = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
    return inter / union if union else 0.0


def search(store, words: str = "", vehicle_type: str | None = None, color: str | None = None,
           plate: str = "any", start: str | None = None, end: str | None = None,
           min_seconds: float = 0.0) -> list[dict]:
    """Filter vehicle passes. words: make/model/colour words ("white toyota camry").
    plate: "any", "read" (plate was read) or "unread" (no plate read). start/end: video time
    "HH:MM:SS" or DVR clock "YYYY-MM-DD HH:MM:SS" (clock needs the video's clock start)."""
    ws = words.lower().split()
    out = []
    for v in sightings(store):
        text = " ".join(str(v[k] or "") for k in ("vehicle_model", "vehicle_color", "vehicle_type", "plate")).lower()
        if any(w not in text for w in ws):
            continue
        if vehicle_type and v["vehicle_type"] != vehicle_type:
            continue
        if color and v["vehicle_color"] != color:
            continue
        if plate == "read" and not v["plate"] or plate == "unread" and v["plate"]:
            continue
        if v["end"] - v["start"] < min_seconds:
            continue
        media = store.media(v["media_id"])
        if media is None or ((start or end) and not _in_range(media, v["start"], start, end)):
            continue
        out.append(v)
    return out


def _in_range(media, t: float, start: str | None, end: str | None) -> bool:
    def secs(s):
        h, m, x = (s.split(":") + ["0", "0"])[:3]
        return int(h) * 3600 + int(m) * 60 + float(x)

    for bound, ok in ((start, lambda a, b: a >= b), (end, lambda a, b: a <= b)):
        if not bound:
            continue
        bound = bound.strip()
        if len(bound) > 8:  # DVR clock
            clock = clock_time(media, t)
            if clock is None or not ok(clock, bound):
                return False
        elif not ok(t, secs(bound)):
            return False
    return True
