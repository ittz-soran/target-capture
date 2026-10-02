"""Write search results as JSON, CSV and a self-contained HTML page with snapshots."""
from __future__ import annotations

import csv
import html
import json
import os

import cv2

from .store import clock_time, fmt_offset

GREEN, RED = (60, 200, 60), (40, 40, 230)


def _save(img, path, max_w=None):
    if img is None or img.size == 0:
        return None
    if max_w and img.shape[1] > max_w:
        img = cv2.resize(img, None, fx=max_w / img.shape[1], fy=max_w / img.shape[1])
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return os.path.basename(path)


def _crop(img, x1, y1, x2, y2, pad=0.15, min_w=160):
    if img is None:
        return None
    w, h = x2 - x1, y2 - y1
    x1, y1 = max(0, int(x1 - w * pad)), max(0, int(y1 - h * pad))
    x2, y2 = min(img.shape[1], int(x2 + w * pad)), min(img.shape[0], int(y2 + h * pad))
    c = img[y1:y2, x1:x2]
    if c.size and c.shape[1] < min_w:
        c = cv2.resize(c, None, fx=min_w / c.shape[1], fy=min_w / c.shape[1], interpolation=cv2.INTER_CUBIC)
    return c


def _times(media, start, end):
    d = {"start": fmt_offset(start), "end": fmt_offset(end), "duration_s": round(end - start, 1),
         "start_seconds": round(start, 2), "end_seconds": round(end, 2)}
    cs, ce = clock_time(media, start), clock_time(media, end)
    if cs:
        d["clock_start"], d["clock_end"] = cs, ce
    return d


def plate_report(store, query, hits, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for i, s in enumerate(hits, 1):
        media = store.media(s["media_id"])
        b = s["best"]
        snap, crop = _plate_snapshot(store, s, b, out_dir, f"{i:03d}")
        b2 = s.get("second")
        snap2, crop2 = _plate_snapshot(store, s, b2, out_dir, f"{i:03d}_2") if b2 else (None, None)
        rows.append({"n": i, "plate": s["text"], "exact": s.get("exact"), "partial_match": s.get("partial"), "edit_distance": s.get("distance"),
                     "ocr_variants": s["variants"], "reads": s["reads"], "ocr_conf": s["conf"],
                     "region_guess": s["region"], "vehicle_type": s["vehicle_type"],
                     "vehicle_color": s["vehicle_color"], "vehicle_model": s["vehicle_model"],
                     "model_confidence": s["model_conf"], "other_possible_models": s["model_alternatives"] or None,
                     **_times(media, s["start"], s["end"]),
                     "best_capture_at": fmt_offset(b["t"]),
                     "second_capture_at": fmt_offset(b2["t"]) if b2 else None,
                     "frame": b["frame"], "video": media["path"], "frame_image": snap, "plate_image": crop,
                     "second_frame_image": snap2, "second_plate_image": crop2})
    _write(out_dir, rows, f'Plate search: "{query}"',
           f"{len(rows)} sighting(s) of plates matching <b>{html.escape(query)}</b>")
    return rows


def _plate_snapshot(store, s, b, out_dir, stem):
    """Save the whole frame (boxes drawn) and the plate close-up of one capture."""
    frame = store.frame(b)
    if frame is None:
        return None, None
    crop = _save(_crop(frame, b["x1"], b["y1"], b["x2"], b["y2"], pad=0.1, min_w=240),
                 os.path.join(out_dir, f"{stem}_plate.jpg"))
    if b["vx1"] is not None:
        cv2.rectangle(frame, (b["vx1"], b["vy1"]), (b["vx2"], b["vy2"]), GREEN, 2)
    cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), RED, 2)
    cv2.putText(frame, s["text"], (b["x1"], max(20, b["y1"] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)
    return _save(frame, os.path.join(out_dir, f"{stem}_frame.jpg"), max_w=1280), crop


def person_report(store, query_paths, sightings, stats, threshold, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for i, s in enumerate(sightings, 1):
        media = store.media(s["media_id"])
        b = s["best"]
        frame = store.frame(b)
        snap = face = None
        if frame is not None:
            face = _save(_crop(frame, b["fx1"], b["fy1"], b["fx2"], b["fy2"], pad=0.3, min_w=160),
                         os.path.join(out_dir, f"{i:03d}_face.jpg"))
            cv2.rectangle(frame, (b["x1"], b["y1"]), (b["x2"], b["y2"]), GREEN, 2)
            cv2.rectangle(frame, (b["fx1"], b["fy1"]), (b["fx2"], b["fy2"]), RED, 2)
            cv2.putText(frame, f"match {s['sim']:.2f}", (b["x1"], max(20, b["y1"] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)
            snap = _save(frame, os.path.join(out_dir, f"{i:03d}_frame.jpg"), max_w=1280)
        rows.append({"n": i, "similarity": round(s["sim"], 3), "confidence": s["confidence"],
                     **_times(media, s["start"], s["end"]), "face_matches": s["face_hits"],
                     "position_in_frame": s["position"], "upper_clothing": s["upper_color"],
                     "lower_clothing": s["lower_color"], "best_frame": b["frame"],
                     "best_frame_time": fmt_offset(b["t"]), "video": media["path"],
                     "frame_image": snap, "face_image": face})
    total = sum(r["duration_s"] for r in rows)
    summary = (f"Person seen <b>{len(rows)}</b> time(s), about {total:.0f}s on camera in total. "
               f"Threshold {threshold:.2f}; {stats.get('faces_indexed', 0)} faces in the index; "
               f"{stats.get('near_misses', 0)} track(s) just below threshold.")
    _write(out_dir, rows, "Person search: " + ", ".join(os.path.basename(p) for p in query_paths), summary)
    return rows


def _write(out_dir, rows, title, summary):
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(rows, f, indent=2, default=str)
    if rows:
        with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow({k: ("; ".join(v) if isinstance(v, list) else v) for k, v in r.items()})
    cards = []
    for r in rows:
        imgs = "".join(f'<img src="{html.escape(r[k])}">' for k in r if k.endswith("_image") and r[k])
        facts = "".join(f"<tr><th>{html.escape(k.replace('_', ' '))}</th><td>{html.escape(str(v))}</td></tr>"
                        for k, v in r.items() if not k.endswith("_image") and k != "n" and v is not None)
        cards.append(f'<section><h2>#{r["n"]}</h2><div class="imgs">{imgs}</div><table>{facts}</table></section>')
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font:14px system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}}
section{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0}}
.imgs{{display:flex;gap:8px;flex-wrap:wrap}}.imgs img{{max-width:100%;max-height:360px;border-radius:4px}}
th{{text-align:left;padding-right:12px;color:#666;font-weight:normal}}h2{{margin:0 0 8px}}</style></head>
<body><h1>{html.escape(title)}</h1><p>{summary}</p>{''.join(cards) or '<p>No matches.</p>'}</body></html>"""
    with open(os.path.join(out_dir, "report.html"), "w") as f:
        f.write(page)


def thumbnail(row, store, cache_dir, box=("x1", "y1", "x2", "y2"), pad=0.15, min_w=160, full=False):
    """Crop (or whole frame with the box drawn, if full=True) for one stored detection,
    cached on disk so galleries do not re-decode video on every page load."""
    x1, y1, x2, y2 = (row[k] for k in box)
    name = f"{row['media_id']}_{row['frame']}_{x1}_{y1}_{'f' if full else 'c'}2.jpg"
    path = os.path.join(cache_dir, name)
    if os.path.exists(path):
        return path
    frame = store.frame(row)
    if frame is None:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    if full:
        cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 3)
        return os.path.join(cache_dir, _save(frame, path, max_w=1280))
    return os.path.join(cache_dir, _save(_crop(frame, x1, y1, x2, y2, pad=pad, min_w=min_w), path))
