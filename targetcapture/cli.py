"""Command line interface.

    python -m targetcapture index VIDEO_OR_IMAGE [--plates] [--persons] [--clock-start "2026-09-30 08:00:00"]
    python -m targetcapture plates                      # every plate read, one line per sighting
    python -m targetcapture find-plate 12ABC345         # search a plate (fuzzy, wildcards ok: "12?BC*")
    python -m targetcapture find-person face.jpg [more.jpg ...]
    python -m targetcapture media                       # what is indexed
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys

from tqdm import tqdm

from . import persons as P
from . import plates as L
from .report import person_report, plate_report
from .store import Store, clock_time, fmt_offset
from .pipeline import index_media
from .video import probe

log = logging.getLogger("targetcapture")


def cmd_index(a):
    store = Store(a.db)
    do_plates = a.plates or not a.persons
    do_persons = a.persons or not a.plates
    bars: dict[str, tqdm] = {}

    def progress(kind, i, total, found, t):
        bar = bars.get(kind)
        if bar is None:
            bar = bars[kind] = tqdm(total=total, initial=i, unit="frame", desc=kind, smoothing=0.05)
        bar.update(i - bar.n)
        bar.set_postfix({"found": found, "at": fmt_offset(t)}, refresh=False)

    for path in a.media:
        info = probe(path)
        print(f"{path}: {info.kind}, {info.width}x{info.height}, {info.fps:.1f} fps, "
              f"{fmt_offset(info.duration)} long")
        bars.clear()
        res = index_media(store, path, plates=do_plates, persons=do_persons, plate_fps=a.plate_fps,
                          person_fps=a.person_fps, clock_start=a.clock_start, min_ocr_conf=a.min_ocr_conf,
                          vehicle_crops=a.vehicle_crops, plate_script=a.plate_script, car_models=not a.no_car_model, motion_skip=not a.no_motion_skip,
                          reindex=a.reindex, progress=progress)
        for bar in bars.values():
            bar.close()
        for kind in ("plates", "persons"):
            if kind in res:
                n = res[kind]
                print(f"[{kind}] " + ("already indexed (use --reindex to redo)" if n < 0 else f"{n} detections stored"))


def _slug(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:40] or "query"


def cmd_plates(a):
    store = Store(a.db)
    sightings = [s for s in L.group_sightings(store.plate_reads(), gap=a.gap) if s["reads"] >= a.min_reads]
    if a.vehicle:  # e.g. --vehicle corolla, --vehicle "white toyota"
        words = a.vehicle.lower().split()
        sightings = [s for s in sightings if all(w in " ".join(str(s[k] or "") for k in (
            "vehicle_model", "vehicle_color", "vehicle_type", "region")).lower() for w in words)]
    print(f"{len(sightings)} plate sighting(s); {len({s['text'] for s in sightings})} distinct plates")
    for s in sightings:
        media = store.media(s["media_id"])
        clock = clock_time(media, s["start"])
        print(f"{s['text']:<12} {fmt_offset(s['start'])}-{fmt_offset(s['end'])}"
              f"{'  ' + clock if clock else ''}  reads={s['reads']:<3} conf={s['conf']:.2f}  "
              f"{s['vehicle_color'] or ''} {s['vehicle_model'] or s['vehicle_type'] or ''}  {os.path.basename(media['path'])}")


def cmd_find_plate(a):
    store = Store(a.db)
    hits = L.search(store, a.plate, max_dist=a.max_dist, gap=a.gap, min_reads=a.min_reads)
    out = a.out or os.path.join("results", "plate_" + _slug(a.plate))
    rows = plate_report(store, a.plate, hits, out)
    if not rows:
        print(f'No plate matching "{a.plate}". Try --max-dist 2, a wildcard like "12?BC*", '
              "or `plates` to list everything read.")
    for r in rows:
        when = r.get("clock_start") or r["start"]
        d = r["edit_distance"]
        tag = ("exact" if r["exact"] else "part of the plate" + (f", {d} character(s) differ" if d else "")
               if r["partial_match"] else "look-alike characters" if d == 0 else f"{d} character(s) differ")
        print(f"#{r['n']} {r['plate']} [{tag}] at {when} for {r['duration_s']}s, {r['reads']} reads, "
              f"{r['vehicle_color'] or '?'} {r['vehicle_model'] or r['vehicle_type'] or 'vehicle'}, "
              f"{r['region_guess'] + ', ' if r['region_guess'] else ''}{os.path.basename(r['video'])}")
    print(f"Report: {os.path.join(out, 'report.html')}")


def cmd_find_person(a):
    store = Store(a.db)
    fm = P.FaceModel()
    embs = [fm.query_embedding(p) for p in a.photos]
    sightings, stats = P.search(store, embs, threshold=a.threshold, merge_gap=a.merge_gap, min_face_px=a.min_face)
    out = a.out or os.path.join("results", "person_" + _slug(os.path.basename(a.photos[0])))
    rows = person_report(store, a.photos, sightings, stats, a.threshold, out)
    print(f"Seen {len(rows)} time(s). ({stats.get('faces_indexed', 0)} faces indexed, "
          f"{stats.get('near_misses', 0)} near misses below threshold {a.threshold})")
    for r in rows:
        when = r.get("clock_start") or r["start"]
        print(f"#{r['n']} {when} to {r.get('clock_end') or r['end']} ({r['duration_s']}s) "
              f"similarity {r['similarity']} [{r['confidence']}], {r['upper_clothing'] or '?'} top / "
              f"{r['lower_clothing'] or '?'} bottom, {r['position_in_frame']} of frame, "
              f"{os.path.basename(r['video'])}")
    print(f"Report: {os.path.join(out, 'report.html')}")


def cmd_media(a):
    store = Store(a.db)
    for m in store.media():
        print(f"[{m['id']}] {m['path']}  {fmt_offset(m['frames'] / m['fps'] if m['kind'] == 'video' else 0)}  "
              f"plates={'done' if m['plates_done'] else m['plates_next_frame']}  "
              f"persons={'done' if m['persons_done'] else m['persons_next_frame']}  "
              f"clock_start={m['clock_start'] or '-'}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="targetcapture", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="targetcapture.db", help="index database (default: %(default)s)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("index", help="analyse videos/images and store every plate and person")
    s.add_argument("media", nargs="+")
    s.add_argument("--plates", action="store_true", help="only index plates")
    s.add_argument("--persons", action="store_true", help="only index persons")
    s.add_argument("--plate-fps", type=float, default=4.0, help="frames/second analysed for plates")
    s.add_argument("--person-fps", type=float, default=3.0, help="frames/second analysed for persons")
    s.add_argument("--clock-start", help='DVR wall-clock time of the first frame, e.g. "2026-09-30 08:00:00"')
    s.add_argument("--min-ocr-conf", type=float, default=0.5)
    s.add_argument("--vehicle-crops", action="store_true",
                   help="also read plates inside each detected vehicle (slower, better for wide 4K views)")
    s.add_argument("--plate-script", choices=["auto", "latin", "arabic"], default="auto",
                   help="auto (default): English plates with the Latin reader, Arabic plates with the Arabic "
                        "reader; latin: English plates only (fastest); arabic: Arabic reader for every plate")
    s.add_argument("--no-car-model", action="store_true", help="skip make/model recognition (faster)")
    s.add_argument("--no-motion-skip", action="store_true", help="analyse sampled frames even when nothing moves")
    s.add_argument("--reindex", action="store_true")
    s.set_defaults(fn=cmd_index)

    s = sub.add_parser("plates", help="list every plate sighting in the index")
    s.add_argument("--min-reads", type=int, default=2, help="ignore plates read fewer times (filters OCR junk)")
    s.add_argument("--gap", type=float, default=4.0)
    s.add_argument("--vehicle", help='only vehicles matching these words, e.g. "corolla" or "white toyota"')
    s.set_defaults(fn=cmd_plates)

    s = sub.add_parser("find-plate", help="search for a plate number")
    s.add_argument("plate")
    s.add_argument("--max-dist", type=int, default=1, help="allowed character differences (default 1)")
    s.add_argument("--min-reads", type=int, default=1)
    s.add_argument("--gap", type=float, default=4.0, help="seconds between reads that still count as one sighting")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_find_plate)

    s = sub.add_parser("find-person", help="search for a person by face photo(s)")
    s.add_argument("photos", nargs="+")
    s.add_argument("--threshold", type=float, default=0.40, help="face similarity needed for a match (0-1)")
    s.add_argument("--merge-gap", type=float, default=3.0, help="merge matches this close in time (seconds)")
    s.add_argument("--min-face", type=int, default=20, help="ignore faces smaller than this many pixels wide")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_find_person)

    s = sub.add_parser("media", help="list indexed media")
    s.set_defaults(fn=cmd_media)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        a.fn(a)
    except KeyboardInterrupt:
        print("\nStopped. Progress is saved; run the same command again to resume.")
        sys.exit(130)


if __name__ == "__main__":
    main()
