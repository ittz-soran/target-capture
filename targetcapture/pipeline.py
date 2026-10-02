"""Indexing pipeline shared by the CLI and the web UI."""
from __future__ import annotations

import os
import time
from typing import Callable

from . import persons as P
from . import plates as L
from .store import Store
from .video import fix_seeking, frame_signature, iter_frames, probe

# Loaded models are kept between videos (and live-camera segments): loading takes seconds.
_INDEXERS: dict = {}

# progress(kind, frame_index, total_frames, found, video_seconds)
Progress = Callable[[str, int, int, int, float], None]


def index_pass(store: Store, media_id: int, info, kind: str, indexer, fps: float,
               motion_skip: bool = True, progress: Progress | None = None) -> int:
    m = store.media(media_id)
    if m[f"{kind}_done"]:
        return -1
    start = m[f"{kind}_next_frame"]
    total = info.frames if info.kind == "video" else 1
    add = store.add_plate_reads if kind == "plates" else store.add_person_dets
    found, last_commit = 0, time.time()
    pending: list[dict] = []
    next_frame = start
    try:
        for frame in iter_frames(info.path, fps, start_frame=start, motion_skip=motion_skip):
            rows = indexer.process(frame)
            vehicles = indexer.drain_vehicles() if hasattr(indexer, "drain_vehicles") else []
            if vehicles:
                store.add_vehicle_dets(media_id, vehicles)
            if (rows or vehicles) and info.kind == "video":
                store.add_frame_sig(media_id, frame.index, frame_signature(frame.image))
            pending += rows
            found += len(rows)
            next_frame = frame.index + 1
            if time.time() - last_commit > 10:
                add(media_id, pending)
                pending = []
                store.set_progress(media_id, kind, next_frame)  # commits; safe to stop and resume
                last_commit = time.time()
            if progress:
                progress(kind, frame.index + 1, total, found, frame.t)
    except BaseException:
        # Stopped (Stop button, Ctrl+C) or failed: keep everything found so far and release the
        # database, so the next run can search it and continue from here.
        try:
            add(media_id, pending)
            store.set_progress(media_id, kind, next_frame)
        except Exception:
            store.db.rollback()
        raise
    add(media_id, pending)
    store.set_progress(media_id, kind, total, done=True)
    if progress:
        progress(kind, total, total, found, info.duration)
    return found


def index_media(store: Store, path: str, plates: bool = True, persons: bool = True,
                plate_fps: float = 4.0, person_fps: float = 3.0, clock_start: str | None = None,
                min_ocr_conf: float = 0.5, vehicle_crops: bool = False, plate_script: str = "auto", car_models: bool = True, motion_skip: bool = True,
                reindex: bool = False, progress: Progress | None = None, skip_empty_road: bool = False) -> dict:
    # DVR exports that cannot jump to an exact frame are copied into a fixed .mp4 first
    path = fix_seeking(path, os.path.join(os.path.dirname(os.path.abspath(store.path)), "fixed_videos"))
    info = probe(path)
    mid = store.upsert_media(info, clock_start)
    result = {"media_id": mid, "info": info}
    if plates:
        if reindex:
            store.reset(mid, "plates")
        from .teach import Memory

        key = ("plates", min_ocr_conf, vehicle_crops, plate_script, car_models)
        if key not in _INDEXERS:
            _INDEXERS[key] = L.PlateIndexer(min_ocr_conf=min_ocr_conf, vehicle_crops=vehicle_crops,
                                            script=plate_script, car_models=car_models)
        idx = _INDEXERS[key]
        idx.start(memory=Memory(store), track_offset=store.max_vehicle_track(mid) + 1, skip_empty=skip_empty_road)
        result["plates"] = index_pass(store, mid, info, "plates", idx, plate_fps, motion_skip, progress)
    if persons:
        if reindex:
            store.reset(mid, "persons")
        if "persons" not in _INDEXERS:
            _INDEXERS["persons"] = P.PersonIndexer()
        idx = _INDEXERS["persons"]
        idx.start(track_offset=store.max_track_id(mid) + 1)
        result["persons"] = index_pass(store, mid, info, "persons", idx, person_fps, motion_skip, progress)
    from .watch import check

    result["alerts"] = check(store, mid)  # wanted plates / faces found in this video
    return result
