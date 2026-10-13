"""Indexing pipeline shared by the CLI and the web UI.

The video is decoded once. Each sampled frame goes to the plate reader and the people finder,
which run side by side in their own threads (the graphics card and the processor work on both
at the same time), while this thread saves what they find.
"""
from __future__ import annotations

import os
import queue
import threading
import time
from typing import Callable

import cv2

from . import persons as P
from . import plates as L
from .store import Store
from .video import fix_seeking, frame_signature, iter_frames, probe

# Loaded models are kept between videos (and live-camera segments): loading takes seconds.
_INDEXERS: dict = {}

# progress(kind, frame_index, total_frames, found, video_seconds)
Progress = Callable[[str, int, int, int, float], None]

_END = object()
_LIVE: dict = {}  # the run in progress: "timings" -> seconds per step so far


def live_timing() -> dict:
    """Seconds per step of the indexing run in progress ({} when none)."""
    fn = _LIVE.get("timings")
    return fn() if fn else {}


class _Worker(threading.Thread):
    """Runs one indexer on its frames, in order, and hands the results back."""

    def __init__(self, kind: str, indexer, start_frame: int, video: bool, out: queue.Queue,
                 stop: threading.Event, keep: bool = True):
        super().__init__(daemon=True, name=f"index-{kind}")
        self.kind, self.indexer, self.start_frame, self.video, self.keep = kind, indexer, start_frame, video, keep
        self.inbox: queue.Queue = queue.Queue(maxsize=8)
        self.out, self.stop = out, stop
        self.busy = 0.0

    def put(self, frame) -> bool:
        while not self.stop.is_set():
            try:
                self.inbox.put(frame, timeout=0.2)
                return True
            except queue.Full:
                pass
        return False

    def run(self):
        try:
            while not self.stop.is_set():
                try:
                    frame = self.inbox.get(timeout=0.2)
                except queue.Empty:
                    continue
                if frame is _END:
                    break
                if frame.index < self.start_frame:
                    continue
                t0 = time.perf_counter()
                rows = self.indexer.process(frame)
                vehicles = self.indexer.drain_vehicles() if hasattr(self.indexer, "drain_vehicles") else []
                sig = frame_signature(frame.image) if (rows or vehicles) and self.video else None
                # keep a picture of frames with a plate or a face picture: results then show at
                # once and exactly, without finding the frame in the video again
                keep = self.keep and self.video and any(
                    self.kind == "plates" or r.get("face_jpg") for r in rows)
                jpg = cv2.imencode(".jpg", frame.image, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes() if keep else None
                self.busy += time.perf_counter() - t0
                self.out.put((self.kind, frame.index, frame.t, rows, vehicles, sig, jpg))
        except BaseException as e:  # handed to the saving thread, which stops the run
            self.out.put((self.kind, e))
            return
        self.out.put((self.kind, _END))


class Detector:
    """One detector run per frame for both people and vehicles (they are the same model), instead
    of one run for the plate reader and another for the people finder."""

    def __init__(self, model="yolo11n.pt"):
        from ultralytics import YOLO

        self.yolo = YOLO(model)

    def __call__(self, img) -> dict:
        from .gpu import half

        res = self.yolo.predict(img, classes=[0, *L.VEHICLE_CLASSES], conf=0.3, verbose=False, **half())[0]
        boxes = res.boxes.cpu().numpy()
        cls, conf = boxes.cls.astype(int), boxes.conf
        vehicles = [(L.VEHICLE_CLASSES[c], [int(v) for v in b])
                    for b, c in zip(boxes.xyxy, cls) if c in L.VEHICLE_CLASSES]
        return {"persons": boxes[(cls == 0) & (conf >= 0.35)], "vehicles": vehicles}


def _feed(path, rates, start, motion_skip, hw, workers, stop, stats, out):
    """Decode the video once and give each frame to the workers it is sampled for."""
    try:
        t0 = time.perf_counter()
        for frame in iter_frames(path, rates, start_frame=start, motion_skip=motion_skip, prefetch=0, hw=hw):
            stats["decode"] += time.perf_counter() - t0
            if not workers["detect"].put(frame):
                return
            t0 = time.perf_counter()
    except BaseException as e:
        out.put(("decode", e))
    finally:
        workers["detect"].put(_END)


class _Detect(threading.Thread):
    """Runs the shared detector on each frame (in order) and passes the frame on."""

    def __init__(self, detector, workers: dict, out: queue.Queue, stop: threading.Event):
        super().__init__(daemon=True, name="index-detect")
        self.detector, self.workers, self.out, self.stop = detector, workers, out, stop
        self.inbox: queue.Queue = queue.Queue(maxsize=8)
        self.busy = 0.0

    put = _Worker.put

    def run(self):
        try:
            while not self.stop.is_set():
                try:
                    frame = self.inbox.get(timeout=0.2)
                except queue.Empty:
                    continue
                if frame is _END:
                    break
                if self.detector is not None:
                    t0 = time.perf_counter()
                    frame.dets = self.detector(frame.image)
                    self.busy += time.perf_counter() - t0
                for kind in frame.kinds:
                    if not self.workers[kind].put(frame):
                        return
        except BaseException as e:
            self.out.put(("detect", e))
        finally:
            for w in self.workers.values():
                w.put(_END)


def index_passes(store: Store, media_id: int, info, indexers: dict, rates: dict,
                 motion_skip: bool = True, progress: Progress | None = None, detector=None) -> dict:
    """Index the kinds in `indexers` ({"plates": PlateIndexer, "persons": PersonIndexer}) in one
    decode of the video. Resumes each kind where it stopped. Returns {kind: detections found
    (-1 if it was already done)} and "timing" (seconds per step)."""
    m = store.media(media_id)
    result = {k: -1 for k in indexers if m[f"{k}_done"]}
    todo = {k: ix for k, ix in indexers.items() if k not in result}
    if not todo:
        return result
    total = info.frames if info.kind == "video" else 1
    nxt = {k: m[f"{k}_next_frame"] for k in todo}
    add = {"plates": store.add_plate_reads, "persons": store.add_person_dets}
    found = {k: 0 for k in todo}
    pending: dict[str, list] = {k: [] for k in todo}
    from .settings import load

    hw = store.decoder(media_id, "gpu" if load()["gpu_decode"] else "cpu") == "gpu"
    out: queue.Queue = queue.Queue()
    stop = threading.Event()
    keep = load()["keep_frames"]
    kept_dir = store.kept_dir(media_id)
    workers = {k: _Worker(k, ix, nxt[k], info.kind == "video", out, stop, keep) for k, ix in todo.items()}
    # both kinds: detect people and vehicles in one run per frame
    detect = _Detect(detector if len(todo) > 1 else None, workers, out, stop)
    stats = {"decode": 0.0, "saving": 0.0}
    feeder = threading.Thread(target=_feed, daemon=True, name="index-decode",
                              args=(info.path, {k: rates[k] for k in todo}, min(nxt.values()), motion_skip,
                                    hw, {"detect": detect}, stop, stats, out))

    def timings() -> dict:
        timing = {"decoding video": stats["decode"], "saving": stats["saving"]}
        if detect.busy:
            timing["people + vehicle detector"] = detect.busy
        for ix in todo.values():
            timing.update(getattr(ix, "timing", {}))
        timing["total"] = time.perf_counter() - started
        return timing

    _LIVE["timings"] = timings
    started = time.perf_counter()

    def save():
        t0 = time.perf_counter()
        for k in todo:
            add[k](media_id, pending[k])
            pending[k] = []
            store.set_progress(media_id, k, nxt[k])  # commits; safe to stop and resume
        stats["saving"] += time.perf_counter() - t0

    last_commit, running = time.time(), set(todo)
    for w in (*workers.values(), detect):
        w.start()
    feeder.start()
    try:
        while running:
            try:
                item = out.get(timeout=0.5)
            except queue.Empty:
                continue
            if isinstance(item[1], BaseException):
                raise item[1]
            if item[1] is _END:
                running.discard(item[0])
                continue
            kind, index, t, rows, vehicles, sig, jpg = item
            if jpg is not None:
                t0 = time.perf_counter()
                os.makedirs(kept_dir, exist_ok=True)
                f = os.path.join(kept_dir, f"{index:08d}.jpg")
                if not os.path.exists(f):
                    with open(f, "wb") as fh:
                        fh.write(jpg)
                stats["saving"] += time.perf_counter() - t0
            if vehicles:
                store.add_vehicle_dets(media_id, vehicles)
            if sig is not None:
                store.add_frame_sig(media_id, index, sig)
            pending[kind] += rows
            found[kind] += len(rows)
            nxt[kind] = index + 1
            if time.time() - last_commit > 10:
                if store.media(media_id) is None:  # removed from the index while being indexed
                    store.db.rollback()
                    return {**result, **found}
                save()
                last_commit = time.time()
            if progress:
                progress(kind, index + 1, total, found[kind], t)
    except BaseException:
        # Stopped (Stop button, Ctrl+C) or failed: keep everything found so far and release the
        # database, so the next run can search it and continue from here.
        stop.set()
        try:
            save()
        except Exception:
            store.db.rollback()
        raise
    finally:
        stop.set()
        for th in (feeder, detect, *workers.values()):  # models must be free before the next video
            th.join()
    for k in todo:
        add[k](media_id, pending[k])
        store.set_progress(media_id, k, total, done=True)
        if progress:
            progress(k, total, total, found[k], info.duration)
    _LIVE.clear()
    return {**result, **found, "timing": timings()}


def timing_lines(timing: dict, video_seconds: float) -> list[str]:
    """Speed of a finished run and where its time went, for the screen."""
    total = timing.get("total") or 0.0
    if total <= 0:
        return []
    lines = [f"Indexed {video_seconds / 60:.1f} min of video in {total / 60:.1f} min: "
             f"{video_seconds / total:.1f}x real time (10 h of video in about {10 * total / max(video_seconds, 1e-6):.1f} h)."]
    line = step_line(timing)
    if line:
        lines.append(line)
    return lines


def step_line(timing: dict) -> str:
    total = timing.get("total") or 0.0
    steps = sorted(((k, v) for k, v in timing.items() if k != "total" and v >= 0.05), key=lambda kv: -kv[1])
    if not steps or total <= 0:
        return ""
    return ("Time per step (side by side, so they add up to more than the total): "
            + ", ".join(f"{k} {v:.0f} s ({100 * v / total:.0f}%)" for k, v in steps))


def index_pass(store: Store, media_id: int, info, kind: str, indexer, fps: float,
               motion_skip: bool = True, progress: Progress | None = None) -> int:
    """One kind on its own (kept for callers that index plates or people separately)."""
    return index_passes(store, media_id, info, {kind: indexer}, {kind: fps}, motion_skip, progress)[kind]


def index_media(store: Store, path: str, plates: bool = True, persons: bool = True,
                plate_fps: float = 4.0, person_fps: float = 3.0, clock_start: str | None = None,
                min_ocr_conf: float = 0.5, vehicle_crops: bool = False, plate_script: str = "auto", car_models: bool = True, motion_skip: bool = True,
                reindex: bool = False, progress: Progress | None = None, skip_empty_road: bool = False,
                model_looks: int = 4) -> dict:
    # DVR exports that cannot jump to an exact frame are copied into a fixed .mp4 first
    path = fix_seeking(path, os.path.join(os.path.dirname(os.path.abspath(store.path)), "fixed_videos"))
    info = probe(path)
    mid = store.upsert_media(info, clock_start)
    result = {"media_id": mid, "info": info}
    indexers, rates = {}, {}
    if plates:
        if reindex:
            store.reset(mid, "plates")
        from .teach import Memory

        if car_models is True:  # which recognizer: Settings tab ("auto" = accurate on GPU, fast on CPU)
            from .settings import load

            car_models = load()["car_model_size"]
        key = ("plates", min_ocr_conf, vehicle_crops, plate_script, car_models)
        if key not in _INDEXERS:
            _INDEXERS[key] = L.PlateIndexer(min_ocr_conf=min_ocr_conf, vehicle_crops=vehicle_crops,
                                            script=plate_script, car_models=car_models)
        idx = _INDEXERS[key]
        idx.start(memory=Memory(store), track_offset=store.max_vehicle_track(mid) + 1, skip_empty=skip_empty_road)
        idx.max_model_looks = model_looks
        indexers["plates"], rates["plates"] = idx, plate_fps
    if persons:
        if reindex:
            store.reset(mid, "persons")
        if "persons" not in _INDEXERS:
            _INDEXERS["persons"] = P.PersonIndexer()
        idx = _INDEXERS["persons"]
        idx.start(track_offset=store.max_track_id(mid) + 1)
        indexers["persons"], rates["persons"] = idx, person_fps
    detector = None
    if len(indexers) > 1:
        if "detector" not in _INDEXERS:
            _INDEXERS["detector"] = Detector()
        detector = _INDEXERS["detector"]
    if indexers:
        result.update(index_passes(store, mid, info, indexers, rates, motion_skip, progress, detector))
    from .watch import check

    result["alerts"] = check(store, mid)  # wanted plates / faces found in this video
    return result
