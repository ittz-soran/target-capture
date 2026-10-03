"""Live cameras: record each camera (RTSP / HTTP stream) in 1-minute pieces with ffmpeg and
index every finished piece straight away, so plates, people, vehicles and watch-list alerts
appear about a minute after they happen. Recordings older than `keep_days` are deleted.

    python -m targetcapture live            # runs until stopped (Ctrl+C)
"""
from __future__ import annotations

import datetime as dt
import glob
import logging
import os
import re
import shutil
import subprocess
import time

from .store import Store, clock_time

log = logging.getLogger(__name__)
SEGMENT = 60  # seconds per recorded piece


def ffmpeg_exe() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _input_args(url: str) -> list[str]:
    if os.path.isfile(url):  # a video file played as if it were a camera (for trying things out)
        return ["-re", "-stream_loop", "-1", "-i", url]
    args = ["-rtsp_transport", "tcp"] if url.lower().startswith("rtsp") else []
    return [*args, "-i", url]


def test_camera(url: str, timeout: float = 15) -> tuple[bool, str, bytes | None]:
    """Grab one picture from the camera: (ok, message, jpeg)."""
    exe = ffmpeg_exe()
    if not exe:
        return False, "ffmpeg is not available (pip install imageio-ffmpeg)", None
    cmd = [exe, "-nostdin", "-loglevel", "error", *_input_args(url), "-frames:v", "1", "-f", "image2pipe",
           "-vcodec", "mjpeg", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "no answer from the camera (check the address, user name and password)", None
    if r.returncode or not r.stdout:
        return False, (r.stderr.decode(errors="replace").strip()[-300:] or "could not read the stream"), None
    return True, "connected", r.stdout


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "camera"


class Recorder:
    def __init__(self, cam, root: str, segment: int = SEGMENT):
        self.cam, self.segment = cam, segment
        self.dir = os.path.join(root, _slug(cam["name"]))
        self.proc: subprocess.Popen | None = None
        os.makedirs(self.dir, exist_ok=True)

    def ensure_running(self) -> bool:
        if self.proc is not None and self.proc.poll() is None:
            return True
        exe = ffmpeg_exe()
        if not exe:
            return False
        out = os.path.join(self.dir, "%Y-%m-%d_%H-%M-%S.mp4")
        cmd = [exe, "-nostdin", "-loglevel", "error", *_input_args(self.cam["url"]), "-map", "0:v:0", "-c", "copy",
               "-f", "segment", "-segment_time", str(self.segment), "-segment_format", "mp4", "-reset_timestamps", "1",
               "-strftime", "1", out]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE)
        return True

    def finished_pieces(self) -> list[str]:
        """Pieces ffmpeg has closed: all but the newest file, and none modified in the last 5 s."""
        files = sorted(glob.glob(os.path.join(self.dir, "*.mp4")))
        running = self.proc is not None and self.proc.poll() is None
        if running and files:
            files = files[:-1]
        return [f for f in files if time.time() - os.path.getmtime(f) > 5]

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def _clock_of(path: str) -> str | None:
    m = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})", os.path.basename(path))
    return f"{m[1]} {m[2]}:{m[3]}:{m[4]}" if m else None


def run(db_path: str, root: str = "recordings", keep_days: float | None = None, poll: float = 10.0,
        on_alert=None, stop_file: str | None = None, segment: int | None = None, max_loops: int | None = None):
    """Record and index all enabled cameras until stopped (Ctrl+C, or `stop_file` appears).
    keep_days, segment and the indexing options not given come from the Settings tab."""
    from . import settings
    from .pipeline import index_media

    cfg = settings.load()
    keep_days = cfg["keep_days"] if keep_days is None else keep_days
    segment = int(cfg["segment_seconds"]) if segment is None else segment
    opts = {k: cfg[k] for k in ("plate_script", "car_models", "vehicle_crops", "min_ocr_conf")}
    opts.update(settings.rates(cfg))  # speed choice of the Settings tab
    opts["skip_empty_road"] = cfg["skip_empty_road"]

    store = Store(db_path)
    recorders: dict[int, Recorder] = {}
    done: set[str] = {m["path"] for m in store.media() if m["plates_done"] or m["persons_done"]}
    try:
        while not (stop_file and os.path.exists(stop_file)):
            cams = {c["id"]: c for c in store.cameras(enabled_only=True)}
            for cid in list(recorders):
                if cid not in cams or recorders[cid].cam["url"] != cams[cid]["url"]:
                    recorders.pop(cid).stop()
            for cid, cam in cams.items():
                rec = recorders.setdefault(cid, Recorder(cam, root, segment))
                was_running = rec.proc is not None and rec.proc.poll() is None
                rec.ensure_running()
                now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if rec.proc.poll() is None:
                    store.update_camera(cid, status="recording", last_seen=now)
                elif not was_running:
                    err = rec.proc.stderr.read().decode(errors="replace")[-200:] if rec.proc.stderr else ""
                    store.update_camera(cid, status=f"not connected: {err.strip() or 'stream ended'}")
                for piece in rec.finished_pieces():
                    full = os.path.abspath(piece)
                    if full in done:
                        continue
                    beat = [0.0]

                    def heartbeat(*_):  # keep the page showing "running" while a piece is indexed
                        if time.time() - beat[0] > 10:
                            beat[0] = time.time()
                            stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            for c in cams:
                                store.update_camera(c, last_seen=stamp)

                    try:
                        res = index_media(store, full, plates=bool(cam["plates"]), persons=bool(cam["persons"]),
                                          clock_start=_clock_of(full), progress=heartbeat, **opts)
                    except Exception as e:  # a broken piece must not stop the cameras
                        log.warning("could not index %s: %s", full, e)
                        res = {}
                    done.add(full)
                    for a in res.get("alerts", []):
                        at = clock_time(store.media(a["media_id"]), a["start"]) or _clock_of(full)
                        msg = f"[{cam['name']}] {a['kind']} {a['value']} matched ({a['text']}) at {at}"
                        log.warning("ALERT %s", msg)
                        if on_alert:
                            on_alert(msg, a)
            _cleanup(store, root, keep_days)
            if max_loops is not None:
                max_loops -= 1
                if max_loops <= 0:
                    break
            time.sleep(poll)
    finally:
        for cid, rec in recorders.items():
            rec.stop()
            store.update_camera(cid, status="stopped", last_seen=None)
        if stop_file and os.path.exists(stop_file):
            os.remove(stop_file)


def _cleanup(store, root: str, keep_days: float):
    """Delete recordings older than keep_days (and forget them in the index), except pieces
    with a watch-list alert, which are kept as evidence."""
    limit = time.time() - keep_days * 86400
    with_alerts = {a["media_id"] for a in store.alerts()}
    for m in store.media():
        p = m["path"]
        if p.startswith(os.path.abspath(root) + os.sep) and os.path.exists(p) and os.path.getmtime(p) < limit \
                and m["id"] not in with_alerts:
            store.remove_media(m["id"])
            os.remove(p)
