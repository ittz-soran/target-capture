"""Live cameras and DVR / NVR channels (RTSP / HTTP streams), analysed as the picture arrives:
plates, people, vehicles and watch-list alerts appear a second or two after they happen. The
picture of every moment with a detection is kept (live_frames/), so results show the exact
frame; no recording is needed. Optionally each camera is also recorded in pieces with ffmpeg
for watching later (Settings: live_record). Pictures and recordings older than `keep_days`
are deleted, except days with a watch-list alert.

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
import threading
import time

import cv2
import numpy as np

from .store import Store

log = logging.getLogger(__name__)
# cameras over TCP: no grey smears from lost packets on busy networks
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
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


def _creds(user: str, password: str) -> list[str]:
    """user:password as it goes into a stream address. Only the characters that would break
    the address are encoded first (many DVRs and ffmpeg builds do not decode the rest, e.g. a
    password with "!" sent as "%21" is refused); then everything encoded."""
    from urllib.parse import quote

    light = f"{quote(user, safe='!$&()*+,;=~')}:{quote(password, safe='!$&()*+,;=~')}"
    full = f"{quote(user, safe='')}:{quote(password, safe='')}"
    return [light] if light == full else [light, full]


def dvr_urls(make: str, ip: str, port: int, user: str, password: str) -> list[tuple[str, object]]:
    """Stream addresses to try for a DVR / NVR, most likely first: [(label, channel -> url)].
    The chosen make's forms come first, then the other make's (some DVRs are rebranded)."""
    out = []
    for i, cred in enumerate(_creds(user, password)):
        base = f"rtsp://{cred}@{ip}:{port}"
        note = "" if i == 0 else ", password written another way"
        hik = [(f"Hikvision main stream{note}", lambda ch, b=base: f"{b}/Streaming/Channels/{ch}01"),
               (f"Hikvision IP-camera channels (from 33){note}", lambda ch, b=base: f"{b}/Streaming/Channels/{ch + 32}01"),
               (f"Hikvision older form{note}", lambda ch, b=base: f"{b}/h264/ch{ch}/main/av_stream")]
        dahua = [(f"Dahua main stream{note}", lambda ch, b=base: f"{b}/cam/realmonitor?channel={ch}&subtype=0")]
        out.append(hik + dahua if make.lower().startswith("hik") else dahua + hik)
    return out


def check_login(make: str, ip: str, user: str, password: str) -> tuple[bool | None, str]:
    """Ask the DVR's own web login whether the user name and password are right (the same check
    as logging in to its web page). (True, model) / (False, reason) / (None, '') if it cannot tell."""
    import requests
    from requests.auth import HTTPDigestAuth

    paths = (["/ISAPI/System/deviceInfo"] if make.lower().startswith("hik")
             else ["/cgi-bin/magicBox.cgi?action=getDeviceType"])
    for path in paths:
        try:
            r = requests.get(f"http://{ip}{path}", auth=HTTPDigestAuth(user, password), timeout=6)
        except requests.RequestException:
            return None, ""
        if r.status_code == 200:
            import re as _re

            m = _re.search(r"<model>([^<]+)</model>|type=(\S+)", r.text)
            return True, (m.group(1) or m.group(2)) if m else ""
        if r.status_code == 401:
            locked = "lock" in r.text.lower()
            return False, ("the DVR has locked this user for a while after wrong passwords; wait 30 minutes "
                           "or restart the DVR" if locked else "wrong user name or password")
    return None, ""


def refused_login(msg: str) -> bool:
    """The camera answered and turned the user name / password down."""
    return "unauthorized" in (msg or "").lower()


def explain_error(msg: str) -> str:
    """ffmpeg's message in plain words (the codes are looked for in words only: addresses and
    memory numbers in the message, like 0x55d01a401540, contain digits too)."""
    m = re.sub(r"rtsp://\S+|0x[0-9a-f]+", " ", (msg or "").lower())
    if "unauthorized" in m:
        return "wrong user name or password (or that user may not watch live video)."
    if "404" in m or "not found" in m:
        return "the DVR has no such channel at this address."
    if "refused" in m:
        return "the DVR refused the connection: check the RTSP port (Network settings of the DVR, normally 554)."
    if "no answer" in m or "timed out" in m or "timeout" in m or "unreachable" in m:
        return ("no answer: check the IP address, that this PC is on the same network, and that RTSP is "
                "allowed in the DVR's network settings.")
    return "could not read the stream."


def hik_rtsp_port(ip: str, user: str, password: str) -> int | None:
    """The RTSP port a Hikvision DVR / camera is set to (its network settings), or None."""
    import requests
    from requests.auth import HTTPDigestAuth

    try:
        r = requests.get(f"http://{ip}/ISAPI/Security/adminAccesses", auth=HTTPDigestAuth(user, password), timeout=6)
    except requests.RequestException:
        return None
    m = re.search(r"<protocol>\s*RTSP\s*</protocol>\s*<portNo>\s*(\d+)", r.text, re.I) if r.ok else None
    return int(m.group(1)) if m else None


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


LIVE_FPS = 25.0   # frame numbers of live pictures: seconds since midnight x 25
FRAMES = "live_frames"

# what the plate reader and people finder remember about one camera (tracks, faces seen...);
# swapped in and out so one set of models serves every camera
PLATE_STATE = ("_tracker", "_tracks", "vehicle_rows", "_t", "_latin_seen", "memory", "skip_empty",
               "track_offset", "timing", "max_model_looks")
PERSON_STATE = ("_bt", "_face_try", "_faces_seen", "_untracked", "track_offset", "timing")


class StreamReader(threading.Thread):
    """Keeps the newest picture of one camera, reconnecting by itself."""

    def __init__(self, url: str, hw: bool):
        super().__init__(daemon=True, name="live-reader")
        self.url, self.hw = url, hw
        self.stop_ev, self.want = threading.Event(), threading.Event()
        self.lock = threading.Lock()
        self.latest = None   # (picture, wall-clock time)
        self.status = "connecting"
        self.want.set()

    def take(self):
        """The newest picture not taken yet, or None."""
        with self.lock:
            got, self.latest = self.latest, None
        self.want.set()
        return got

    def run(self):
        from .video import open_video

        is_file = os.path.isfile(self.url)  # a video file played as if it were a camera (trying things out)
        while not self.stop_ev.is_set():
            cap = open_video(self.url, self.hw)
            if not cap.isOpened():
                self.status = "not connected (check the address, user name and password)"
                cap.release()
                self.stop_ev.wait(5)
                continue
            self.status = "live"
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            t_next, fails = time.time(), 0
            while not self.stop_ev.is_set():
                if is_file:  # play at normal speed
                    t_next += 1.0 / fps
                    time.sleep(max(0.0, t_next - time.time()))
                if not cap.grab():
                    fails += 1
                    if is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    if fails > 50:
                        break
                    continue
                fails = 0
                if self.want.is_set():  # only turn into a picture when one is wanted
                    ok, img = cap.retrieve()
                    if ok:
                        with self.lock:
                            self.latest = (img, time.time())
                        self.want.clear()
            cap.release()
            if not self.stop_ev.is_set():
                self.status = "connection lost, reconnecting"
                self.stop_ev.wait(2)

    def stop(self):
        self.stop_ev.set()


def _state(ix, names) -> dict:
    return {k: getattr(ix, k) for k in names if hasattr(ix, k)}


def _restore(ix, state: dict):
    for k, v in state.items():
        setattr(ix, k, v)


def _imwrite(path: str, img, quality: int = 88):
    """cv2.imwrite fails on some Windows folder names; encoding first works everywhere."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(buf.tobytes())
        try:
            os.replace(tmp, path)
        except OSError:  # the page is reading it right now (Windows): next time
            pass


class LiveCamera:
    def __init__(self, cam, store: Store, frames_root: str, hw: bool):
        self.cam, self.store, self.root = cam, store, frames_root
        self.reader = StreamReader(cam["url"], hw)
        self.reader.start()
        self.day = None
        self.media_id = None
        self.dir = None
        self.last = {"plates": 0.0, "persons": 0.0}
        self.states: dict[str, dict] = {}
        self.analysed: list[float] = []   # times of analysed pictures (speed shown on the page)
        self.last_sig, self.last_t = None, 0.0
        self.recent_alerts: dict[int, float] = {}

    def ensure_day(self, now: dt.datetime, plate_ix, person_ix, cfg, img):
        """One index entry per camera per day; a new day starts fresh tracks."""
        day = now.strftime("%Y-%m-%d")
        if day == self.day:
            return
        from .teach import Memory
        from .video import MediaInfo

        self.day = day
        self.dir = os.path.abspath(os.path.join(self.root, f"{_slug(self.cam['name'])}_{day}"))
        os.makedirs(self.dir, exist_ok=True)
        info = MediaInfo(self.dir, "video", LIVE_FPS, 0, img.shape[1], img.shape[0])
        self.media_id = self.store.upsert_media(info, f"{day} 00:00:00")
        self.states = {}
        if plate_ix is not None:
            plate_ix.start(memory=Memory(self.store), track_offset=self.store.max_vehicle_track(self.media_id) + 1,
                           skip_empty=cfg["skip_empty_road"])
            plate_ix.max_model_looks = cfg["model_looks"]
            self.states["plates"] = _state(plate_ix, PLATE_STATE)
        if person_ix is not None:
            person_ix.start(track_offset=self.store.max_track_id(self.media_id) + 1)
            self.states["persons"] = _state(person_ix, PERSON_STATE)

    def stop(self):
        self.reader.stop()
        self.reader.join(timeout=10)  # release the stream before the program ends


def _models(cfg, cams):
    """Load the models once for every camera (they are the slow part to load)."""
    from . import persons as P
    from . import plates as L
    from .pipeline import _INDEXERS, Detector

    plate_ix = person_ix = None
    if any(c["plates"] for c in cams):
        car = cfg["car_model_size"] if cfg["car_models"] else False
        key = ("plates", cfg["min_ocr_conf"], cfg["vehicle_crops"], cfg["plate_script"], car)
        if key not in _INDEXERS:
            _INDEXERS[key] = L.PlateIndexer(min_ocr_conf=cfg["min_ocr_conf"], vehicle_crops=cfg["vehicle_crops"],
                                            script=cfg["plate_script"], car_models=car)
        plate_ix = _INDEXERS[key]
    if any(c["persons"] for c in cams):
        if "persons" not in _INDEXERS:
            _INDEXERS["persons"] = P.PersonIndexer()
        person_ix = _INDEXERS["persons"]
    if "detector" not in _INDEXERS:
        _INDEXERS["detector"] = Detector()
    return plate_ix, person_ix, _INDEXERS["detector"]


def _draw(img, plate_rows, person_rows, vehicles):
    out = img.copy()
    for v in vehicles:
        cv2.rectangle(out, (v["x1"], v["y1"]), (v["x2"], v["y2"]), (200, 160, 0), 2)
    for r in person_rows:
        cv2.rectangle(out, (r["x1"], r["y1"]), (r["x2"], r["y2"]), (0, 200, 0), 2)
        if r.get("fx1") is not None:
            cv2.rectangle(out, (r["fx1"], r["fy1"]), (r["fx2"], r["fy2"]), (0, 255, 255), 2)
    for r in plate_rows:
        cv2.rectangle(out, (r["x1"], r["y1"]), (r["x2"], r["y2"]), (0, 0, 255), 3)
        if r["text"]:
            cv2.putText(out, r["text"], (r["x1"], max(20, r["y1"] - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.8, out.shape[1] / 1600), (0, 0, 255), 2)
    if out.shape[1] > 960:
        out = cv2.resize(out, (960, int(out.shape[0] * 960 / out.shape[1])), interpolation=cv2.INTER_AREA)
    return out


def _check_watch(store, lc: LiveCamera, watch, plate_rows, person_rows, cfg, on_alert):
    """Watch list on the newest results only (instant, however big the index gets)."""
    from . import plates as L

    now = time.time()
    hits = []
    for w in watch:
        if now - lc.recent_alerts.get(w["id"], 0) < 60:  # one alert a minute per wanted plate / face
            continue
        if w["kind"] == "plate":
            for r in plate_rows:
                if r["text"] and L.match_score(w["value"], r["text"], int(cfg["watch_plate_dist"]))[0]:
                    hits.append((w, r, r["text"], 1.0, ("x1", "y1", "x2", "y2")))
                    break
        elif w["emb"] is not None:
            e = np.frombuffer(w["emb"], np.float32)
            e = e / (np.linalg.norm(e) + 1e-9)
            for r in person_rows:
                if r.get("emb") is None:
                    continue
                f = np.asarray(r["emb"], np.float32)
                sim = float(f @ e / (np.linalg.norm(f) + 1e-9))
                if sim >= cfg["face_threshold"]:
                    hits.append((w, r, f"face {sim:.0%} alike", sim, ("fx1", "fy1", "fx2", "fy2")))
                    break
    for w, r, text, score, box in hits:
        lc.recent_alerts[w["id"]] = now
        a = {"watch_id": w["id"], "media_id": lc.media_id, "start": r["t"], "end": r["t"], "text": text,
             "score": score, "frame": r["frame"], **{k: r[b] for k, b in zip(("x1", "y1", "x2", "y2"), box)}}
        if store.add_alert(a):
            at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = f"[{lc.cam['name']}] {w['kind']} {w['value']} matched ({text}) at {at}"
            log.warning("ALERT %s", msg)
            if on_alert:
                on_alert(msg, {**a, "kind": w["kind"], "value": w["value"]})


def run(db_path: str, root: str = "recordings", keep_days: float | None = None, poll: float = 10.0,
        on_alert=None, stop_file: str | None = None, segment: int | None = None, max_loops: int | None = None,
        max_seconds: float | None = None):
    """Analyse all enabled cameras live until stopped (Ctrl+C, or `stop_file` appears).
    keep_days, segment and the indexing options not given come from the Settings tab."""
    from . import settings
    from .video import Frame, _motion_signature

    cfg = settings.load()
    cfg.update(settings.rates(cfg))
    keep_days = cfg["keep_days"] if keep_days is None else keep_days
    segment = int(cfg["segment_seconds"]) if segment is None else segment
    frames_root = os.path.join(os.path.dirname(os.path.abspath(db_path)), FRAMES)
    preview_dir = os.path.join(frames_root, "_now")
    os.makedirs(preview_dir, exist_ok=True)

    store = Store(db_path)
    cams_rows = store.cameras(enabled_only=True)
    plate_ix, person_ix, detector = _models(cfg, cams_rows)
    live: dict[int, LiveCamera] = {}
    recorders: dict[int, Recorder] = {}
    watch, watch_at, status_at, commit_at, cams_at, clean_at = [], 0.0, 0.0, time.time(), 0.0, 0.0
    began = time.time()
    try:
        while not (stop_file and os.path.exists(stop_file)):
            now = time.time()
            if max_seconds is not None and now - began > max_seconds:
                break
            if now - cams_at > 5:  # cameras added, removed or switched off in the page
                cams_at = now
                cams = {c["id"]: c for c in store.cameras(enabled_only=True)}
                for cid in list(live):
                    if cid not in cams or live[cid].cam["url"] != cams[cid]["url"]:
                        live.pop(cid).stop()
                for cid in list(recorders):
                    if cid not in cams or not cfg["live_record"]:
                        recorders.pop(cid).stop()
                if any(cid not in live for cid in cams):
                    plate_ix, person_ix, detector = _models(cfg, list(cams.values()))
                for cid, cam in cams.items():
                    if cid not in live:
                        live[cid] = LiveCamera(cam, store, frames_root, cfg["gpu_decode"])
                    else:
                        live[cid].cam = cam
                    if cfg["live_record"]:
                        recorders.setdefault(cid, Recorder(cam, root, segment)).ensure_running()
            if now - watch_at > 10:
                watch, watch_at = store.watchlist(active_only=True), now
            busy = False
            for lc in list(live.values()):
                cam = lc.cam
                due = [k for k, on, fps in (("plates", cam["plates"] and plate_ix, cfg["plate_fps"]),
                                            ("persons", cam["persons"] and person_ix, cfg["person_fps"]))
                       if on and now - lc.last[k] >= 1.0 / fps]
                if not due:
                    continue
                got = lc.reader.take()
                if got is None:
                    continue
                img, wall = got
                busy = True
                for k in due:
                    lc.last[k] = now
                stamp = dt.datetime.fromtimestamp(wall)
                lc.ensure_day(stamp, plate_ix, person_ix, cfg, img)
                t = (stamp - stamp.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
                # nothing moved: skip (but still look every 5 s)
                sig = _motion_signature(img)
                if lc.last_sig is not None and float(np.abs(sig - lc.last_sig).mean()) < 1.5 and t - lc.last_t < 5:
                    continue
                lc.last_sig, lc.last_t = sig, t
                frame = Frame(int(round(t * LIVE_FPS)), t, img, tuple(due), detector(img))
                plate_rows, person_rows, vehicles = [], [], []
                if "plates" in due:
                    _restore(plate_ix, lc.states["plates"])
                    plate_rows = plate_ix.process(frame)
                    vehicles = plate_ix.drain_vehicles()
                    lc.states["plates"] = _state(plate_ix, PLATE_STATE)
                if "persons" in due:
                    _restore(person_ix, lc.states["persons"])
                    person_rows = person_ix.process(frame)
                    lc.states["persons"] = _state(person_ix, PERSON_STATE)
                lc.analysed = [x for x in lc.analysed if now - x < 10] + [now]
                if plate_rows or person_rows or vehicles:
                    _imwrite(os.path.join(lc.dir, f"{frame.index:08d}.jpg"), img)
                    store.add_plate_reads(lc.media_id, plate_rows)
                    store.add_person_dets(lc.media_id, person_rows)
                    if vehicles:
                        store.add_vehicle_dets(lc.media_id, vehicles)
                    if watch:
                        _check_watch(store, lc, watch, plate_rows, person_rows, cfg, on_alert)
                _imwrite(os.path.join(preview_dir, f"{cam['id']}.jpg"), _draw(img, plate_rows, person_rows, vehicles),
                         80)
            now = time.time()
            if now - commit_at > 2:
                for lc in live.values():
                    if lc.media_id is not None and lc.last_t:
                        store.db.execute("UPDATE media SET frames=MAX(frames, ?) WHERE id=?",
                                         (int(lc.last_t * LIVE_FPS), lc.media_id))
                store.db.commit()
                commit_at = now
            if now - status_at > 5:
                status_at = now
                stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for cid, lc in live.items():
                    st = lc.reader.status
                    if st == "live":
                        rate = len(lc.analysed) / 10.0
                        st = f"live · analysing {rate:.1f} pictures/s"
                    store.update_camera(cid, status=st, last_seen=stamp)
            if now - clean_at > 600:
                clean_at = now
                _cleanup(store, root, keep_days)
                _cleanup_frames(store, frames_root, keep_days)
            if max_loops is not None:
                max_loops -= 1
                if max_loops <= 0:
                    break
            if not busy:
                time.sleep(0.02)
    finally:
        store.db.commit()
        for lc in live.values():
            lc.stop()
        for rec in recorders.values():
            rec.stop()
        for cid in set(live) | set(recorders):
            store.update_camera(cid, status="stopped", last_seen=None)
        if stop_file and os.path.exists(stop_file):
            os.remove(stop_file)


def _cleanup_frames(store, frames_root: str, keep_days: float):
    """Forget live days older than keep_days and delete their pictures, except days with a
    watch-list alert, which are kept as evidence."""
    limit = dt.datetime.now() - dt.timedelta(days=keep_days)
    with_alerts = {a["media_id"] for a in store.alerts()}
    root = os.path.abspath(frames_root) + os.sep
    for m in store.media():
        p = m["path"]
        if not p.startswith(root) or m["id"] in with_alerts or not m["clock_start"]:
            continue
        try:
            day = dt.datetime.strptime(m["clock_start"][:10], "%Y-%m-%d")
        except ValueError:
            continue
        if day + dt.timedelta(days=1) < limit:
            store.remove_media(m["id"])
            shutil.rmtree(p, ignore_errors=True)


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
