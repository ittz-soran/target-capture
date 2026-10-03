"""Reading frames from videos and images, with frame sampling and motion gating.

A 5 hour DVR file at 25 fps is 450,000 frames. Running detectors on every frame is
wasteful, so frames are sampled at a fixed rate (e.g. 4 per second) and, optionally,
frames where nothing changed since the last processed frame are skipped. A still
frame is still processed every `max_still` seconds so parked cars or standing
people are not missed entirely.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class Frame:
    index: int        # frame number in the source (0 for a still image)
    t: float          # seconds from the start of the video
    image: np.ndarray  # BGR


@dataclass
class MediaInfo:
    path: str
    kind: str  # "video" | "image"
    fps: float
    frames: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frames / self.fps if self.kind == "video" else 0.0


def is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def probe(path: str) -> MediaInfo:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if is_image(path):
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"cannot read image {path}")
        return MediaInfo(path, "image", 1.0, 1, img.shape[1], img.shape[0])
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:
        # DVR exports often carry broken fps metadata.
        log.warning("video reports fps=%s, assuming 25", fps)
        fps = 25.0
    info = MediaInfo(
        path,
        "video",
        float(fps),
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return info


def _motion_signature(img: np.ndarray) -> np.ndarray:
    small = cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0).astype(np.int16)


def open_video(path: str, hw: bool = True) -> cv2.VideoCapture:
    """Open a video, decoding on the graphics card when OpenCV and the driver allow it
    (NVIDIA, Intel or AMD; much less CPU on long 1080p/4K H.264/H.265 files). Falls back to
    normal decoding by itself."""
    if hw and hasattr(cv2, "VIDEO_ACCELERATION_ANY"):
        try:
            cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
            if cap.isOpened():
                if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):  # phone videos upright, like normal decoding
                    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
                return cap
            cap.release()
        except cv2.error:
            pass
    return cv2.VideoCapture(path)


def iter_frames(
    path: str,
    sample_fps: float,
    start_frame: int = 0,
    motion_skip: bool = True,
    motion_threshold: float = 1.5,
    max_still: float = 5.0,
    prefetch: int = 16,
    hw: bool = False,
) -> Iterator[Frame]:
    """Yield sampled frames. For an image, yields the image once. The video is decoded in a
    background thread, so decoding the next frames overlaps with the detectors' work."""
    info = probe(path)
    if info.kind == "image":
        if start_frame == 0:
            yield Frame(0, 0.0, cv2.imread(path))
        return
    frames = _sampled(path, info, sample_fps, start_frame, motion_skip, motion_threshold, max_still, hw)
    yield from (_prefetched(frames, prefetch) if prefetch > 0 else frames)


def _sampled(path, info, sample_fps, start_frame, motion_skip, motion_threshold, max_still, hw):
    step = max(1, round(info.fps / sample_fps))
    cap = open_video(path, hw)
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    i = start_frame
    last_sig = None
    last_t = -1e9
    try:
        while True:
            if i % step:
                if not cap.grab():  # grab() skips the colour conversion of read()
                    break
                i += 1
                continue
            ok, img = cap.read()
            if not ok:
                break
            t = i / info.fps
            if motion_skip:
                sig = _motion_signature(img)
                still = last_sig is not None and float(np.abs(sig - last_sig).mean()) < motion_threshold
                if still and t - last_t < max_still:
                    i += 1
                    continue
                last_sig = sig
            last_t = t
            yield Frame(i, t, img)
            i += 1
    finally:
        cap.release()


def _prefetched(frames: Iterator[Frame], size: int) -> Iterator[Frame]:
    """Run a frame generator in a background thread, keeping up to `size` frames ready."""
    import queue
    import threading

    q: queue.Queue = queue.Queue(maxsize=size)
    stop = threading.Event()
    end = object()

    def work():
        try:
            for f in frames:
                while not stop.is_set():
                    try:
                        q.put(f, timeout=0.5)
                        break
                    except queue.Full:
                        pass
                if stop.is_set():
                    break
        except Exception as e:  # handed to the reader below
            q.put(e)
        finally:
            frames.close()
            q.put(end)

    th = threading.Thread(target=work, daemon=True, name="frame-reader")
    th.start()
    try:
        while True:
            f = q.get()
            if f is end:
                break
            if isinstance(f, Exception):
                raise f
            yield f
    finally:
        stop.set()
        while th.is_alive():  # let the reader finish so the file is closed
            try:
                q.get(timeout=0.1)
            except queue.Empty:
                pass


SIG_W, SIG_H = 48, 27
MATCH = 3.0  # mean grey-level difference below which two frame signatures are the same frame


def frame_signature(img: np.ndarray) -> np.ndarray:
    """Tiny grey thumbnail (48x27) stored for every frame that had a detection, so the exact
    frame can be found again later even in videos where seeking is inaccurate."""
    small = cv2.resize(img, (SIG_W, SIG_H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def _sig_diff(img: np.ndarray, sig: np.ndarray) -> float:
    return float(np.abs(frame_signature(img).astype(np.int16) - sig.astype(np.int16)).mean())


def read_frame(path: str, index: int, sig: np.ndarray | None = None, window: float = 4.0,
               hw: bool = False) -> np.ndarray | None:
    """Fetch one frame by index (used when showing results, so the index stays small).

    Many DVR files cannot seek exactly: the frame OpenCV jumps to is not the frame that was
    analysed, and boxes would land on the wrong spot. With `sig` (the frame's signature saved
    while indexing) the frames around the target are checked and the matching one is
    returned; None if it cannot be found."""
    if is_image(path):
        return cv2.imread(path)
    # the decoder the video was indexed with (graphics-card pictures can differ a little in
    # colour), then the other one (indexes from versions that did not record it)
    img = _read_frame(open_video(path, hw), index, sig, window)
    if img is None and sig is not None:
        img = _read_frame(open_video(path, not hw), index, sig, window)
    if img is None and sig is not None:
        # still not found (e.g. the video file was replaced or re-encoded): take the closest
        # frame if it is clearly the same scene, rather than showing no picture at all
        img = _read_frame(cv2.VideoCapture(path), index, sig, window, limit=4 * MATCH)
    return img


def _read_frame(cap, index, sig, window, limit=None):
    limit = limit or MATCH
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, img = cap.read()
    if sig is None or (ok and _sig_diff(img, sig) < limit):
        cap.release()
        return img if ok else None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not 0 < fps <= 240:
        fps = 25.0
    span = int(window * fps)
    best, best_img = 1e9, None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, index - span))
    for _ in range(2 * span + 1):
        ok, img = cap.read()
        if not ok:
            break
        d = _sig_diff(img, sig)
        if d < best:
            best, best_img = d, img
            if d < MATCH / 3:
                break
    cap.release()
    return best_img if best < limit else None


def seeks_exactly(path: str, info: MediaInfo | None = None, nearly: bool = False) -> bool:
    """True when jumping to a frame number gives the same picture as reading up to it
    (nearly=True: or a picture a few frames away, which read_frame's check then corrects).
    Checked at a few points, because many files seek fine inside the first keyframe group only."""
    info = info or probe(path)
    if info.kind == "image":
        return True
    if info.frames <= 0:  # raw .h264/.265 streams have no frame count and cannot seek
        return False
    # Test frames that look different from the first one (in a still scene a wrong frame
    # would look right), spread over the first minute.
    sigs: dict[int, np.ndarray] = {}
    first = None
    cap = cv2.VideoCapture(path)
    last = min(info.frames - 2, int(info.fps * 60))
    for i in range(last + 1):
        if i == 0 or (i % max(1, int(info.fps)) == 0 and i >= info.fps * 3):
            ok, img = cap.read()
            if not ok:
                break
            sig = frame_signature(img)
            if first is None:
                first = sig
            elif all(np.abs(sig.astype(np.int16) - o.astype(np.int16)).mean() > 2 * MATCH
                     for o in [first, *sigs.values()]):
                sigs[i] = sig
                if len(sigs) == 3:
                    break
        elif not cap.grab():
            break
    cap.release()
    for t, sig in sigs.items():
        if nearly:
            if read_frame(path, t, sig, window=2.0) is None:
                return False
            continue
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, img = cap.read()
        cap.release()
        if not ok or _sig_diff(img, sig) >= MATCH / 3:
            return False
    return True


def fix_seeking(path: str, out_dir: str, info: MediaInfo | None = None) -> str:
    """Copy a video that cannot seek exactly (raw .h264/.265 DVR exports, broken timestamps)
    into a new .mp4 container without re-encoding (fast, same quality). Returns the path to
    index: the copy if it seeks exactly, otherwise the original."""
    info = info or probe(path)
    if info.kind == "image" or seeks_exactly(path, info, nearly=True):
        return path
    import shutil
    import subprocess

    exes = [shutil.which("ffmpeg")]
    try:
        import imageio_ffmpeg  # bundles ffmpeg, so nothing extra to install on Windows

        exes.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    exes = [e for e in exes if e]
    if not exes:
        log.warning("%s cannot seek exactly and no ffmpeg is available to repair it", path)
        return path
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + "_fixed.mp4")
    raw = info.frames <= 0  # elementary stream: no timestamps, give it the frame rate
    err = ""
    for exe in exes if not os.path.exists(out) else []:
        cmd = [exe, "-y", "-loglevel", "error", "-fflags", "+genpts",
               *(["-r", f"{info.fps:g}"] if raw else []), "-i", path, "-map", "0:v:0", "-c", "copy", out]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(out):
            break
        err = r.stderr.strip()[-300:] or f"exit code {r.returncode}"
        if os.path.exists(out):
            os.remove(out)
    if not os.path.exists(out):
        log.warning("could not repair %s: %s", path, err)
        return path
    if seeks_exactly(out, nearly=True):
        log.warning("%s cannot seek exactly; indexing the repaired copy %s", path, out)
        return out
    return path
