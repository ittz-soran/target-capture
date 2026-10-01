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


def iter_frames(
    path: str,
    sample_fps: float,
    start_frame: int = 0,
    motion_skip: bool = True,
    motion_threshold: float = 1.5,
    max_still: float = 5.0,
) -> Iterator[Frame]:
    """Yield sampled frames. For an image, yields the image once."""
    info = probe(path)
    if info.kind == "image":
        if start_frame == 0:
            yield Frame(0, 0.0, cv2.imread(path))
        return

    step = max(1, round(info.fps / sample_fps))
    cap = cv2.VideoCapture(path)
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    i = start_frame
    last_sig = None
    last_t = -1e9
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
    cap.release()


def read_frame(path: str, index: int) -> np.ndarray | None:
    """Fetch one frame by index (used when building reports, so the index stays small)."""
    if is_image(path):
        return cv2.imread(path)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, img = cap.read()
    cap.release()
    return img if ok else None
