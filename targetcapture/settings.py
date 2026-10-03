"""Saved settings (settings.json next to the app), edited in the web page's Settings tab.

Updates never touch this file, so your choices survive them.
"""
from __future__ import annotations

import json
import os
import subprocess

from . import __version__

PATH = "settings.json"

DEFAULTS = {
    # Indexing (defaults of the Upload & index tab; live cameras use them too)
    "read_plates": True,
    "find_people": True,
    "speed": "fast",          # SPEEDS below, or "custom" (the two frame rates that follow)
    "plate_fps": 4.0,
    "person_fps": 3.0,
    "plate_script": "auto",
    "car_models": True,
    "car_model_size": "auto",  # "fast", "accurate" or "auto" (accurate on an NVIDIA GPU)
    "vehicle_crops": False,
    "skip_empty_road": False,
    "min_ocr_conf": 0.5,
    "gpu_decode": True,       # decode video on the graphics card when possible (falls back by itself)
    # Searching
    "plate_max_dist": 1,
    "face_threshold": 0.45,   # watch list: how alike a face must be (higher = stricter)
    "watch_plate_dist": 1,    # watch list: characters a plate may differ by
    "page_size": 36,
    # Live cameras
    "keep_days": 7.0,
    "segment_seconds": 60,
    # Web page
    "db_path": "targetcapture.db",
    "max_upload_mb": 20000,   # read by run_windows.bat when the page starts
}

# How much of each second of video is looked at. Most plates and faces are seen in several
# frames, so fewer frames lose little and index much faster.
SPEEDS = {
    "fast": {"plate_fps": 3.0, "person_fps": 2.0, "model_looks": 2},
    "balanced": {"plate_fps": 4.0, "person_fps": 3.0, "model_looks": 4},
    "thorough": {"plate_fps": 6.0, "person_fps": 4.0, "model_looks": 4},
}
SPEED_NAMES = {"fast": "Fast (long recordings)", "balanced": "Balanced",
               "thorough": "Thorough (slowest, catches the most)", "custom": "Custom frame rates"}


def rates(cfg: dict, speed: str | None = None) -> dict:
    """plate_fps, person_fps and model_looks for a speed choice (cfg's own when None)."""
    speed = speed or cfg["speed"]
    if speed in SPEEDS:
        return dict(SPEEDS[speed])
    return {"plate_fps": float(cfg["plate_fps"]), "person_fps": float(cfg["person_fps"]), "model_looks": 4}


def load(path: str = PATH) -> dict:
    s = dict(DEFAULTS)
    try:
        with open(path, encoding="utf-8") as f:
            s.update({k: v for k, v in json.load(f).items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return s


def save(values: dict, path: str = PATH):
    s = load(path)
    s.update({k: v for k, v in values.items() if k in DEFAULTS})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=2, ensure_ascii=False)


def _git(*args, timeout=30) -> str | None:
    try:
        r = subprocess.run(["git", *args], capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def version() -> dict:
    """This copy's version: number, commit and its date (commit is None outside a git folder)."""
    line = _git("log", "-1", "--format=%h|%cd|%s", "--date=format:%Y-%m-%d %H:%M")
    commit, date, subject = (line.split("|", 2) if line else (None, None, None))
    return {"version": __version__, "commit": commit, "date": date, "subject": subject}


def check_update() -> dict:
    """Ask GitHub for newer changes. {'ok': False, 'error': ...} when it cannot be reached."""
    if _git("rev-parse", "--git-dir") is None:
        return {"ok": False, "error": "This folder is not linked to GitHub yet: run update_windows.bat once."}
    if _git("fetch", "-q", "origin", "main", timeout=60) is None:
        return {"ok": False, "error": "Could not reach GitHub. Check the internet connection."}
    log = _git("log", "HEAD..origin/main", "--format=%cd|%s", "--date=format:%Y-%m-%d") or ""
    changes = [line.split("|", 1) for line in log.splitlines() if "|" in line]
    return {"ok": True, "changes": changes}


if __name__ == "__main__":  # run_windows.bat: python -m targetcapture.settings max_upload_mb
    import sys

    print(load()[sys.argv[1]])
