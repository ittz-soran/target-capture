"""Teaching the plate reader.

In the Teach tab you look at plates the system could not read (or read wrongly) and type
the correct text. Each correction is used three ways:

1. **Right away:** the reads of that capture get the correct text, so searches find it.
2. **Same mistake:** if the reader produced a wrong text (e.g. "07VA9"), every other read
   with exactly that text can be fixed in one click, and future indexing fixes it too.
3. **Same plate again:** a small picture of the plate is kept, and the Teach tab suggests
   other captures that look like it, for you to confirm with a tick. Pictures are never
   applied automatically: plates of the same style with different numbers look almost
   identical to a picture comparison (tested: up to 0.97 alike, while one plate across
   frames of the same video ranged 0.57-0.93), so only a person can confirm the number.

Corrections live in their own table, so they survive re-indexing, and `export` writes
them out as a labelled dataset for training a better plate reader later.
"""
from __future__ import annotations

import csv
import os

import cv2
import numpy as np

from .plates import fold, normalize
from .video import read_frame

SUGGEST_SIM = 0.75   # Teach tab: show as "may be the same plate" (you confirm)
SIG_W, SIG_H = 144, 48


def plate_signature(crop: np.ndarray) -> np.ndarray | None:
    """Small fixed-size grey picture of a plate (uint8, SIG_H x SIG_W)."""
    if crop is None or crop.size == 0 or min(crop.shape[:2]) < 6:
        return None
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return cv2.resize(g, (SIG_W, SIG_H), interpolation=cv2.INTER_AREA)


def signature_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """How alike two plate pictures are (0-1). The middle of each is slid over the other,
    so detection boxes that are a few pixels off still line up."""
    best = 0.0
    for x, y in ((a, b), (b, a)):
        inner = x[5:SIG_H - 5, 10:SIG_W - 10]
        best = max(best, float(cv2.matchTemplate(y, inner, cv2.TM_CCOEFF_NORMED).max()))
    return best


def crop_of(row, frame: np.ndarray) -> np.ndarray | None:
    if frame is None:
        return None
    c = frame[max(0, row["y1"]):row["y2"], max(0, row["x1"]):row["x2"]]
    return c if c.size else None


def row_signature(store, row) -> np.ndarray | None:
    media = store.media(row["media_id"])
    return plate_signature(crop_of(row, read_frame(media["path"], row["frame"])))


class Memory:
    """Reading mistakes you corrected in the Teach tab, applied while indexing: a read that
    comes out exactly as a corrected mistake (e.g. "07VA9") gets the corrected text and is
    marked "learned"."""

    def __init__(self, store):
        self.rules: dict[str, str] = {}
        for lab in store.labels():
            if lab["ocr_text"] and lab["label"]:
                self.rules[fold(lab["ocr_text"])] = lab["label"]

    def __bool__(self):
        return bool(self.rules)

    def apply(self, read: dict, img: np.ndarray = None) -> dict:
        if read["text"] and fold(read["text"]) in self.rules:
            label = self.rules[fold(read["text"])]
            if label != read["text"]:
                read.update(ocr_guess=read["text"], text=label, source="learned")
        return read


def teach(store, ids: list[int], best, label: str, not_a_plate: bool = False) -> dict:
    """Save a correction for one capture / sighting (all its reads, by id)."""
    label = "" if not_a_plate else normalize(label)
    ocr_text = best["text"] or ""
    if not_a_plate:
        store.delete_plate_reads(ids)
    else:
        store.set_plate_text(ids, label, "taught")
    sig = None if not_a_plate else row_signature(store, best)
    store.add_label(best, ocr_text, label, None if sig is None else sig.tobytes())
    same_mistake = [r["id"] for r in store.reads_with_text(ocr_text)] if ocr_text and label else []
    return {"label": label, "ocr_text": ocr_text, "fixed": len(ids), "same_mistake": same_mistake, "sig": sig}


def similar_captures(store, sig: np.ndarray, candidates: list[dict], min_sim: float = SUGGEST_SIM) -> list[tuple]:
    """Captures/sightings whose best picture looks like `sig`, best first: [(sim, candidate)]."""
    out = []
    for c in candidates:
        s = row_signature(store, c["best"])
        if s is not None:
            sim = signature_similarity(s, sig)
            if sim >= min_sim:
                out.append((sim, c))
    return sorted(out, key=lambda x: -x[0])


def export(store, out_dir: str) -> int:
    """Write every taught plate as an image + labels.csv (for training a plate reader)."""
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    n = 0
    with open(os.path.join(out_dir, "labels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "plate_text", "ocr_said"])
        for lab in store.labels():
            if not lab["label"]:
                continue
            media = store.media(lab["media_id"])
            crop = crop_of(lab, read_frame(media["path"], lab["frame"])) if media else None
            if crop is None or crop.size == 0:
                continue
            name = f"images/{lab['id']:05d}.png"
            cv2.imwrite(os.path.join(out_dir, name), crop)
            w.writerow([name, lab["label"], lab["ocr_text"]])
            n += 1
    return n
