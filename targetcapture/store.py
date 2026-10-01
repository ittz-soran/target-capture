"""SQLite index. Indexing a video is slow; searching the index is instant.

One database can hold many videos, so a search covers every indexed camera/file.
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS media (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    kind TEXT, fps REAL, frames INTEGER, width INTEGER, height INTEGER,
    clock_start TEXT,                 -- wall-clock time of frame 0 (DVR time), optional
    plates_next_frame INTEGER DEFAULT 0,   -- resume point for plate indexing
    persons_next_frame INTEGER DEFAULT 0,  -- resume point for person indexing
    plates_done INTEGER DEFAULT 0,
    persons_done INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS plate_reads (
    id INTEGER PRIMARY KEY,
    media_id INTEGER NOT NULL REFERENCES media(id),
    frame INTEGER, t REAL,
    text TEXT, conf REAL, region TEXT,
    x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
    vehicle_type TEXT, vehicle_color TEXT,
    vehicle_model TEXT, model_conf REAL,   -- e.g. "Toyota Corolla", 0.62
    ocr_guess TEXT,                        -- for plates that could not be read (text = ''): OCR's best guess
    vx1 INTEGER, vy1 INTEGER, vx2 INTEGER, vy2 INTEGER
);
CREATE INDEX IF NOT EXISTS plate_reads_media_t ON plate_reads(media_id, t);
CREATE TABLE IF NOT EXISTS person_dets (
    id INTEGER PRIMARY KEY,
    media_id INTEGER NOT NULL REFERENCES media(id),
    frame INTEGER, t REAL, track_id INTEGER,
    x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
    fx1 INTEGER, fy1 INTEGER, fx2 INTEGER, fy2 INTEGER,  -- face box, NULL if no face
    face_score REAL,
    emb BLOB,                                           -- 512-d float32 ArcFace, L2-normalised
    upper_color TEXT, lower_color TEXT
);
CREATE INDEX IF NOT EXISTS person_dets_media_track ON person_dets(media_id, track_id);
"""


class Store:
    def __init__(self, path: str):
        self.path = path
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(plate_reads)")}
        for col, typ in (("vehicle_model", "TEXT"), ("model_conf", "REAL"), ("ocr_guess", "TEXT")):  # older indexes
            if col not in cols:
                self.db.execute(f"ALTER TABLE plate_reads ADD COLUMN {col} {typ}")

    # ---- media -------------------------------------------------------------
    def upsert_media(self, info, clock_start: str | None) -> int:
        path = os.path.abspath(info.path)
        row = self.db.execute("SELECT id FROM media WHERE path=?", (path,)).fetchone()
        if row:
            if clock_start:
                self.db.execute("UPDATE media SET clock_start=? WHERE id=?", (clock_start, row["id"]))
                self.db.commit()
            return row["id"]
        cur = self.db.execute(
            "INSERT INTO media(path, kind, fps, frames, width, height, clock_start) VALUES (?,?,?,?,?,?,?)",
            (path, info.kind, info.fps, info.frames, info.width, info.height, clock_start),
        )
        self.db.commit()
        return cur.lastrowid

    def media(self, media_id: int | None = None):
        if media_id is None:
            return self.db.execute("SELECT * FROM media ORDER BY id").fetchall()
        return self.db.execute("SELECT * FROM media WHERE id=?", (media_id,)).fetchone()

    def set_progress(self, media_id: int, kind: str, next_frame: int, done: bool = False):
        self.db.execute(
            f"UPDATE media SET {kind}_next_frame=?, {kind}_done=? WHERE id=?",
            (next_frame, int(done), media_id),
        )
        self.db.commit()

    def reset(self, media_id: int, kind: str):
        table = "plate_reads" if kind == "plates" else "person_dets"
        self.db.execute(f"DELETE FROM {table} WHERE media_id=?", (media_id,))
        self.set_progress(media_id, kind, 0, False)

    # ---- plates ------------------------------------------------------------
    def add_plate_reads(self, media_id: int, rows: list[dict]):
        self.db.executemany(
            """INSERT INTO plate_reads(media_id, frame, t, text, conf, region, x1, y1, x2, y2,
               vehicle_type, vehicle_color, vehicle_model, model_conf, ocr_guess, vx1, vy1, vx2, vy2)
               VALUES (:media_id, :frame, :t, :text, :conf, :region, :x1, :y1, :x2, :y2,
               :vehicle_type, :vehicle_color, :vehicle_model, :model_conf, :ocr_guess, :vx1, :vy1, :vx2, :vy2)""",
            [{"media_id": media_id, "vehicle_model": None, "model_conf": None, "ocr_guess": None, **r} for r in rows],
        )

    def plate_reads(self, media_id: int | None = None, unread: bool = False):
        """Plates whose characters were read, or with unread=True the plates that were
        detected but could not be read (text = '')."""
        q = "SELECT * FROM plate_reads WHERE " + ("text = ''" if unread else "text != ''")
        args: tuple = ()
        if media_id is not None:
            q += " AND media_id=?"
            args = (media_id,)
        return self.db.execute(q + " ORDER BY media_id, t", args).fetchall()

    # ---- persons -----------------------------------------------------------
    def max_track_id(self, media_id: int) -> int:
        row = self.db.execute("SELECT MAX(track_id) m FROM person_dets WHERE media_id=?", (media_id,)).fetchone()
        return row["m"] or 0

    def add_person_dets(self, media_id: int, rows: list[dict]):
        out = []
        for r in rows:
            r = {"media_id": media_id, **r}
            emb = r.get("emb")
            r["emb"] = None if emb is None else np.asarray(emb, np.float32).tobytes()
            for k in ("fx1", "fy1", "fx2", "fy2", "face_score"):
                r.setdefault(k, None)
            out.append(r)
        self.db.executemany(
            """INSERT INTO person_dets(media_id, frame, t, track_id, x1, y1, x2, y2,
               fx1, fy1, fx2, fy2, face_score, emb, upper_color, lower_color)
               VALUES (:media_id, :frame, :t, :track_id, :x1, :y1, :x2, :y2,
               :fx1, :fy1, :fx2, :fy2, :face_score, :emb, :upper_color, :lower_color)""",
            out,
        )

    def face_rows(self, min_face_px: int = 0):
        return self.db.execute(
            "SELECT * FROM person_dets WHERE emb IS NOT NULL AND (fx2 - fx1) >= ? ORDER BY media_id, t",
            (min_face_px,),
        ).fetchall()

    def track_rows(self, media_id: int, track_id: int):
        return self.db.execute(
            "SELECT * FROM person_dets WHERE media_id=? AND track_id=? ORDER BY t", (media_id, track_id)
        ).fetchall()

    def commit(self):
        self.db.commit()


def clock_time(media_row, t: float) -> str | None:
    """Convert a video offset into DVR wall-clock time when --clock-start was given."""
    start = media_row["clock_start"] if media_row is not None else None
    if not start:
        return None
    try:
        base = dt.datetime.fromisoformat(start)
    except ValueError:
        return None
    return (base + dt.timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S")


def fmt_offset(t: float) -> str:
    t = max(0.0, t)
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
