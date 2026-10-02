"""SQLite index. Indexing a video is slow; searching the index is instant.

One database can hold many videos, so a search covers every indexed camera/file.
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3

import numpy as np

from .video import SIG_H, SIG_W, read_frame

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
    source TEXT,                           -- NULL = read by OCR, 'taught' = typed by the user, 'learned' = from a lesson
    vx1 INTEGER, vy1 INTEGER, vx2 INTEGER, vy2 INTEGER
);
CREATE INDEX IF NOT EXISTS plate_reads_media_t ON plate_reads(media_id, t);
CREATE TABLE IF NOT EXISTS plate_labels (   -- corrections typed in the Teach tab; survive re-indexing
    id INTEGER PRIMARY KEY,
    media_id INTEGER, frame INTEGER, x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
    ocr_text TEXT,          -- what the reader said ('' if it could not read it)
    label TEXT,             -- the correct plate text, '' for "not a plate"
    sig BLOB,               -- small grey picture of the plate (48x144 uint8), for look-alike suggestions
    created TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS frame_sigs (   -- 48x27 grey thumbnail of each frame with a detection
    media_id INTEGER, frame INTEGER, sig BLOB,   -- (finds the exact frame again in videos that seek badly)
    PRIMARY KEY (media_id, frame)
);
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
        for col, typ in (("vehicle_model", "TEXT"), ("model_conf", "REAL"), ("ocr_guess", "TEXT"), ("source", "TEXT")):  # older indexes
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
        self.db.execute("""DELETE FROM frame_sigs WHERE media_id=? AND frame NOT IN
                           (SELECT frame FROM plate_reads WHERE media_id=? UNION
                            SELECT frame FROM person_dets WHERE media_id=?)""", (media_id,) * 3)
        self.set_progress(media_id, kind, 0, False)

    # ---- frames ------------------------------------------------------------
    def add_frame_sig(self, media_id: int, frame: int, sig: np.ndarray):
        self.db.execute("INSERT OR REPLACE INTO frame_sigs VALUES (?,?,?)", (media_id, frame, sig.tobytes()))

    def frame_sig(self, row) -> np.ndarray | None:
        r = self.db.execute("SELECT sig FROM frame_sigs WHERE media_id=? AND frame=?",
                            (row["media_id"], row["frame"])).fetchone()
        return None if r is None else np.frombuffer(r["sig"], np.uint8).reshape(SIG_H, SIG_W)

    def frame(self, row) -> np.ndarray | None:
        """The video frame a detection was made in (exact, also for videos that seek badly)."""
        media = self.media(row["media_id"])
        return read_frame(media["path"], row["frame"], self.frame_sig(row)) if media else None

    # ---- plates ------------------------------------------------------------
    def add_plate_reads(self, media_id: int, rows: list[dict]):
        self.db.executemany(
            """INSERT INTO plate_reads(media_id, frame, t, text, conf, region, x1, y1, x2, y2,
               vehicle_type, vehicle_color, vehicle_model, model_conf, ocr_guess, source, vx1, vy1, vx2, vy2)
               VALUES (:media_id, :frame, :t, :text, :conf, :region, :x1, :y1, :x2, :y2,
               :vehicle_type, :vehicle_color, :vehicle_model, :model_conf, :ocr_guess, :source, :vx1, :vy1, :vx2, :vy2)""",
            [{"media_id": media_id, "vehicle_model": None, "model_conf": None, "ocr_guess": None, "source": None, **r}
             for r in rows],
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

    def set_plate_text(self, ids: list[int], text: str, source: str):
        """Overwrite the text of these reads, keeping what the OCR said in ocr_guess."""
        if not ids:
            return
        q = ",".join("?" * len(ids))
        self.db.execute(f"UPDATE plate_reads SET ocr_guess = COALESCE(ocr_guess, NULLIF(text, '')) WHERE id IN ({q})", ids)
        self.db.execute(f"UPDATE plate_reads SET text=?, source=?, conf=1.0 WHERE id IN ({q})", [text, source, *ids])
        self.db.commit()

    def delete_plate_reads(self, ids: list[int]):
        if ids:
            self.db.execute(f"DELETE FROM plate_reads WHERE id IN ({','.join('?' * len(ids))})", ids)
            self.db.commit()

    def reads_with_text(self, text: str):
        return self.db.execute("SELECT * FROM plate_reads WHERE text=? AND source IS NULL", (text,)).fetchall()

    def add_label(self, row, ocr_text: str, label: str, sig):
        self.db.execute(
            "INSERT INTO plate_labels(media_id, frame, x1, y1, x2, y2, ocr_text, label, sig) VALUES (?,?,?,?,?,?,?,?,?)",
            (row["media_id"], row["frame"], row["x1"], row["y1"], row["x2"], row["y2"], ocr_text, label,
             sig))
        self.db.commit()

    def labels(self):
        return self.db.execute("SELECT * FROM plate_labels ORDER BY id").fetchall()

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
