"""Fetch recorded footage from a DVR / NVR and index it, piece by piece.

The time range is cut into pieces (10 minutes by default). While one piece is being indexed the
next one is already downloading, so the graphics card and the network work at the same time and
results show up after the first piece, not at the end.

Download, fastest first:
  * Hikvision / HiLook: ISAPI download (/ISAPI/ContentMgmt/download), faster than real time.
  * Dahua / Imou: loadfile.cgi, faster than real time.
  * Either make: RTSP playback copied with ffmpeg (always works where live view works, but
    only at real-time speed).
Each piece is copied into an .mp4 (no re-encoding) so it seeks exactly.

Times are the DVR's own clock (what its playback screen shows).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import queue
import re
import subprocess
import threading
import time
import uuid
from urllib.parse import quote

from .live import _creds, ffmpeg_exe

HIK = "hik"
_URIS: dict = {}  # (ip, channel) -> [(start, end, playbackURI)] from the last recording search


def is_hik(make: str) -> bool:
    return make.lower().startswith(HIK)


def _auth(user, password):
    from requests.auth import HTTPDigestAuth

    return HTTPDigestAuth(user, password)


def _hik_t(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_t(s: str) -> dt.datetime:
    s = s.strip().replace("Z", "").replace("T", " ").replace("_", " ")
    s = re.sub(r"[+-]\d\d:\d\d$", "", s)
    return dt.datetime.fromisoformat(s)


def find_recordings(make: str, ip: str, user: str, password: str, channel: int, start: dt.datetime,
                    end: dt.datetime, http_port: int = 80) -> list[tuple[dt.datetime, dt.datetime]] | None:
    """Recorded stretches of one channel between start and end, or None if the DVR could not
    be asked (then every piece is simply tried)."""
    import requests

    base = f"http://{ip}:{http_port}"
    try:
        if is_hik(make):
            out, pos = [], 0
            while True:
                body = (f'<?xml version="1.0" encoding="UTF-8"?><CMSearchDescription><searchID>{uuid.uuid4()}</searchID>'
                        f"<trackList><trackID>{channel}01</trackID></trackList><timeSpanList><timeSpan>"
                        f"<startTime>{_hik_t(start)}</startTime><endTime>{_hik_t(end)}</endTime></timeSpan>"
                        f"</timeSpanList><maxResults>50</maxResults><searchResultPostion>{pos}</searchResultPostion>"
                        "<metadataList><metadataDescriptor>//recordType.meta.std-cgi.com</metadataDescriptor>"
                        "</metadataList></CMSearchDescription>")
                r = requests.post(f"{base}/ISAPI/ContentMgmt/search", data=body, auth=_auth(user, password), timeout=15)
                if r.status_code != 200:
                    return None
                items = re.findall(r"<timeSpan>\s*<startTime>([^<]+)</startTime>\s*<endTime>([^<]+)</endTime>", r.text)
                out += [(_parse_t(a), _parse_t(b)) for a, b in items]
                for m in re.finditer(r"<startTime>([^<]+)</startTime>\s*<endTime>([^<]+)</endTime>.*?"
                                     r"<playbackURI>([^<]+)</playbackURI>", r.text, re.S):
                    _URIS.setdefault((ip, channel), []).append((_parse_t(m.group(1)), _parse_t(m.group(2)),
                                                                 m.group(3)))
                if "MORE" not in r.text or not items or pos > 2000:
                    return out
                pos += len(items)
        # Dahua: create a finder, search, read the matches, close it
        get = lambda q: requests.get(f"{base}/cgi-bin/mediaFileFind.cgi?{q}", auth=_auth(user, password),  # noqa: E731
                                     timeout=15)
        r = get("action=factory.create")
        m = re.search(r"result=(\d+)", r.text) if r.status_code == 200 else None
        if not m:
            return None
        obj = m.group(1)
        fmt = "%Y-%m-%d%%20%H:%M:%S"
        get(f"action=findFile&object={obj}&condition.Channel={channel}&condition.StartTime={start.strftime(fmt)}"
            f"&condition.EndTime={end.strftime(fmt)}&condition.Types[0]=dav")
        out = []
        for _ in range(40):
            r = get(f"action=findNextFile&object={obj}&count=100")
            found = re.search(r"found=(\d+)", r.text)
            starts = dict(re.findall(r"items\[(\d+)\]\.StartTime=(.+)", r.text))
            ends = dict(re.findall(r"items\[(\d+)\]\.EndTime=(.+)", r.text))
            out += [(_parse_t(starts[k]), _parse_t(ends[k])) for k in starts if k in ends]
            if not found or int(found.group(1)) < 100:
                break
        get(f"action=close&object={obj}")
        get(f"action=destroy&object={obj}")
        return out
    except (requests.RequestException, ValueError):
        return None


def pieces(start: dt.datetime, end: dt.datetime, minutes: int,
           recorded: list[tuple[dt.datetime, dt.datetime]] | None) -> list[tuple[dt.datetime, dt.datetime]]:
    """Cut start..end into pieces, keeping only the parts with recordings (all, if unknown)."""
    spans = [(start, end)] if recorded is None else sorted(
        (max(a, start), min(b, end)) for a, b in recorded if b > start and a < end)
    merged: list[list[dt.datetime]] = []
    for a, b in spans:
        if merged and a <= merged[-1][1] + dt.timedelta(seconds=5):
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    step = dt.timedelta(minutes=minutes)
    out = []
    for a, b in merged:
        while a < b:
            c = min(a + step, b)
            if (c - a).total_seconds() >= 3:
                out.append((a, c))
            a = c
    return out


def playback_url(make: str, ip: str, rtsp_port: int, user: str, password: str, channel: int,
                 start: dt.datetime, end: dt.datetime) -> str:
    cred = _creds(user, password)[0]
    if is_hik(make):
        f = "%Y%m%dT%H%M%SZ"
        return (f"rtsp://{cred}@{ip}:{rtsp_port}/Streaming/tracks/{channel}01"
                f"?starttime={start.strftime(f)}&endtime={end.strftime(f)}")
    f = "%Y_%m_%d_%H_%M_%S"
    return (f"rtsp://{cred}@{ip}:{rtsp_port}/cam/playback?channel={channel}&subtype=0"
            f"&starttime={start.strftime(f)}&endtime={end.strftime(f)}")


def _hik_requests(ip, channel, start, end):
    """The forms of the ISAPI download request, most common first (firmware differ)."""
    f = "%Y%m%dT%H%M%SZ"
    uris = [f"rtsp://{ip}/Streaming/tracks/{channel}01/?starttime={start.strftime(f)}&endtime={end.strftime(f)}"]
    for a, b, uri in _URIS.get((ip, channel), []):  # the recording's own address from the search
        if a <= start < b:
            uri = uri.replace("&amp;", "&")
            uri = re.sub(r"starttime=[^&]*", f"starttime={start.strftime(f)}", uri)
            uri = re.sub(r"endtime=[^&]*", f"endtime={end.strftime(f)}", uri)
            uris.append(uri)
            break
    out = []
    for uri in uris:
        esc = uri.replace("&", "&amp;")
        body = (f'<?xml version="1.0" encoding="UTF-8"?><downloadRequest version="1.0" '
                f'xmlns="http://www.isapi.org/ver20/XMLSchema"><playbackURI>{esc}</playbackURI></downloadRequest>')
        out += [("GET", "/ISAPI/ContentMgmt/download", body), ("POST", "/ISAPI/ContentMgmt/download", body),
                ("GET", "/ISAPI/ContentMgmt/download?playbackURI=" + quote(uri, safe=""), None)]
    return out


def _http_download(make, ip, http_port, user, password, channel, start, end, path, stop,
                   progress=lambda got, total: None) -> str:
    """Download one piece over the DVR's web port. Returns '' when done, else the reason."""
    import requests

    base = f"http://{ip}:{http_port}"
    if is_hik(make):
        tries = _hik_requests(ip, channel, start, end)
    else:
        q = (f"action=startLoad&channel={channel}&startTime={quote(start.strftime('%Y-%m-%d %H:%M:%S'), safe=':')}"
             f"&endTime={quote(end.strftime('%Y-%m-%d %H:%M:%S'), safe=':')}&subtype=0")
        tries = [("GET", f"/cgi-bin/loadfile.cgi?{q}", None)]
    why = ""
    for method, url, body in tries:
        try:
            with requests.request(method, base + url, data=body, auth=_auth(user, password), stream=True,
                                  timeout=(10, 60)) as r:
                ctype = r.headers.get("Content-Type", "")
                if r.status_code == 401:
                    return "the DVR refused the user name / password for downloads"
                if r.status_code != 200 or "xml" in ctype or "text" in ctype or "json" in ctype:
                    text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r.text[:400])).strip() if r.status_code != 200 \
                        or "xml" in ctype or "text" in ctype else ""
                    why = f"answered {r.status_code}" + (f": {text[:150]}" if text else "")
                    continue
                total = int(r.headers.get("Content-Length") or 0)
                got = 0
                with open(path, "wb") as fh:
                    for chunk in r.iter_content(256 << 10):
                        if stop.is_set():
                            return "stopped"
                        fh.write(chunk)
                        got += len(chunk)
                        progress(got, total)
            if os.path.getsize(path) > 50_000:
                return ""
            why = "gave no video"
        except requests.ConnectionError:
            return f"could not connect to the DVR's web port {http_port} (check the IP address and Web port)"
        except requests.RequestException as e:
            why = f"failed: {e.__class__.__name__}"
    return why or "not supported"


def _rtsp_download(url, seconds, path, stop, status=lambda msg: None) -> str:
    """Copy a piece from RTSP playback (real-time speed), telling how far it is."""
    exe = ffmpeg_exe()
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    proc = subprocess.Popen([exe, "-nostdin", "-y", "-loglevel", "error", "-progress", "pipe:1", "-nostats",
                             "-rtsp_transport", "tcp", "-timeout", "15000000", "-i", url,
                             "-map", "0:v:0", "-c", "copy", "-t", f"{seconds:.0f}", "-f", "mpegts", path],
                            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=flags)
    got = [0.0, time.time()]  # seconds copied, when it last grew

    def read_progress():
        for line in proc.stdout:
            m = re.match(rb"out_time_us=(\d+)", line)
            if m and int(m.group(1)) / 1e6 > got[0]:
                got[0], got[1] = int(m.group(1)) / 1e6, time.time()

    threading.Thread(target=read_progress, daemon=True).start()
    shown = 0.0
    while proc.poll() is None:
        if stop.is_set() or time.time() - got[1] > 60:  # nothing new for a minute: give up on it
            proc.kill()
            break
        if time.time() - shown > 5:
            shown = time.time()
            status(f"downloading (playback, real-time speed): {got[0] / 60:.1f} of {seconds / 60:.0f} min")
        time.sleep(0.5)
    err = proc.stderr.read().decode(errors="replace").strip()[-200:]
    if stop.is_set():
        return "stopped"
    if os.path.exists(path) and os.path.getsize(path) > 50_000:
        return ""
    if not got[0] and not err:
        err = "the DVR sent no video for a minute"
    return "playback gave no video" + (f" ({re.sub(r'rtsp://[^@ ]*@', 'rtsp://', err)})" if err else "")


def _to_mp4(raw: str, mp4: str) -> bool:
    """Copy into .mp4 without re-encoding (exact seeking). False if ffmpeg cannot read it."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    r = subprocess.run([ffmpeg_exe(), "-nostdin", "-y", "-loglevel", "error", "-probesize", "50M",
                        "-analyzeduration", "20M", "-fflags", "+genpts", "-i", raw, "-map", "0:v:0", "-c", "copy",
                        mp4], capture_output=True, creationflags=flags)
    ok = r.returncode == 0 and os.path.exists(mp4) and os.path.getsize(mp4) > 50_000
    if not ok and os.path.exists(mp4):
        os.remove(mp4)
    return ok


def _readable(path: str) -> bool:
    try:
        from .video import probe

        return probe(path).frames > 0
    except Exception:
        return False


class Meter:
    """Bytes downloaded, for the speed shown on the page (shared by the pieces downloading)."""

    def __init__(self):
        self.lock = threading.Lock()
        self.total = 0
        self.recent: list[tuple[float, int]] = []  # (time, total) over the last seconds

    def add(self, n: int):
        with self.lock:
            self.total += n
            now = time.time()
            self.recent.append((now, self.total))
            while len(self.recent) > 2 and now - self.recent[0][0] > 10:
                self.recent.pop(0)

    def speed(self) -> float:
        """Bytes per second over the last 10 seconds."""
        with self.lock:
            if len(self.recent) < 2 or time.time() - self.recent[-1][0] > 10:
                return 0.0
            (t0, b0), (t1, b1) = self.recent[0], self.recent[-1]
            return (b1 - b0) / max(t1 - t0, 0.5)


def mb(n: float) -> str:
    return f"{n / 1e6:.0f} MB" if n >= 10e6 else f"{n / 1e6:.1f} MB"


def download_piece(job: dict, channel: int, start: dt.datetime, end: dt.datetime, folder: str,
                   stop: threading.Event, status=lambda msg: None, meter: Meter | None = None
                   ) -> tuple[str | None, str]:
    """Fetch one piece into folder. Returns (path to an .mp4, how) or (None, reason)."""
    meter = meter or Meter()
    last = {"got": 0, "shown": 0.0, "t0": time.time()}

    def web_progress(got, total):
        meter.add(got - last["got"])
        last["got"] = got
        now = time.time()
        if now - last["shown"] >= 2:
            last["shown"] = now
            rate = got / max(now - last["t0"], 0.1)
            left = f", about {(total - got) / rate:.0f} s left" if total and rate else ""
            status(f"downloading: {mb(got)}" + (f" of {mb(total)} ({100 * got / total:.0f}%)" if total else "")
                   + f", {rate / 1e6:.1f} MB/s{left}")
    os.makedirs(folder, exist_ok=True)
    name = f"{job['name']}_ch{channel}_{start.strftime('%Y%m%d_%H%M%S')}"
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name)
    mp4 = os.path.join(folder, name + ".mp4")
    if os.path.exists(mp4) and os.path.getsize(mp4) > 50_000:
        return mp4, "already downloaded"
    raw = os.path.join(folder, name + ".part")
    reasons = []
    ways = []
    if job.get("method", "auto") in ("auto", "web"):
        ways.append(("web download", lambda: _http_download(
            job["make"], job["ip"], job.get("http_port", 80), job["user"], job["password"], channel, start, end, raw,
            stop, web_progress)))
    if job.get("method", "auto") in ("auto", "rtsp"):
        url = playback_url(job["make"], job["ip"], job.get("rtsp_port", 554), job["user"], job["password"],
                           channel, start, end)
        ways.append(("playback (real-time speed)", lambda: _rtsp_download(url, (end - start).total_seconds(), raw,
                                                                          stop, status)))
    for how, fetch in ways:
        if stop.is_set():
            break
        status(f"downloading ({how})" + (f" because {reasons[-1]}" if reasons else ""))
        why = fetch()
        took = time.time() - last["t0"]
        size = os.path.getsize(raw) if os.path.exists(raw) else 0
        if not why and "playback" in how:
            meter.add(size)
        note = how + (f", {mb(size)} at {size / max(took, 0.1) / 1e6:.1f} MB/s" if not why else "") + (
            f" ({'; '.join(reasons)})" if reasons else "")
        if not why:
            status("copying into .mp4")
        if not why and _to_mp4(raw, mp4):
            os.remove(raw)
            return mp4, note
        if not why:  # the .mp4 copy failed: index the file as it came, if it can be read
            kept = os.path.join(folder, name + (".ts" if "playback" in how else ".ps" if is_hik(job["make"]) else ".dav"))
            os.replace(raw, kept)
            if _readable(kept):
                return kept, note
            os.remove(kept)
        reasons.append(f"{how}: {why or 'not a readable video'}")
        last.update(got=0, t0=time.time())
        if why == "stopped":
            break
    if os.path.exists(raw):
        os.remove(raw)
    return None, "; ".join(reasons)


# ---- jobs: run in their own process (python -m targetcapture fetch JOB), so they keep going
#      when the web page is closed. Progress goes to <job>.status.json next to the job file.

def jobs_dir(db_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(db_path)), "dvr_jobs")


def new_job(db_path: str, job: dict) -> str:
    d = jobs_dir(db_path)
    os.makedirs(d, exist_ok=True)
    jid = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(d, f"{jid}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({**job, "id": jid}, f)
    return path


def read_status(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def list_jobs(db_path: str) -> list[dict]:
    d = jobs_dir(db_path)
    if not os.path.isdir(d):
        return []
    out = []
    for f in sorted(os.listdir(d), reverse=True):
        if f.endswith(".status.json"):
            s = read_status(os.path.join(d, f))
            if s:
                s["_path"] = os.path.join(d, f)
                out.append(s)
    return out


def default_folder(db_path: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(db_path)), "dvr_videos")


class _Job:
    """Status file and Stop file of one job."""

    def __init__(self, job_path: str, state: dict):
        self.status_path = job_path[:-5] + ".status.json"
        self.stop_path = job_path[:-5] + ".stop"
        self.job_path = job_path
        self.stop = threading.Event()
        self.state = state
        self.lock = threading.Lock()
        state.setdefault("started", dt.datetime.now().isoformat(timespec="seconds"))
        threading.Thread(target=self._watch_stop, daemon=True).start()
        self.save()

    def _watch_stop(self):
        while not self.stop.is_set():
            if os.path.exists(self.stop_path):
                self.stop.set()
            time.sleep(1)

    def save(self):
        with self.lock:
            self.state["updated"] = dt.datetime.now().isoformat(timespec="seconds")
            tmp = self.status_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.state, fh)
            os.replace(tmp, self.status_path)

    def finish(self):
        for f in (self.stop_path, self.job_path):  # the job file holds the password: remove it
            if os.path.exists(f):
                os.remove(f)


def _index_piece(store, p: dict, job: dict, cfg: dict, r: dict, clock_start: str | None):
    """Index one downloaded video into the piece's status."""
    from .pipeline import index_media

    t0 = time.time()
    res = index_media(store, p["path"], plates=job.get("plates", True), persons=job.get("persons", True), **r,
                      clock_start=clock_start, min_ocr_conf=cfg["min_ocr_conf"], plate_script=cfg["plate_script"],
                      car_models=cfg["car_models"], vehicle_crops=cfg["vehicle_crops"],
                      skip_empty_road=cfg["skip_empty_road"])
    p["state"] = "done"
    p["media_id"] = res["media_id"]
    p["found"] = {k: res[k] for k in ("plates", "persons") if k in res and res[k] >= 0}
    dur = res["info"].duration if res.get("info") else 0
    p["indexed_in"] = f"{dur / 60:.0f} min of video in {(time.time() - t0) / 60:.1f} min" if dur else ""
    if res.get("alerts"):
        p["alerts"] = len(res["alerts"])
    if job.get("delete_after"):  # the pictures of the results are kept, so they still show
        try:
            os.remove(p["path"])
            p["note"] = (p.get("note", "") + ", video deleted after indexing").lstrip(", ")
        except OSError:
            pass


def run_job(db_path: str, job_path: str):
    """Download and index every piece of a job. Several pieces download at the same time (the
    DVR's web download is limited per connection) while the finished ones are indexed in order."""
    from .settings import load, rates
    from .store import Store

    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    if job.get("kind") == "folder":
        return run_folder(db_path, job_path, job)
    J = _Job(job_path, {"id": job["id"], "name": job["name"], "channels": job["channels"], "from": job["start"],
                        "to": job["end"], "state": "finding recordings", "pieces": []})
    state, stop = J.state, J.stop
    start, end = _parse_t(job["start"]), _parse_t(job["end"])
    plan = []
    for ch in job["channels"]:
        rec = find_recordings(job["make"], job["ip"], job["user"], job["password"], ch, start, end,
                              job.get("http_port", 80))
        for a, b in pieces(start, end, job.get("piece_minutes", 10), rec):
            plan.append({"channel": ch, "start": a.isoformat(sep=" "), "end": b.isoformat(sep=" "),
                         "state": "waiting", "note": ""})
        if rec is not None and not any(p["channel"] == ch for p in plan):
            state.setdefault("notes", []).append(f"Channel {ch}: no recordings in this time.")
    state["pieces"] = plan
    state["state"] = "running" if plan else "nothing to fetch"
    J.save()
    folder = job.get("folder") or default_folder(db_path)
    parallel = max(1, int(job.get("parallel", 2)))
    meter = Meter()
    ahead = threading.Semaphore(parallel + 1)  # downloaded but not yet indexed (disk space)
    done_ev = [threading.Event() for _ in plan]
    nxt = iter(range(len(plan)))
    nxt_lock = threading.Lock()

    def downloader():
        while not stop.is_set():
            with nxt_lock:
                i = next(nxt, None)
            if i is None:
                return
            while not ahead.acquire(timeout=1):
                if stop.is_set():
                    return
            p = plan[i]

            def st_(msg, p=p):
                p["state"] = msg
                J.save()

            path, how = download_piece(job, p["channel"], _parse_t(p["start"]), _parse_t(p["end"]), folder, stop, st_,
                                       meter)
            if path:
                p["state"], p["note"], p["path"] = "downloaded, waiting to be indexed", how, path
            else:
                p["state"], p["note"] = ("stopped" if stop.is_set() else "failed"), how
                ahead.release()
            done_ev[i].set()
            J.save()

    for _ in range(min(parallel, len(plan))):
        threading.Thread(target=downloader, daemon=True, name="dvr-download").start()

    def speed_line():
        while not stop.is_set() and state["state"] == "running":
            got = sum(1 for p in plan if p.get("path") or p["state"] in ("done", "failed"))
            state["download"] = (f"{got} of {len(plan)} pieces downloaded, {mb(meter.total)} so far, "
                                 f"{meter.speed() / 1e6:.1f} MB/s now")
            J.save()
            time.sleep(3)

    threading.Thread(target=speed_line, daemon=True).start()
    cfg = load()
    r = rates(cfg, job.get("speed", cfg["speed"]))
    store = Store(db_path)
    for i, p in enumerate(plan):
        while not done_ev[i].wait(1):
            if stop.is_set():
                break
        if stop.is_set():
            break
        if not p.get("path"):
            continue
        p["state"] = "indexing"
        J.save()
        try:
            _index_piece(store, p, job, cfg, r, p["start"])
        except Exception as e:  # one bad piece must not stop the rest
            p["state"], p["note"] = "failed", f"indexing: {e}"
        ahead.release()
        J.save()
    done = sum(p["state"] == "done" for p in plan)
    state["state"] = "stopped" if stop.is_set() else f"finished: {done} of {len(plan)} pieces indexed"
    state["download"] = f"{mb(meter.total)} downloaded"
    J.save()
    J.finish()


VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".dav", ".h264", ".h265", ".ts", ".ps", ".asf")


def clock_from_name(name: str) -> str | None:
    """The recording time in a downloaded file's name, e.g. 192.168.1.230_01_20261014100000_...mp4,
    ch01_20261014_100000.mp4 or 2026-10-14 10.00.00.mp4 (None if there is none)."""
    m = re.search(r"(20\d\d)[-_.]?([01]\d)[-_.]?([0-3]\d)[-_. T]?([0-2]\d)[-_.:]?([0-5]\d)[-_.:]?([0-5]\d)", name)
    if not m:
        return None
    try:
        return dt.datetime(*map(int, m.groups())).isoformat(sep=" ")
    except ValueError:
        return None


def run_folder(db_path: str, job_path: str, job: dict):
    """Index the videos in a folder (e.g. downloaded with the DVR's own web page), oldest name
    first, and keep watching for new ones until stopped. A file is taken once its size has
    stopped growing, so files still being downloaded are left alone."""
    from .settings import load, rates
    from .store import Store

    J = _Job(job_path, {"id": job["id"], "name": job["name"], "kind": "folder", "folder": job["folder"],
                        "channels": [], "from": "", "to": "", "state": "running", "pieces": []})
    state, stop = J.state, J.stop
    cfg = load()
    r = rates(cfg, job.get("speed", cfg["speed"]))
    store = Store(db_path)
    indexed = {os.path.abspath(m["path"]) for m in store.db.execute("SELECT path FROM media")}
    sizes: dict[str, int] = {}
    seen: set[str] = set()
    while not stop.is_set():
        try:
            files = sorted(f for f in os.listdir(job["folder"]) if f.lower().endswith(VIDEO_EXTS))
        except OSError as e:
            state["state"] = f"cannot read the folder: {e}"
            J.save()
            break
        todo = []
        for f in files:
            path = os.path.abspath(os.path.join(job["folder"], f))
            if path in seen or path in indexed:
                continue
            size = os.path.getsize(path)
            if sizes.get(path) == size and size > 0:  # finished downloading
                todo.append(path)
            sizes[path] = size
        waiting = sum(1 for f in files if os.path.abspath(os.path.join(job["folder"], f)) not in seen | indexed)
        state["download"] = (f"{waiting} new file(s) in the folder" if waiting else "no new files") + (
            ", watching for more" if job.get("watch") else "")
        J.save()
        for path in todo:
            if stop.is_set():
                break
            seen.add(path)
            clock = clock_from_name(os.path.basename(path))
            p = {"file": os.path.basename(path), "path": path, "start": clock or "", "end": "", "channel": "",
                 "state": "indexing", "note": "" if clock else "no time in the file name: times are from the video start"}
            state["pieces"].append(p)
            J.save()
            try:
                _index_piece(store, p, job, cfg, r, clock)
            except Exception as e:
                p["state"], p["note"] = "failed", f"indexing: {e}"
            J.save()
        if not todo and not job.get("watch") and waiting == 0:
            break
        stop.wait(5)
    done = sum(p["state"] == "done" for p in state["pieces"])
    state["state"] = "stopped" if stop.is_set() else f"finished: {done} file(s) indexed"
    J.save()
    J.finish()
