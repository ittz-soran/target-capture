"""Web UI: upload a video or image, index it, then search plates or people.

    streamlit run app.py
"""
import os
import tempfile

import streamlit as st

from targetcapture import persons as P
from targetcapture import plates as L
from targetcapture import teach as T
from targetcapture.pipeline import index_media
from targetcapture.report import person_report, plate_report, thumbnail
from targetcapture.store import Store, clock_time, fmt_offset

UPLOADS = "uploads"
RESULTS = "results"
THUMBS = os.path.join(RESULTS, "_thumbs")
PAGE = 36

st.set_page_config(page_title="Target Capture", layout="wide")
st.title("Target Capture")
db_path = st.sidebar.text_input("Index database", "targetcapture.db")
store = Store(db_path)

@st.cache_resource
def device_line():
    from targetcapture.gpu import describe

    return describe()


with st.sidebar:
    st.caption(device_line())
    st.subheader("Indexed media")
    for m in store.media():
        dur = fmt_offset(m["frames"] / m["fps"]) if m["kind"] == "video" else "image"
        st.caption(f"{os.path.basename(m['path'])} · {dur} · plates {'✓' if m['plates_done'] else '…'}"
                   f" · people {'✓' if m['persons_done'] else '…'}")


@st.cache_resource
def face_model():
    return P.FaceModel()


def show_rows(rows, out_dir, image_keys):
    for r in rows:
        with st.container(border=True):
            cols = st.columns([2, 1, 2])
            if r.get("frame_image"):
                cols[0].image(os.path.join(out_dir, r["frame_image"]))
            for k in image_keys:
                if r.get(k):
                    cols[1].image(os.path.join(out_dir, r[k]))
            cols[2].json({k: v for k, v in r.items() if not k.endswith("_image")}, expanded=True)


def pager(n, key):
    """Page picker for long galleries; returns the slice to show."""
    pages = max(1, (n + PAGE - 1) // PAGE)
    page = st.number_input(f"Page (of {pages})", 1, pages, 1, key=key) if pages > 1 else 1
    return slice((page - 1) * PAGE, page * PAGE)


@st.cache_data(show_spinner="Grouping faces into people…")
def cached_identities(db, db_mtime, threshold):
    ppl = P.identities(Store(db), threshold=threshold)
    for p in ppl:
        p["embedding"] = p["embedding"].tolist()
        p["best"] = dict(p["best"])
        p["best"].pop("emb", None)
        for t in p["tracks"].values():
            t["best"] = dict(t["best"])
            t["best"].pop("emb", None)
    return ppl


tab_index, tab_plate, tab_person, tab_faces, tab_all, tab_teach, tab_tips = st.tabs(
    ["1 · Upload & index", "2 · Find a plate", "3 · Find a person", "All faces", "All plates", "Teach",
     "Best results"])

with tab_index:
    up = st.file_uploader("Video or image", type=["mp4", "avi", "mkv", "mov", "dav", "h264", "ts",
                                                  "jpg", "jpeg", "png", "bmp", "webp"])
    path_in = st.text_input("…or a path to a file already on this machine (best for multi-hour DVR files)")
    c1, c2, c3 = st.columns(3)
    do_plates = c1.checkbox("Read plates", True)
    do_persons = c2.checkbox("Find people", True)
    clock = c3.text_input("DVR clock at first frame (optional)", placeholder="2026-09-30 08:00:00")
    with st.expander("Advanced"):
        plate_fps = st.slider("Frames per second analysed for plates", 1.0, 10.0, 4.0)
        person_fps = st.slider("Frames per second analysed for people", 1.0, 10.0, 3.0)
        plate_script = st.radio("Plate script", ["auto", "latin", "arabic"], horizontal=True,
                                format_func=lambda v: {"auto": "Auto (English and Arabic)",
                                                       "latin": "English only (fastest)",
                                                       "arabic": "Arabic / Kurdish only"}[v])
        car_models = st.checkbox("Recognise car make and model (Toyota Corolla, Nissan Altima...)", True)
        vehicle_crops = st.checkbox("Also read plates inside each vehicle (slower; wide or 4K views)")
        reindex = st.checkbox("Re-index from scratch")
    if st.button("Start indexing", type="primary", disabled=not (up or path_in)):
        if up:
            os.makedirs(UPLOADS, exist_ok=True)
            path = os.path.join(UPLOADS, up.name)
            with open(path, "wb") as f:
                f.write(up.getbuffer())
        else:
            path = path_in
        bars = {}

        def progress(kind, i, total, found, t):
            if kind not in bars:
                bars[kind] = st.progress(0.0)
            bars[kind].progress(min(1.0, i / max(1, total)),
                                text=f"{kind}: {fmt_offset(t)} analysed, {found} detections")

        res = index_media(store, path, plates=do_plates, persons=do_persons, plate_fps=plate_fps,
                          person_fps=person_fps, clock_start=clock or None, vehicle_crops=vehicle_crops, plate_script=plate_script, car_models=car_models,
                          reindex=reindex, progress=progress)
        st.success(f"Indexed {os.path.basename(path)}. Now search it in the next tabs.")

with tab_plate:
    q = st.text_input("Plate number (wildcards allowed: 12?BC* )")
    c1, c2 = st.columns(2)
    max_dist = c1.number_input("Allowed character differences", 0, 3, 1)
    min_reads = c2.number_input("Minimum reads per sighting", 1, 10, 1)
    if st.button("Search plate", disabled=not q):
        hits = L.search(store, q, max_dist=max_dist, min_reads=min_reads)
        out = os.path.join(RESULTS, "plate_" + "".join(ch for ch in q if ch.isalnum()))
        rows = plate_report(store, q, hits, out)
        st.subheader(f"{len(rows)} sighting(s)")
        st.caption(f"Full report saved to {os.path.join(out, 'report.html')}")
        show_rows(rows, out, ["plate_image"])

with tab_person:
    photos = st.file_uploader("Photo(s) of the person's face", type=["jpg", "jpeg", "png", "webp"],
                              accept_multiple_files=True)
    threshold = st.slider("Match threshold (higher = stricter)", 0.25, 0.7, 0.40, 0.01)
    if st.button("Search person", disabled=not photos):
        tmp = tempfile.mkdtemp()
        paths = []
        for p in photos:
            paths.append(os.path.join(tmp, p.name))
            with open(paths[-1], "wb") as f:
                f.write(p.getbuffer())
        try:
            embs = [face_model().query_embedding(p) for p in paths]
        except ValueError as e:
            st.error(str(e))
            st.stop()
        sightings, stats = P.search(store, embs, threshold=threshold)
        out = os.path.join(RESULTS, "person_" + os.path.splitext(photos[0].name)[0])
        rows = person_report(store, paths, sightings, stats, threshold, out)
        st.subheader(f"Seen {len(rows)} time(s)")
        st.caption(f"{stats.get('faces_indexed', 0)} faces in the index · {stats.get('near_misses', 0)} "
                   f"near misses just below the threshold · report: {os.path.join(out, 'report.html')}")
        show_rows(rows, out, ["face_image"])

with tab_faces:
    st.caption("Every face found in the indexed videos, grouped into people. Click a person to see "
               "every time they were seen.")
    c1, c2 = st.columns(2)
    f_thr = c1.slider("How alike faces must be to count as the same person", 0.35, 0.65, 0.45, 0.01,
                      help="Lower merges more (risk: two people become one). Higher splits more.")
    hide_oneoff = c2.checkbox("Hide people only caught in single frames (small or side-on faces)", True)
    db_mtime = os.path.getmtime(db_path) if os.path.exists(db_path) else 0
    people = [p for p in cached_identities(db_path, db_mtime, f_thr)
              if not hide_oneoff or p["max_track_faces"] >= 3]
    sel = st.session_state.get("face_sel")
    if sel is not None:
        person = next((p for p in people if p["id"] == sel), None)
        if person:
            tracks = {tuple(k) if isinstance(k, list) else k: v for k, v in person["tracks"].items()}
            sightings = P.person_sightings(store, {**person, "tracks": tracks}, threshold=f_thr)
            out = os.path.join(RESULTS, f"face_person_{sel}")
            rows = person_report(store, [f"person #{sel}"], sightings,
                                 {"faces_indexed": person["n_faces"]}, f_thr, out)
            hdr = st.columns([4, 1])
            hdr[0].subheader(f"Person #{sel}: seen {len(rows)} time(s), {person['n_faces']} face detections")
            if hdr[1].button("Close", key="face_close"):
                st.session_state.face_sel = None
                st.rerun()
            st.caption(f"Report: {os.path.join(out, 'report.html')}")
            show_rows(rows, out, ["face_image"])
            st.divider()
    st.subheader(f"{len(people)} people")
    cols = st.columns(6)
    for i, p in enumerate(people[pager(len(people), "face_page")]):
        b = p["best"]
        media = store.media(b["media_id"])
        with cols[i % 6]:
            img = thumbnail(b, media["path"], THUMBS, box=("fx1", "fy1", "fx2", "fy2"), pad=0.35, min_w=140)
            if img:
                st.image(img, use_container_width=True)
            tracks = {tuple(k) if isinstance(k, list) else k: v for k, v in p["tracks"].items()}
            seen = len(P.person_sightings(store, {**p, "tracks": tracks}, threshold=f_thr))
            st.caption(f"#{p['id']} · seen {seen}x · first at {fmt_offset(p['first'])}")
            if st.button(f"Show #{p['id']}", key=f"face_{p['id']}"):
                st.session_state.face_sel = p["id"]
                st.rerun()

with tab_all:
    st.subheader("Read plates")
    min_r = st.number_input("Hide plates read fewer times than", 1, 10, 2)
    vfilter = st.text_input("Filter by vehicle (e.g. corolla, white toyota, Erbil)").lower().split()
    sightings = [s for s in L.group_sightings(store.plate_reads()) if s["reads"] >= min_r and all(
        w in " ".join(str(s[k] or "") for k in ("vehicle_model", "vehicle_color", "vehicle_type", "region")).lower()
        for w in vfilter)]
    st.dataframe([{
        "plate": s["text"], "start": fmt_offset(s["start"]), "end": fmt_offset(s["end"]),
        "clock": clock_time(store.media(s["media_id"]), s["start"]), "reads": s["reads"],
        "conf": s["conf"], "vehicle": f"{s['vehicle_color'] or ''} {s['vehicle_model'] or s['vehicle_type'] or ''}".strip(),
        "details": s["region"],
        "source": {"taught": "typed by you", "learned": "learned"}.get(s["source"], "read"),
        "video": os.path.basename(store.media(s["media_id"])["path"]),
    } for s in sightings], use_container_width=True)

    unread = L.group_unread(store.plate_reads(unread=True))
    st.subheader(f"Plates that could not be read ({len(unread)})")
    st.caption("Plates the system found but could not read (other alphabets, blur, too small). "
               "Click one to see the whole frame.")
    sel = st.session_state.get("cap_sel")
    if sel is not None and sel < len(unread):
        c = unread[sel]
        media = store.media(c["media_id"])
        img = thumbnail(c["best"], media["path"], THUMBS, full=True)
        hdr = st.columns([4, 1])
        hdr[0].markdown(f"**Capture {sel + 1}** · {fmt_offset(c['start'])}"
                        f"{' (' + clock_time(media, c['start']) + ')' if clock_time(media, c['start']) else ''}"
                        f" · {os.path.basename(media['path'])}")
        if hdr[1].button("Close", key="cap_close"):
            st.session_state.cap_sel = None
            st.rerun()
        if img:
            st.image(img)
    cols = st.columns(6)
    sl = pager(len(unread), "cap_page")
    for i, c in enumerate(unread[sl]):
        n = (sl.start or 0) + i
        media = store.media(c["media_id"])
        with cols[i % 6]:
            img = thumbnail(c["best"], media["path"], THUMBS, pad=0.1, min_w=200)
            if img:
                st.image(img, use_container_width=True)
            vehicle = f"{c['vehicle_color'] or ''} {c['vehicle_model'] or c['vehicle_type'] or ''}".strip()
            st.caption(f"{fmt_offset(c['start'])} · {vehicle or 'vehicle ?'}"
                       f"{' · seen ' + str(c['detections']) + 'x' if c['detections'] > 1 else ''}")
            if st.button("Open", key=f"cap_{n}"):
                st.session_state.cap_sel = n
                st.rerun()

with tab_teach:
    st.caption("Help the system: type the correct text for plates it could not read or read wrongly. "
               "Your answers are searchable at once, and the system uses them to recognise the same plate "
               "(and the same reading mistake) in the future.")
    mode = st.radio("What to check", ["unread", "check", "find"], horizontal=True, key="teach_mode",
                    format_func=lambda m: {"unread": "Plates it could not read",
                                           "check": "Check read plates (least sure first)",
                                           "find": "Correct a specific plate"}[m])
    if mode == "unread":
        queue = L.group_unread(store.plate_reads(unread=True))
    else:
        queue = [s for s in L.group_sightings(store.plate_reads()) if s["source"] != "taught"]
        if mode == "find":
            q = st.text_input("Plate as the system shows it (from the All plates table)", key="teach_find")
            queue = [s for s in queue if q and L.match_score(q, s["text"], 1)[0]]
        else:
            queue.sort(key=lambda s: (s["conf"], s["reads"]))

    last = st.session_state.get("teach_last")
    if last:  # follow-ups to the correction just saved
        with st.container(border=True):
            st.success(f"Saved **{last['label'] or 'not a plate'}**.")
            if last["same_mistake"]:
                n = len(last["same_mistake"])
                if st.button(f"Also fix {n} other read(s) that were read as \"{last['ocr_text']}\""):
                    store.set_plate_text(last["same_mistake"], last["label"], "taught")
                    last["same_mistake"] = []
                    st.rerun()
            if last["sig"] is not None and last["label"]:
                pool = L.group_unread(store.plate_reads(unread=True)) + [
                    s for s in L.group_sightings(store.plate_reads()) if s["source"] is None and s["conf"] < 0.9]
                sims = T.similar_captures(store, last["sig"], pool)[:12]
                if sims:
                    st.markdown(f"These look like the same plate. Tick the ones that are **{last['label']}**:")
                    cols = st.columns(6)
                    picked = []
                    for i, (sim, c) in enumerate(sims):
                        media = store.media(c["media_id"])
                        with cols[i % 6]:
                            img = thumbnail(c["best"], media["path"], THUMBS, pad=0.1, min_w=200)
                            if img:
                                st.image(img, use_container_width=True)
                            if st.checkbox(f"{fmt_offset(c['start'])} · {sim:.0%} alike", key=f"sim_{i}_{c['ids'][:1]}"):
                                picked.append(c)
                    if picked and st.button(f"Apply {last['label']} to {len(picked)} selected"):
                        for c in picked:
                            store.set_plate_text(c["ids"], last["label"], "taught")
                        st.session_state.teach_last = None
                        st.rerun()
            if st.button("Done", key="teach_done"):
                st.session_state.teach_last = None
                st.rerun()

    if not queue:
        st.info("Nothing to check here." if mode != "find" else "Type a plate above to find it.")
    else:
        i = min(st.session_state.get(f"teach_i_{mode}", 0), len(queue) - 1)
        c = queue[i]
        media = store.media(c["media_id"])
        clock = clock_time(media, c["start"])
        st.markdown(f"**{i + 1} of {len(queue)}** · {fmt_offset(c['start'])}{' (' + clock + ')' if clock else ''}"
                    f" · {os.path.basename(media['path'])}")
        left, right = st.columns([1, 2])
        crop = thumbnail(c["best"], media["path"], THUMBS, pad=0.1, min_w=360)
        frame = thumbnail(c["best"], media["path"], THUMBS, full=True)
        if crop:
            left.image(crop, use_container_width=True)
        if frame:
            right.image(frame, use_container_width=True)
        current = c.get("text") or ""
        guess = current or c.get("ocr_guess") or ""
        vehicle = f"{c['vehicle_color'] or ''} {c['vehicle_model'] or c['vehicle_type'] or ''}".strip()
        left.caption(f"System read: **{current or 'nothing'}**"
                     f"{' · guess: ' + c['ocr_guess'] if not current and c.get('ocr_guess') else ''}"
                     f"{' · ' + vehicle if vehicle else ''}")
        with left.form(f"teach_form_{mode}_{i}"):
            label = st.text_input("Correct plate text (Arabic or English digits both fine)", value=guess)
            b1, b2, b3 = st.columns(3)
            save = b1.form_submit_button("Save", type="primary")
            notplate = b2.form_submit_button("No plate")
            skip = b3.form_submit_button("Skip")
        if save or notplate:
            if save and not L.normalize(label):
                st.warning("Type the plate text first, or click No plate.")
            else:
                st.session_state.teach_last = T.teach(store, c["ids"], c["best"], label, not_a_plate=bool(notplate))
                st.rerun()
        if skip:
            st.session_state[f"teach_i_{mode}"] = (i + 1) % len(queue)
            st.rerun()
    with st.expander("Export what you taught (for training a better reader)"):
        n_labels = len([lab for lab in store.labels() if lab["label"]])
        st.caption(f"{n_labels} plate(s) taught so far.")
        if st.button("Export to results/taught_plates", disabled=not n_labels):
            n = T.export(store, os.path.join(RESULTS, "taught_plates"))
            st.success(f"Wrote {n} plate pictures and labels.csv to {os.path.join(RESULTS, 'taught_plates')}")

with tab_tips:
    st.markdown(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "TIPS.md"), encoding="utf-8").read())
