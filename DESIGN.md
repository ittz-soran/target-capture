# Design note

## Shape: index once, search many times

Running detectors over 5 hours of video is the expensive part, and you usually want several searches over the same footage.
So the work is split in two:

1. **Index**: walk the video, detect everything of interest, and store compact records (text, boxes, face embeddings, times) in SQLite. No images are stored.
2. **Search**: query the database (milliseconds), then pull only the matching frames back out of the original video to build the report.

## Making multi-hour video practical

* **Frame sampling**: analyse 3 to 4 frames per second instead of 25 to 30. A car or person is on screen for seconds, so they are still seen several times.
* **Motion gating**: DVR footage is mostly an empty scene. Frames that barely differ from the last analysed one are skipped, with a forced check every 5 seconds.
* **Resumable**: progress is committed every 10 seconds, so a crash or Ctrl-C loses at most a few seconds of work.

## Plates

| Step | Tool |
|---|---|
| Plate detection | fast-alpr YOLOv9-t plate detector (ONNX) |
| Plate reading (OCR) | fast-alpr `cct-s-v2-global` model, trained on plates from many countries |
| Vehicle type and colour | YOLO11n (COCO car / truck / bus / motorcycle) + HSV colour naming |

CCTV OCR is noisy, so search does not depend on a perfect read:

* look-alike characters are folded together before comparing (O/0/D/Q, I/1/L, B/8, S/5, Z/2, G/6);
* an edit distance (default 1) catches one wrong or missing character;
* wildcards (`12?BC*`) handle a half-remembered plate;
* reads of the same plate within 4 seconds merge into one **sighting**, with a confidence-weighted vote for the final text and a list of every OCR variant seen.

### Arabic / Kurdish plates (Iraq, Kurdistan Region)

`--plate-script arabic` keeps the same plate detector but reads the crop with EasyOCR (Arabic + English).
`arabic.py` then converts Arabic-Indic and Persian digits to 0-9, unifies letter variants (أ/إ/آ, ى/ي, ة/ه),
moves governorate names (Arabic, Kurdish, English, with fuzzy matching for OCR slips), "IRQ"/"KRG" badges and
plate types (private, taxi, truck...) into the details, and drops the second copy of the number when a plate
prints it in both digit styles. Search folds Arabic letters to a Latin key, so `21 B 34567` finds `21 ب 34567`,
and a number alone finds the plate even when it also carries a governorate code and letter.
Tested on rendered Iraqi and Kurdistan style plates only; real plates are needed to tune it further.

### Car make and model

A CLIP model (open_clip ViT-B/32, LAION-400M) compares each plate's vehicle crop with the text
"a photo of a <make model>" for every line of `car_models.txt`, plus catch-all classes (bus, lorry, blurry
vehicle, generic car) so it says nothing rather than guess wildly. Reads of one sighting vote, and the report
shows the winner, its confidence, and the runners-up. It is reliable on make and body shape, less so on
twin models (it called a Skoda Citigo a VW up!, which is the same car with another badge) or model years.
A model trained on local cars would be the upgrade.

## People

| Step | Tool |
|---|---|
| Person detection | YOLO11n, class "person" |
| Tracking (one walk past the camera = one track) | ByteTrack (built into Ultralytics) |
| Face detection | InsightFace SCRFD, run on the head area of each person box, upscaled, so small faces in wide shots are found |
| Face identity | InsightFace ArcFace (`buffalo_l`), 512-d embeddings, cosine similarity |

Search compares the query face(s) with every stored face.
A track matches when its best face is above the threshold, and the sighting covers the whole track, so the reported time on camera includes moments when the face was turned away.
Matching tracks less than 3 seconds apart merge into one sighting, since trackers sometimes split one walk into two.
The number of sightings is the "how many times" answer.

## Known limits and next steps

* **No face visible, no match.** People seen only from behind are not found. Next step: add a whole-body re-identification model (e.g. OSNet via torchreid) to link tracks by clothing and build, as a fallback when faces are missing.
* **Small plates and faces.** Below about 20 px wide, plates and faces cannot be read reliably. Use `--vehicle-crops` for plates on wide views; higher resolution cameras help most.
* **Night and IR footage** lowers OCR and face accuracy; plates with reflective IR are usually fine.
* **Region-specific plates** (non-Latin characters, e.g. Arabic or Persian digits) need an OCR model trained on that country's plates. fast-alpr can be fine-tuned, or the OCR step swapped (it is one function in `plates.py`).
* **Licensing.** For commercial use, replace the `buffalo_l` face models (research-only) and consider the Ultralytics licence (AGPL-3.0).
* **Scale-out.** For many cameras, run indexing as a background worker per video and share one database (or move to PostgreSQL + a vector index such as pgvector or FAISS for faces).

## Code map

```
targetcapture/
  video.py     frame sampling, motion gating, frame lookup
  store.py     SQLite schema and queries, clock-time conversion
  plates.py    plate indexer, look-alike folding, fuzzy matching, sighting grouping
  persons.py   person tracking, face embedding, face search and sighting grouping
  pipeline.py  indexing loop shared by CLI and web page (progress, resume)
  report.py    HTML / JSON / CSV reports with snapshots
  cli.py       command line
app.py         Streamlit web page
tests/         unit tests and a synthetic test clip generator
examples/      two finished reports from test clips
```
