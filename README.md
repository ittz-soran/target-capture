# Target Capture

Index a long video (or an image) once, then search it:

* **Plate search**: every license plate in the footage is read; search a number (exact, near-miss, part of the number, or wildcard) and get each time that car was seen, with clock time, vehicle make and model (e.g. Toyota Corolla), colour, and snapshots.
* **Arabic and Kurdish plates** (Iraq, Kurdistan Region): Arabic-Indic digits, Arabic letters, governorate and plate type. Detected automatically by default. Search with either digit style: `١٢٣٤٥` and `12345` find the same car.
* **Person search**: give one or more face photos; get how many times that person appears, and for each appearance the start and end time, duration, match confidence, clothing colours, and snapshots.

## Install

Python 3.10 or newer (3.11 recommended).

### Windows

1. Install Python 3.11 from python.org and tick "Add python.exe to PATH".
2. Install "Microsoft C++ Build Tools" (choose "Desktop development with C++"); InsightFace needs it.
3. Double-click `setup_windows.bat` once, then `run_windows.bat` whenever you want to use it.
4. To get a new version, double-click `update_windows.bat` (needs Git for Windows). Your videos, index and results are kept.

Or by hand in Command Prompt, inside the `target-capture` folder:

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

In PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.
Each time you open a new window, run `.venv\Scripts\activate` again before using the tool.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The models (YOLO11n, the fast-alpr plate detector and OCR, EasyOCR Arabic, CLIP for car models, InsightFace `buffalo_l`) download automatically the first time you run the tool, about 1 GB in total.
If you have an NVIDIA GPU, install the CUDA build of `torch` and `onnxruntime-gpu` for a large speed-up.

## Web page

```bash
streamlit run app.py
```

1. **Upload & index**: upload a file, or paste a path for big DVR files (uploading 5 hours through a browser is slow). Optionally enter the DVR clock time of the first frame so results show real times.
2. **Find a plate**: type the number.
3. **Find a person**: upload a face photo or several.
4. **All faces**: every face found, grouped into numbered people. Click a person to see every time they were seen. A slider sets how alike faces must be to count as the same person.
5. **All plates**: a table of every plate that was read, and below it a gallery of plates that were found but could not be read (other alphabets, blur, too small). Click one to see the whole frame.

## Command line

```bash
# 1. Index (both plates and people by default; --plates or --persons for only one)
python -m targetcapture index /videos/cam1_2026-09-30.mp4 --clock-start "2026-09-30 08:00:00"

# English and Arabic plates are told apart automatically (--plate-script auto)

# 2. Search plates
python -m targetcapture find-plate 12ABC345          # exact or 1 character off
python -m targetcapture find-plate "12?BC*"          # partial plate: ? = one char, * = anything
python -m targetcapture find-plate 34567             # only the number: ignores governorate code and letter
python -m targetcapture find-plate "١٢٣٤٥"           # Arabic-Indic digits work too
python -m targetcapture plates                       # list every plate seen
python -m targetcapture plates --vehicle "white corolla"   # filter by make, model, colour or governorate

# 3. Search a person
python -m targetcapture find-person suspect1.jpg suspect2.jpg

python -m targetcapture media                        # what is indexed
```

Each search writes `results/<query>/report.html` (snapshots and details), plus `results.json` and `results.csv`.
See `examples/` for two finished reports from test clips.

Indexing saves progress every 10 seconds; press Ctrl-C at any time and run the same command again to resume.
Several videos can go into one index (`--db` picks the file, default `targetcapture.db`), and searches cover all of them.

### Useful options

| Option | Default | What it does |
|---|---|---|
| `index --plate-fps` | 4 | Frames per second checked for plates. Raise for fast traffic. |
| `index --person-fps` | 3 | Frames per second checked for people. |
| `index --vehicle-crops` | off | Also reads plates inside each detected vehicle. Slower; helps on wide or 4K views where plates are tiny. |
| `index --plate-script` | auto | `auto` reads English plates with the Latin reader and Arabic plates with the Arabic reader; `latin` is English only (fastest); `arabic` uses the Arabic reader for every plate. |
| `index --no-car-model` | off | Skip make/model recognition (faster). The list of models is `car_models.txt`; edit it to match your roads. |
| `index --no-motion-skip` | off | By default frames where nothing moved are skipped (a still frame is still checked every 5 s). |
| `find-plate --max-dist` | 1 | How many characters may differ. |
| `find-plate --min-reads` | 1 | Ignore one-off reads (OCR noise). |
| `find-person --threshold` | 0.40 | Face similarity needed. Lower finds more but risks wrong people; check "near misses" in the report. |

## Speed

Measured on a 4-core CPU with no GPU: person indexing processed about 8 analysed frames per second on a 768x432 clip with 1 to 3 people in view, and plate indexing took 6 s for a 1-minute 720p clip.
At those rates a 5-hour busy 1080p street video would take a few hours on CPU; a GPU is typically 5 to 20 times faster. Searching an existing index takes seconds.

## Run the tests

```bash
python -m pytest tests                                     # matching logic, no models needed
python tests/make_plate_sample.py car.jpg demo.mp4         # synthetic clip: one car passes twice
```

## Responsible use

Identifying people in footage is regulated in many places (for example GDPR in the EU, and biometric laws in several US states).
Make sure you have a lawful basis for processing the footage and for face matching, keep the index and reports secured, and delete them when no longer needed.
Treat matches as leads for a human to verify, never as proof of identity.

## Licences of the bundled models

* Ultralytics YOLO is AGPL-3.0 (a commercial licence is available from Ultralytics).
* The InsightFace `buffalo_l` face models are for non-commercial research use only. For commercial use, swap in a commercially licensed face model (see DESIGN.md).
* fast-alpr is MIT.
