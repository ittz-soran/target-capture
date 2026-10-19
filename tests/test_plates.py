"""Fast checks for plate matching and grouping (no models needed): python -m pytest tests"""
from targetcapture.plates import fold, group_sightings, match_score, search


def test_fold_lookalikes():
    assert fold("12-abc 345") == fold("I2A8C345")


def test_match_exact_fuzzy_wildcard():
    assert match_score("12ABC345", "12ABC345", 1) == (True, 0)
    assert match_score("12ABC345", "12ABC346", 1) == (True, 1)
    assert match_score("12ABC345", "99XYZ000", 1)[0] is False
    assert match_score("12?BC*", "12ABC345", 0)[0] is True


def test_group_sightings_merges_noisy_reads():
    base = dict(media_id=1, region=None, vehicle_type="car", vehicle_color="white")
    reads = [dict(base, t=10.0, text="12ABC345", conf=0.9), dict(base, t=10.5, text="12A8C345", conf=0.6),
             dict(base, t=11.0, text="12ABC345", conf=0.95), dict(base, t=60.0, text="12ABC345", conf=0.9)]
    s = group_sightings(reads, gap=4.0)
    assert len(s) == 2
    assert s[0]["text"] == "12ABC345" and s[0]["reads"] == 3


def test_arabic_digits_and_partial_number():
    # "١٢٣٤٥" is 12345 in Arabic-Indic digits; the plate also has a governorate code and letter.
    assert match_score("١٢٣٤٥", "12345", 0)[0]
    assert match_score("34567", "21 ب 34567", 1)[0]
    assert match_score("21 B 34567", "21 ب 34567", 0)[0]  # Latin B for Arabic ب


def test_parse_iraqi_plate_tokens():
    from targetcapture.arabic import parse_tokens

    text, d = parse_tokens([(10, "اربل"), (20, "خصوصي"), (30, "١٢٣٤٥"), (30, "12345")])
    assert text == "12345" and d["governorate"] == "Erbil" and d["plate_type"] == "private"
    text, d = parse_tokens([(5, "IRQ"), (40, "بغداد"), (50, "21"), (60, "ب"), (70, "34567")])
    assert text == "21 ب 34567" and d["governorate"] == "Baghdad"


def test_taught_mistake_is_applied_to_new_reads():
    from targetcapture.teach import Memory

    class FakeStore:
        def labels(self):
            return [{"ocr_text": "07VA9", "label": "56789", "sig": None}]

    m = Memory(FakeStore())
    r = m.apply({"text": "07VA9", "conf": 0.99})
    assert r["text"] == "56789" and r["source"] == "learned" and r["ocr_guess"] == "07VA9"
    assert m.apply({"text": "12345", "conf": 0.99})["text"] == "12345"


def test_plate_pictures_tolerate_small_shifts():
    import numpy as np
    from targetcapture.teach import plate_signature, signature_similarity

    rng = np.random.default_rng(0)
    plate = (rng.random((60, 200)) * 255).astype(np.uint8)
    shifted = np.roll(plate, 3, axis=1)
    other = (rng.random((60, 200)) * 255).astype(np.uint8)
    a, b, c = plate_signature(plate), plate_signature(shifted), plate_signature(other)
    assert signature_similarity(a, b) > 0.8 > signature_similarity(a, c)


def test_one_car_one_row_with_best_and_second_capture():
    def r(t, text, conf, x, w):
        return dict(media_id=1, region=None, vehicle_type="car", vehicle_color="white", t=t, text=text,
                    conf=conf, x1=x, y1=300, x2=x + w, y2=300 + w // 3)
    reads = [r(10.0, "5AU5", 0.7, 600, 60), r(10.25, "5AU5341", 0.8, 590, 80), r(10.5, "5AU5841", 0.6, 580, 100),
             r(10.75, "5AU5341", 0.95, 570, 130), r(11.5, "5AU5341", 0.9, 560, 150),
             r(12.0, "7KD2210", 0.9, 100, 120)]  # another car elsewhere in the picture
    s = group_sightings(reads)
    assert [x["text"] for x in s] == ["5AU5341", "7KD2210"]
    assert s[0]["reads"] == 5
    assert s[0]["best"]["t"] == 10.75 and s[0]["second"]["t"] == 11.5


def test_remap_upscaled_crop_box_to_full_frame():
    from targetcapture.plates import _remap
    # crop taken at (x=500, y=300), upscaled 2x; a plate box at (40,10)-(120,34) in the upscaled crop
    r = {"x1": 40, "y1": 10, "x2": 120, "y2": 34}
    _remap(r, 2.0, 500, 300)
    assert (r["x1"], r["y1"], r["x2"], r["y2"]) == (520, 305, 560, 317)
    # sc=1.0 is the old native-scale behaviour: just the crop offset
    r2 = {"x1": 5, "y1": 7, "x2": 9, "y2": 11}
    _remap(r2, 1.0, 100, 200)
    assert (r2["x1"], r2["y1"], r2["x2"], r2["y2"]) == (105, 207, 109, 211)


def test_search_with_pregrouped_sightings_matches_and_does_not_mutate():
    """The UI passes a cached, pre-grouped list to search(); search must give the same result
    as grouping itself, and must not add its distance/exact/partial fields to that shared list."""
    base = dict(media_id=1, region=None, vehicle_type="car", vehicle_color="white")
    reads = [dict(base, t=10.0, text="12ABC345", conf=0.9), dict(base, t=10.5, text="12A8C345", conf=0.6),
             dict(base, t=60.0, text="7KD2210", conf=0.9)]
    grouped = group_sightings(reads)

    class FakeStore:  # search() must not touch the store when sightings are supplied
        def plate_reads(self):
            raise AssertionError("should use the supplied sightings, not re-read the store")

    hits = search(FakeStore(), "12ABC345", sightings=grouped)
    assert [h["text"] for h in hits] == ["12ABC345"]
    assert hits[0]["exact"] is True and hits[0]["distance"] == 0
    # the shared/cached list is untouched
    assert all("distance" not in s and "exact" not in s and "partial" not in s for s in grouped)
