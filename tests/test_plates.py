"""Fast checks for plate matching and grouping (no models needed): python -m pytest tests"""
from targetcapture.plates import fold, group_sightings, match_score


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
