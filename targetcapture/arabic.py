"""Arabic-script plates (Iraq and the Kurdistan Region first).

Iraqi plates mix scripts: Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩) next to Western digits,
an Arabic or Latin series letter, and the governorate or region written in Arabic,
Kurdish and/or English ("بغداد", "اربيل", "ERBIL", "KRG", "IRAQ"). This module

* reads plate crops with EasyOCR (Arabic + English models),
* converts every digit to 0-9 and unifies letter variants (أ/إ/آ -> ا, ى -> ي, ة -> ه),
* pulls governorate / region / plate-type words out into details instead of the plate text,
* drops the duplicate number when a plate prints it in both digit styles.
"""
from __future__ import annotations

import re
from collections import Counter

import cv2
import numpy as np

_DIGITS = {**{chr(0x0660 + i): str(i) for i in range(10)},   # Arabic-Indic
           **{chr(0x06F0 + i): str(i) for i in range(10)}}   # Persian / Kurdish (Eastern Arabic-Indic)
_LETTER_VARIANTS = {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ی": "ي", "ة": "ه", "ک": "ك", "ؤ": "و",
                    "ئ": "ي"}
_DIACRITICS = re.compile("[ً-ٰٟـ]")  # harakat and tatweel

# Arabic letter -> Latin key, used only for matching so "A 12345" finds "أ 12345".
_TO_LATIN = {"ا": "A", "ب": "B", "ت": "T", "ث": "T", "ج": "J", "ح": "H", "خ": "K", "د": "D", "ذ": "D",
             "ر": "R", "ز": "Z", "س": "S", "ش": "S", "ص": "S", "ض": "D", "ط": "T", "ظ": "Z", "ع": "E",
             "غ": "G", "ف": "F", "ق": "Q", "ك": "K", "ل": "L", "م": "M", "ن": "N", "ه": "H", "و": "W",
             "ي": "Y", "پ": "P", "چ": "C", "ژ": "J", "گ": "G", "ڤ": "V", "ڵ": "L", "ڕ": "R", "ۆ": "O",
             "ێ": "E", "ە": "H"}

# Words that describe the plate rather than identify it (after normalisation, upper-case for Latin).
GOVERNORATES = {
    "بغداد": "Baghdad", "BAGHDAD": "Baghdad", "البصره": "Basra", "بصره": "Basra", "BASRA": "Basra",
    "نينوى": "Nineveh", "نينوي": "Nineveh", "NINEVEH": "Nineveh", "NINAWA": "Nineveh",
    "اربيل": "Erbil", "هولير": "Erbil", "ERBIL": "Erbil", "ARBIL": "Erbil", "HAWLER": "Erbil",
    "السليمانيه": "Sulaymaniyah", "سليمانيه": "Sulaymaniyah", "سليماني": "Sulaymaniyah",
    "SULAIMANI": "Sulaymaniyah", "SULAYMANIYAH": "Sulaymaniyah", "SLEMANI": "Sulaymaniyah",
    "دهوك": "Duhok", "DUHOK": "Duhok", "DOHUK": "Duhok", "كركوك": "Kirkuk", "KIRKUK": "Kirkuk",
    "حلبجه": "Halabja", "HALABJA": "Halabja", "الانبار": "Anbar", "انبار": "Anbar", "ANBAR": "Anbar",
    "ديالى": "Diyala", "ديالي": "Diyala", "DIYALA": "Diyala", "صلاح الدين": "Salahaddin",
    "SALAHADDIN": "Salahaddin", "بابل": "Babil", "BABIL": "Babil", "BABYLON": "Babil",
    "كربلاء": "Karbala", "كربلا": "Karbala", "KARBALA": "Karbala", "النجف": "Najaf", "نجف": "Najaf",
    "NAJAF": "Najaf", "واسط": "Wasit", "WASIT": "Wasit", "القادسيه": "Qadisiyah", "QADISIYAH": "Qadisiyah",
    "الديوانيه": "Qadisiyah", "ميسان": "Maysan", "MAYSAN": "Maysan", "ذي قار": "Dhi Qar", "ذيقار": "Dhi Qar",
    "DHIQAR": "Dhi Qar", "المثنى": "Muthanna", "المثني": "Muthanna", "MUTHANNA": "Muthanna",
    # Kurdish spellings and the 3-letter codes printed on Kurdistan Region plates
    "سليمانيه": "Sulaymaniyah", "سلێماني": "Sulaymaniyah", "دهۆك": "Duhok", "دهوک": "Duhok",
    "کەرکووک": "Kirkuk", "كەركووك": "Kirkuk", "هەڵەبجە": "Halabja",
    "SUL": "Sulaymaniyah", "SLE": "Sulaymaniyah", "SLM": "Sulaymaniyah", "ERB": "Erbil", "EBL": "Erbil",
    "HWL": "Erbil", "DUH": "Duhok", "DHK": "Duhok", "KRK": "Kirkuk", "HLB": "Halabja", "HAL": "Halabja",
}
# Codes that may stick to the plate number in an English read ("SUL21E41707")
PLATE_CODES = {k: v for k, v in GOVERNORATES.items() if len(k) == 3 and k.isascii()}
REGION_WORDS = {"العراق": "Iraq", "IRAQ": "Iraq", "IRQ": "Iraq", "كوردستان": "Kurdistan",
                "كردستان": "Kurdistan", "KURDISTAN": "Kurdistan", "KRG": "Kurdistan", "KRI": "Kurdistan",
                "اقليم": "Kurdistan", "هريم": "Kurdistan", "هەرێم": "Kurdistan"}
TYPE_WORDS = {"خصوصي": "private", "PRIVATE": "private", "اجره": "taxi", "اجرة": "taxi", "TAXI": "taxi",
              "حمل": "truck", "TRUCK": "truck", "حكومي": "government", "GOVERNMENT": "government",
              "مؤقت": "temporary", "موقت": "temporary", "TEMPORARY": "temporary", "تصدير": "export",
              "EXPORT": "export", "دراجه": "motorcycle", "تايبەت": "private"}


def normalize_chars(text: str) -> str:
    text = _DIACRITICS.sub("", text)
    return "".join(_DIGITS.get(c, _LETTER_VARIANTS.get(c, c)) for c in text).upper()


def to_latin_key(text: str) -> str:
    return "".join(_TO_LATIN.get(c, c) for c in text)


# Arabic-script letters (Arabic, Persian, Kurdish), without punctuation such as ؟ ، ؛
AR = "\u0621-\u063A\u0641-\u064A\u0671-\u06D5"
_AR_LETTER = re.compile(f"[{AR}]")


def is_arabic(c: str) -> bool:
    """Arabic-script letter or Arabic-Indic / Persian digit."""
    return bool(_AR_LETTER.match(c)) or "\u0660" <= c <= "\u0669" or "\u06F0" <= c <= "\u06F9"


def has_arabic(text: str) -> bool:
    """True when OCR output contains Arabic-script letters or Arabic-Indic digits."""
    return sum(is_arabic(c) for c in text or "") >= 2


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _lookup(word: str, table: dict) -> str | None:
    """Exact or near match (OCR often drops or swaps one letter: "اربل" for "اربيل")."""
    if word in table:
        return table[word]
    if len(word) < 4:
        return None
    best = min(table, key=lambda k: _levenshtein(word, k))
    return table[best] if _levenshtein(word, best) <= max(1, round(len(word) * 0.34)) else None


_LATIN_NOISE = {"IRQ", "IR", "RQ", "IQ", "KR", "RG"}  # fragments of "IRQ" / "KRG" badges


def parse_tokens(tokens: list[tuple[float, str]]) -> tuple[str, dict]:
    """tokens: (x_position, text) from OCR. Returns (plate_text, details)."""
    details: dict = {}
    words: list[tuple[float, str, bool]] = []  # (x, word, written in Arabic-Indic digits)
    for x, raw in tokens:
        eastern = any(c in _DIGITS for c in raw)
        t = re.sub(f"[^0-9A-Z{AR} ]", " ", normalize_chars(raw))
        for w in t.split():
            # OCR often glues digits and letters together ("12345أ"): split on script changes
            for part in re.findall(f"[0-9]+|[A-Z]+|[{AR}]+", w):
                words.append((x, part, eastern and part.isdigit()))
    joined = " ".join(w for _, w, _ in words)
    for name, en in GOVERNORATES.items():  # two-word names
        if " " in name and name in joined:
            details["governorate"] = en
            words = [t for t in words if t[1] not in name.split()]
    keep: list[tuple[float, str, bool]] = []
    for x, w, eastern in words:
        if w.isdigit():
            keep.append((x, w, eastern))
        elif (g := _lookup(w, GOVERNORATES)):
            details.setdefault("governorate", g)
        elif (r := _lookup(w, REGION_WORDS)):
            details.setdefault("region", r)
        elif (k := _lookup(w, TYPE_WORDS)):
            details.setdefault("plate_type", k)
        elif w in _LATIN_NOISE:
            details.setdefault("region", "Iraq" if "Q" in w else "Kurdistan")
        elif len(w) == 1:  # the series letter
            keep.append((x, w, False))
        else:
            details.setdefault("other_text", []).append(w)
    # Many Iraqi plates print the number twice (Arabic-Indic and Western digits). The Western
    # read is more reliable, so drop Arabic-Indic groups whose digits it already contains.
    western = Counter("".join(w for _, w, e in keep if w.isdigit() and not e))
    eastern = Counter("".join(w for _, w, e in keep if e))
    if sum(western.values()) >= 3 and eastern and not (eastern - western):
        keep = [t for t in keep if not t[2]]
    seen, out = set(), []
    for _, w, _ in sorted(keep, key=lambda t: t[0]):
        if w.isdigit() and len(w) >= 3 and w in seen:
            continue
        seen.add(w)
        out.append(w)
    return " ".join(out), details


class ArabicPlateOCR:
    """EasyOCR (Arabic + English) on plate crops."""

    def __init__(self, gpu: bool | None = None):
        import easyocr

        if gpu is None:
            try:
                import torch

                gpu = torch.cuda.is_available()
            except Exception:
                gpu = False
        self.reader = easyocr.Reader(["ar", "en"], gpu=gpu, verbose=False)

    @staticmethod
    def _prep(crop: np.ndarray) -> np.ndarray:
        h = crop.shape[0]
        if h < 96:
            s = 96 / max(1, h)
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        return cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

    def read(self, crop: np.ndarray) -> tuple[str, float, dict, str]:
        """Returns (plate_text, confidence, details, raw_ocr_text)."""
        if crop is None or crop.size == 0:
            return "", 0.0, {}, ""
        res = self.reader.readtext(self._prep(crop), detail=1, paragraph=False)
        res = [r for r in res if r[2] > 0.1 and r[1].strip()]
        if not res:
            return "", 0.0, {}, ""
        tokens = [(float(np.mean([p[0] for p in box])), text) for box, text, _ in res]
        text, details = parse_tokens(tokens)
        # Confidence of the parts that make up the plate text
        kept = [c for (box, t, c) in res if any(ch.isdigit() for ch in normalize_chars(t))] or [c for *_, c in res]
        raw = " | ".join(t for _, t, _ in res)
        return text, float(np.mean(kept)), details, raw
