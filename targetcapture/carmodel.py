"""Car make / model recognition (e.g. "Toyota Corolla", "Nissan Altima").

Uses CLIP zero-shot classification: the vehicle crop is compared with the text
"a photo of a <make model>, a type of car." for every entry in car_models.txt and the
closest one wins. No training is needed, and adding a model is one line in that file.
Weights: open_clip ViT-B/32 trained on LAION-400M (about 600 MB, downloaded once from
the open_clip GitHub release). Treat answers as a strong hint, not proof: CLIP is good
at make and body shape, weaker at telling close generations or twin models apart.
"""
from __future__ import annotations

import logging
import os
import urllib.request

import cv2
import numpy as np

log = logging.getLogger(__name__)

WEIGHTS_URL = ("https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/"
               "vit_b_32-quickgelu-laion400m_e32-46683a32.pt")
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "targetcapture")
DEFAULT_LIST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "car_models.txt")


def load_names(path: str = DEFAULT_LIST) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]


def _weights() -> str:
    os.makedirs(CACHE, exist_ok=True)
    dst = os.path.join(CACHE, os.path.basename(WEIGHTS_URL))
    if not os.path.exists(dst):
        print(f"Downloading car model recognizer (~600 MB) to {dst} ...")
        tmp = dst + ".part"
        urllib.request.urlretrieve(WEIGHTS_URL, tmp)
        os.replace(tmp, dst)
    return dst


class CarModelClassifier:
    def __init__(self, names_path: str = DEFAULT_LIST, min_conf: float = 0.25):
        import open_clip
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained=_weights(), device=self.device)
        self.model.eval()
        tok = open_clip.get_tokenizer("ViT-B-32-quickgelu")
        self.names = load_names(names_path)
        # Catch-all classes so a bus or a blurry blob is not forced into a car model.
        self.n_models = len(self.names)
        self.names += ["bus", "lorry", "motorcycle", "blurry vehicle", "car"]
        templates = ["a photo of a {}, a type of car.", "a photo of a {}.", "a {} car on the street."]
        with torch.no_grad():
            feats = []
            for t in templates:
                e = self.model.encode_text(tok([t.format(n) for n in self.names]).to(self.device))
                feats.append(e / e.norm(dim=-1, keepdim=True))
            text = torch.stack(feats).mean(0)
            self.text = text / text.norm(dim=-1, keepdim=True)
        self.min_conf = min_conf

    def predict(self, crop: np.ndarray, top: int = 3) -> list[tuple[str, float]]:
        """Best guesses for a BGR vehicle crop, as (name, probability)."""
        from PIL import Image

        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 32:
            return []
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        with self.torch.no_grad():
            x = self.preprocess(img).unsqueeze(0).to(self.device)
            f = self.model.encode_image(x)
            f = f / f.norm(dim=-1, keepdim=True)
            p = (100.0 * f @ self.text.T).softmax(dim=-1)[0].cpu().numpy()
        idx = np.argsort(-p)[:top]
        return [(self.names[i] if i < self.n_models else None, float(p[i])) for i in idx]

    def best(self, crop: np.ndarray) -> tuple[str | None, float]:
        r = self.predict(crop, top=1)
        if not r or r[0][0] is None or r[0][1] < self.min_conf:
            return None, (r[0][1] if r else 0.0)
        return r[0]
