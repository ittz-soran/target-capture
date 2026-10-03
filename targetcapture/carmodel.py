"""Car make / model recognition (e.g. "Toyota Corolla", "Nissan Altima").

Uses CLIP zero-shot classification: the vehicle crop is compared with the text
"a photo of a <make model>, a type of car." for every entry in car_models.txt and the
closest one wins. No training is needed, and adding a model is one line in that file.
Weights: open_clip ViT-B/32 or ViT-L/14 trained on LAION-400M (see MODELS; downloaded once
from the open_clip GitHub release). Treat answers as a strong hint, not proof: CLIP is good
at make and body shape, weaker at telling close generations or twin models apart.
"""
from __future__ import annotations

import logging
import os
import urllib.request

import cv2
import numpy as np

log = logging.getLogger(__name__)

RELEASE = "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/"
# "fast": ViT-B/32 (600 MB). "accurate": ViT-L/14 (1.7 GB), clearly better at telling models apart
# (published zero-shot accuracy on the Stanford Cars set: about 79% vs 90%) but about 8x slower,
# so meant for a GPU. Both trained on LAION-400M.
MODELS = {
    "fast": ("ViT-B-32-quickgelu", "vit_b_32-quickgelu-laion400m_e32-46683a32.pt", "600 MB"),
    "accurate": ("ViT-L-14", "vit_l_14-laion400m_e32-3d133497.pt", "1.7 GB"),
}
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "targetcapture")
DEFAULT_LIST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "car_models.txt")


def load_names(path: str = DEFAULT_LIST) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]


def resolve_size(size: str = "auto") -> str:
    """'auto': the accurate model on an NVIDIA GPU, the fast one on CPU."""
    if size in MODELS:
        return size
    from .gpu import torch_cuda

    return "accurate" if torch_cuda() else "fast"


def _weights(size: str) -> str:
    _, name, mb = MODELS[size]
    os.makedirs(CACHE, exist_ok=True)
    dst = os.path.join(CACHE, name)
    if not os.path.exists(dst):
        print(f"Downloading car model recognizer (~{mb}, once) to {dst} ...")
        tmp = dst + ".part"
        urllib.request.urlretrieve(RELEASE + name, tmp)
        os.replace(tmp, dst)
    return dst


class CarModelClassifier:
    def __init__(self, names_path: str = DEFAULT_LIST, min_conf: float = 0.25, size: str = "auto"):
        import open_clip
        import torch

        self.torch = torch
        self.size = resolve_size(size)
        arch = MODELS[self.size][0]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=_weights(self.size), device=self.device)
        self.model.eval()
        tok = open_clip.get_tokenizer(arch)
        self.names = load_names(names_path)
        # Catch-all classes so a bus or a blurry blob is not forced into a car model.
        self.n_models = len(self.names)
        self.names += ["bus", "lorry", "motorcycle", "blurry vehicle", "car"]
        templates = ["a photo of a {}, a type of car.", "a photo of a {}.", "a {} car on the street.",
                     "a CCTV photo of a {}.", "a photo of my {}.", "the front of a {}.", "the back of a {}."]
        with torch.no_grad():
            feats = []
            for t in templates:
                e = self.model.encode_text(tok([t.format(n) for n in self.names]).to(self.device))
                feats.append(e / e.norm(dim=-1, keepdim=True))
            text = torch.stack(feats).mean(0)
            self.text = text / text.norm(dim=-1, keepdim=True)
        self.min_conf = min_conf

    def probs(self, crop: np.ndarray) -> np.ndarray | None:
        """Probability of every name for a BGR vehicle crop (the crop and its mirror image are
        averaged: steadier than one look). None when the crop is too small."""
        from PIL import Image

        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 32:
            return None
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        with self.torch.no_grad(), self.torch.autocast("cuda", enabled=self.device == "cuda"):
            # half precision on an NVIDIA card: about twice as fast, same answers
            x = self.preprocess(img).unsqueeze(0)
            x = self.torch.cat([x, x.flip(-1)]).to(self.device)
            f = self.model.encode_image(x).float()
            f = f / f.norm(dim=-1, keepdim=True)
            return (100.0 * f @ self.text.T).softmax(dim=-1).mean(0).cpu().numpy()

    def name_of(self, p: np.ndarray) -> tuple[str | None, float]:
        """Best car model in a probability vector, or None when unsure or not a car."""
        i = int(np.argmax(p))
        if i >= self.n_models or p[i] < self.min_conf:
            return None, float(p[i])
        return self.names[i], float(p[i])

    def predict(self, crop: np.ndarray, top: int = 3) -> list[tuple[str, float]]:
        """Best guesses for a BGR vehicle crop, as (name, probability)."""
        p = self.probs(crop)
        if p is None:
            return []
        idx = np.argsort(-p)[:top]
        return [(self.names[i] if i < self.n_models else None, float(p[i])) for i in idx]

    def best(self, crop: np.ndarray) -> tuple[str | None, float]:
        p = self.probs(crop)
        return (None, 0.0) if p is None else self.name_of(p)
