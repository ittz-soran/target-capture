"""Train the Iraqi / Kurdistan Region plate reader (targetcapture/plateocr.py) on synthetic plates.

    python tools/train_plate_ocr.py --minutes 50 [--real teach_export_dir] [--workers 2 --threads 2]

Plates are rendered and augmented on the fly (targetcapture/synth.py). Each crop is fed in the
form that fits its layout (one-line or two-line); now and then it is fed in the other form with
an empty label, so the model learns to return nothing for the wrong form. `--real DIR` mixes in
labelled crops from a Teach-tab export (images/*.png + labels.csv), oversampled.

Each epoch prints exact-match accuracy on a fixed held-out synthetic set and on the real sample
photos (cropped with fast-alpr's plate detector); the best checkpoint (by synthetic accuracy) is
exported to ONNX.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
import time
from multiprocessing import Pool

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from targetcapture import plateocr as P  # noqa: E402
from targetcapture import synth  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "samples", "plates")
NCLS = 1 + len(P.CHARSET)
CHAR_IDX = {c: i + 1 for i, c in enumerate(P.CHARSET)}


def _cbr(i: int, o: int, k=3, p=1, s=1) -> list[nn.Module]:
    return [nn.Conv2d(i, o, k, stride=s, padding=p, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]


class CRNN(nn.Module):
    """1x32x256 -> 64 time steps x NCLS logits."""

    def __init__(self, ch=(24, 48, 96, 128), hidden=96):
        super().__init__()
        a, b, c, d = ch
        self.cnn = nn.Sequential(*_cbr(1, a, s=2), *_cbr(a, b), nn.MaxPool2d(2),
                                 *_cbr(b, c), *_cbr(c, c), nn.MaxPool2d((2, 1)),
                                 *_cbr(c, d), nn.MaxPool2d((2, 1)), *_cbr(d, d, (2, 1), 0))
        self.rnn = nn.LSTM(d, hidden, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden, NCLS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.cnn(x).squeeze(2).permute(0, 2, 1)
        return self.fc(self.rnn(f)[0])


class Probs(nn.Module):
    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x).softmax(-1)


# ---------------------------------------------------------------- data

def load_real(folder: str) -> list[tuple[np.ndarray, str, str]]:
    """Teach-tab export -> [(crop, label, layout)], skipping labels with Arabic letters."""
    out = []
    with open(os.path.join(folder, "labels.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = P.normalize_label(row.get("plate_text") or "")
            img = cv2.imread(os.path.join(folder, row["image_path"]))
            if not label or img is None:
                continue
            out.append((img, label, guess_layout(img, label)))
    return out


def guess_layout(crop: np.ndarray, label: str) -> str:
    if re.fullmatch(r"\d{9}|\d{3,7}", label):
        return "two-line"
    g = P.trim(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    return "one-line" if g.shape[1] / g.shape[0] > 2.6 else "two-line"


def _jitter(crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = crop.shape[:2]
    m = rng.uniform(-0.05, 0.05, 4)
    x0, y0 = int(max(0, m[0] * w)), int(max(0, m[1] * h))
    x1, y1 = int(min(w, w - m[2] * w)), int(min(h, h - m[3] * h))
    out = crop[y0:y1, x0:x1] if x1 - x0 > 8 and y1 - y0 > 8 else crop
    rot = cv2.getRotationMatrix2D((out.shape[1] / 2, out.shape[0] / 2), rng.uniform(-4, 4), 1.0)
    out = cv2.warpAffine(out, rot, (out.shape[1], out.shape[0]), borderMode=cv2.BORDER_REPLICATE)
    out = np.clip(out.astype(np.float32) * rng.uniform(0.7, 1.2) + rng.uniform(-30, 30), 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.5, 2.0))
    return out


def make_input(crop: np.ndarray, label: str, layout: str, rng: np.random.Generator,
               neg_p: float) -> tuple[np.ndarray, str]:
    wrong = rng.random() < neg_p
    if (layout == "one-line") != wrong:
        return P.one_line(crop), "" if wrong else label
    return P.two_line(crop), "" if wrong else label


def synth_sample(rng: np.random.Generator) -> tuple[np.ndarray, str, str, str]:
    """(augmented crop, label, layout, style)"""
    img, label, style = synth.render_plate_style(rng)
    layout = "two-line" if style in synth.TWO_LINE_STYLES else "one-line"
    return synth.augment(img, rng), label, layout, style


class Stream(IterableDataset):
    def __init__(self, seed: int, real=None, real_p=0.0, neg_p=0.1):
        self.seed, self.real, self.real_p, self.neg_p = seed, real or [], real_p, neg_p

    def __iter__(self):
        wi = get_worker_info()
        rng = np.random.default_rng([self.seed, wi.id if wi else 0, int(time.time())])
        cv2.setNumThreads(1)
        while True:
            try:
                if self.real and rng.random() < self.real_p:
                    crop, label, layout = self.real[int(rng.integers(len(self.real)))]
                    crop = _jitter(crop, rng)
                else:
                    crop, label, layout, _ = synth_sample(rng)
                x, y = make_input(crop, label, layout, rng, self.neg_p)
            except Exception as e:  # a rare degenerate render must not stop training
                print("sample error:", e, file=sys.stderr)
                continue
            yield torch.from_numpy(x), y


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    targets = torch.tensor([CHAR_IDX[c] for s in labels for c in s], dtype=torch.long)
    lengths = torch.tensor([len(s) for s in labels], dtype=torch.long)
    return xs, targets, lengths


def _heldout_one(seed: int):
    rng = np.random.default_rng([987654, seed])
    crop, label, layout, style = synth_sample(rng)
    return np.stack(P.prepare(crop)), label, style


def heldout_set(n: int, procs: int) -> tuple[np.ndarray, list[str], list[str]]:
    with Pool(procs) as pool:
        items = pool.map(_heldout_one, range(n), chunksize=32)
    return np.stack([i[0] for i in items]), [i[1] for i in items], [i[2] for i in items]


def real_samples(folder: str) -> list[dict]:
    """Labelled sample photos, cropped by fast-alpr's detector (10% margin; whole photo if none)."""
    from fast_alpr import ALPR
    alpr = ALPR(detector_model="yolo-v9-t-384-license-plate-end2end", ocr_model="cct-s-v2-global-model")
    out = []
    with open(os.path.join(folder, "labels.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            label = P.normalize_label(row.get("plate") or "")
            img = cv2.imread(os.path.join(folder, row["file"]))
            if not label or img is None:
                continue
            dets = alpr.detector.predict(img)
            crop = img
            if dets:
                b = max(dets, key=lambda d: d.confidence).bounding_box
                mx, my = int((b.x2 - b.x1) * 0.1), int((b.y2 - b.y1) * 0.1)
                crop = img[max(0, b.y1 - my):b.y2 + my, max(0, b.x1 - mx):b.x2 + mx]
            cct = alpr.ocr.predict(crop)
            cct = (cct.text or "").replace("_", "") if cct else ""
            out.append(dict(file=row["file"], label=label, crop=crop, x=np.stack(P.prepare(crop)),
                            cct=cct, detected=bool(dets)))
    return out


# ---------------------------------------------------------------- train / eval

@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, bs: int = 256) -> list[tuple[str, float, str]]:
    """x: (N, 2, 1, 32, 256) candidates -> best (text, conf, layout) per item."""
    model.eval()
    n = x.shape[0]
    flat = torch.from_numpy(x.reshape(-1, 1, P.H, P.W))
    probs = torch.cat([model(flat[i:i + bs]).softmax(-1) for i in range(0, len(flat), bs)]).numpy()
    probs = probs.reshape(n, len(P.LAYOUTS), *probs.shape[1:])
    out = []
    for p in probs:
        out.append(P.choose([(*P.ctc_decode(q), lay) for q, lay in zip(p, P.LAYOUTS)]))
    model.train()
    return out


def export_onnx(model: nn.Module, path: str) -> None:
    model.eval()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.onnx.export(Probs(model), torch.zeros(2, 1, P.H, P.W), path, input_names=["x"], output_names=["probs"],
                      dynamic_axes={"x": {0: "n"}, "probs": {0: "n"}}, opset_version=17, dynamo=False)
    model.train()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--minutes", type=float, default=50, help="training time budget (cosine LR over it)")
    ap.add_argument("--steps-per-epoch", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--heldout", type=int, default=1200)
    ap.add_argument("--real", help="Teach-tab export folder (images/ + labels.csv) to mix in")
    ap.add_argument("--real-p", type=float, default=0.15, help="share of batches drawn from --real")
    ap.add_argument("--neg-p", type=float, default=0.1, help="share fed in the wrong layout (empty label)")
    ap.add_argument("--samples", default=SAMPLES, help="real sample photos + labels.csv for evaluation")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "runs", "plate_ocr_best.pt"))
    ap.add_argument("--resume", help="checkpoint to start from")
    ap.add_argument("--onnx", default=P.DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    t0 = time.time()
    hx, hy, hs = heldout_set(args.heldout, max(1, args.workers + args.threads))
    print(f"held-out synthetic set: {len(hy)} plates ({time.time() - t0:.0f}s)", flush=True)
    reals = []
    if args.samples and os.path.exists(os.path.join(args.samples, "labels.csv")):
        reals = real_samples(args.samples)
        for r in reals:
            print(f"  real {r['file']}: label {r['label']}  cct said {r['cct'] or '-'}"
                  f"{'' if r['detected'] else '  (no detection, whole photo)'}", flush=True)
    train_real = load_real(args.real) if args.real else []
    if args.real:
        print(f"mixing in {len(train_real)} real crops from {args.real}", flush=True)

    model = CRNN()
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location="cpu")["model"])
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(Stream(args.seed, train_real, args.real_p if train_real else 0.0, args.neg_p),
                        batch_size=args.batch, num_workers=args.workers, collate_fn=collate,
                        persistent_workers=args.workers > 0, prefetch_factor=4 if args.workers else None)
    budget = args.minutes * 60
    start, step, epoch, best, run_loss = time.time(), 0, 0, -1.0, 0.0
    it = iter(loader)
    os.makedirs(os.path.dirname(args.ckpt) or ".", exist_ok=True)
    while True:
        frac = (time.time() - start) / budget
        lr = args.lr * min(1.0, step / 300) * 0.5 * (1 + math.cos(math.pi * min(frac, 1.0)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, targets, lengths = next(it)
        logp = model(x).log_softmax(-1).permute(1, 0, 2)
        loss = F.ctc_loss(logp, targets, torch.full((x.shape[0],), logp.shape[0], dtype=torch.long), lengths,
                          zero_infinity=True)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        run_loss += loss.item()
        step += 1
        if step % args.steps_per_epoch and frac < 1.0:
            continue
        epoch += 1
        preds = predict(model, hx)
        ok = [p[0] == y for p, y in zip(preds, hy)]
        acc = float(np.mean(ok))
        per_style = " ".join(f"{s}:{np.mean([o for o, t in zip(ok, hs) if t == s]):.3f}" for s in synth.STYLES)
        msg = (f"epoch {epoch} step {step} {(time.time() - start) / 60:.1f}min lr {lr:.2e} "
               f"loss {run_loss / args.steps_per_epoch:.3f} synth {acc:.3f} ({per_style})")
        run_loss = 0.0
        if reals:
            rp = predict(model, np.stack([r["x"] for r in reals]))
            rok = sum(p[0] == r["label"] for p, r in zip(rp, reals))
            msg += f" | real {rok}/{len(reals)}: " + " ".join(
                f"{r['file'][:2]}={p[0] or '-'}{'' if p[0] == r['label'] else '✗'}" for p, r in zip(rp, reals))
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "epoch": epoch, "synth_acc": acc}, args.ckpt)
            export_onnx(model, args.onnx)
            msg += " [saved]"
        print(msg, flush=True)
        if frac >= 1.0:
            break
    print(f"done in {(time.time() - start) / 60:.1f} min, best synthetic accuracy {best:.3f}; "
          f"onnx {args.onnx} ({os.path.getsize(args.onnx) / 1e6:.2f} MB)", flush=True)


if __name__ == "__main__":
    main()
