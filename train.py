"""Train the dual-head model on streamed data. Same command on laptop and Colab.

  python train.py --config configs/default.yaml

Resumes automatically from <out_dir>/last.pt (point out_dir at Drive on Colab).
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import roc_auc_score

from data_stream import ForgeryStream
from model import ForgeryNet
from sofake_stream import SoFakeStream


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)).mean()


def make_stream(cfg, seed):
    if cfg["data"].get("source") == "sofake":
        return SoFakeStream(cfg, cfg["data"]["sofake"].get("split", "train"), seed=seed, train=True)
    return ForgeryStream(cfg, seed=seed)


_VAL = {}


def val_set(cfg):
    """Fixed held-out samples, built once per run from the validation split (seed fixed), so every
    checkpoint is scored on the same data. Only used for sofake configs."""
    if "d" not in _VAL:
        n = cfg["train"].get("val_n", 450)
        ds = SoFakeStream(cfg, cfg["data"]["sofake"].get("val_split", "validation"), seed=777,
                          train=False, max_items=n)
        xs, ms, ys = [], [], []
        for x, m, y in ds:
            xs.append((x * 255).round().byte()); ms.append(m.bool()); ys.append(int(y))
        _VAL["d"] = (torch.stack(xs), torch.stack(ms), torch.tensor(ys))
        print(f"validation set: {len(ys)} samples, class counts {np.bincount(ys, minlength=3).tolist()}")
    return _VAL["d"]


@torch.no_grad()
def validate(model, cfg, dev):
    X, M, Y = val_set(cfg)
    model.eval()
    preds, probs, f1s = [], [], []
    for i in range(0, len(Y), 32):
        x = X[i:i + 32].float().div(255).to(dev)
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            ml, cl = model(x)
        p = torch.softmax(cl.float(), 1).cpu()
        probs.append(p); preds.append(p.argmax(1))
        pm = (torch.sigmoid(ml.float()).cpu() > 0.5)
        for j in range(len(p)):
            if Y[i + j] == 1:
                g = M[i + j]
                tp = (pm[j] & g).sum().item()
                f1s.append(2 * tp / max(pm[j].sum().item() + g.sum().item(), 1))
    model.train()
    preds, probs = torch.cat(preds).numpy(), torch.cat(probs).numpy()
    y = Y.numpy()
    f1 = []
    for k in range(3):
        tp = ((preds == k) & (y == k)).sum()
        pr, rc = tp / max((preds == k).sum(), 1), tp / max((y == k).sum(), 1)
        f1.append(2 * pr * rc / max(pr + rc, 1e-8))
    auc = roc_auc_score((y != 0).astype(int), 1 - probs[:, 0]) if len(set(y.tolist())) > 1 else float("nan")
    return {"acc": float((preds == y).mean()), "f1_authentic": float(f1[0]), "f1_edited": float(f1[1]),
            "f1_ai": float(f1[2]), "macro_f1": float(np.mean(f1)), "auc_real_vs_fake": float(auc),
            "mask_f1_edited": float(np.mean(f1s)) if f1s else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out_dir", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out = Path(a.out_dir or cfg["train"]["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = cfg["train"]
    model = ForgeryNet(cfg["model"]["encoder"], cfg["model"]["weights"], 3, cfg["model"]["use_srm"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    step = 0
    ck = out / "last.pt"
    if ck.exists():
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); step = s["step"]
        print(f"resumed from step {step}")

    dl = DataLoader(make_stream(cfg, step), batch_size=t["batch"], num_workers=t["workers"],
                    pin_memory=dev == "cuda")
    model.train()
    t0 = time.time()
    accum = t["accum"]
    use_val = cfg["data"].get("source") == "sofake" and t.get("val_every")
    best_path = out / "best.pt"
    best = -1.0
    if best_path.exists():
        best = torch.load(best_path, map_location="cpu").get("val_macro_f1", -1.0)
    for x, m, y in dl:
        if t.get("cosine"):
            frac = min(step / t["max_steps"], 1.0)
            for g in opt.param_groups:
                g["lr"] = float(t["lr"] * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * frac))))  # plain float: numpy scalars break torch.load on resume
        x, m, y = x.to(dev), m.to(dev), y.to(dev)
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            ml, cl = model(x)
            loss = (F.binary_cross_entropy_with_logits(ml, m) + dice_loss(ml, m)
                    + F.cross_entropy(cl, y)) / accum
        scaler.scale(loss).backward()
        step += 1
        if step % accum == 0:
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if step % 50 == 0:
            try:
                import psutil
                ram = f" ram {psutil.virtual_memory().percent:.0f}%"
            except Exception:
                ram = ""
            print(f"step {step} loss {loss.item() * accum:.4f} ({time.time() - t0:.0f}s){ram}", flush=True)
        if step % t["save_every"] == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, ck)
        if use_val and step % t["val_every"] == 0:
            r = validate(model, cfg, dev)
            print(f"VAL step {step} " + " ".join(f"{k} {v:.3f}" for k, v in r.items()))
            if r["macro_f1"] > best:
                best = r["macro_f1"]
                torch.save({"model": model.state_dict(), "step": step, "val_macro_f1": best}, best_path)
                print(f"  new best (macro_f1 {best:.3f}) -> {best_path.name}")
        if step >= t["max_steps"]:
            break
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, ck)


if __name__ == "__main__":
    main()