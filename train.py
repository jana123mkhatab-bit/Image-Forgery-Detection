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

from data_stream import ForgeryStream
from model import ForgeryNet


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    return 1 - ((2 * inter + eps) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps)).mean()


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

    dl = DataLoader(ForgeryStream(cfg, seed=step), batch_size=t["batch"], num_workers=t["workers"],
                    pin_memory=dev == "cuda")
    model.train()
    t0 = time.time()
    accum = t["accum"]
    for x, m, y in dl:
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
            print(f"step {step} loss {loss.item() * accum:.4f} ({time.time() - t0:.0f}s)")
        if step % t["save_every"] == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, ck)
        if step >= t["max_steps"]:
            break
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step}, ck)


if __name__ == "__main__":
    main()
