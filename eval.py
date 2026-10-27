"""Evaluate or try the trained forgery model.

  # 1) quick sanity check on freshly generated samples (NOT a generalization test:
  #    same source photos as training, only new random edits)
  python eval.py synthetic --ckpt baseline_final.pt --n 1000

  # 2) real forgeries made by people (CASIA, Columbia, ...): never seen in training
  python eval.py folder --ckpt baseline_final.pt --images Tp/ --masks gt/ [--authentic Au/]

  # 3) try any image(s): verdict + heatmap overlay
  python eval.py predict --ckpt baseline_final.pt --image photo.jpg
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import roc_auc_score

from model import ForgeryNet
from run_baselines import EXTS, find_mask, prf

CLASSES = ["authentic", "edited", "ai_generated"]
THRS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MAX_SIDE = 2048  # larger images are skipped: we never resize (it destroys the traces)


def load_model(ckpt, cfg, dev):
    m = ForgeryNet(cfg["model"]["encoder"], None, 3, cfg["model"]["use_srm"]).to(dev)
    s = torch.load(ckpt, map_location=dev)
    m.load_state_dict(s["model"] if "model" in s else s)
    return m.eval()


@torch.no_grad()
def infer(model, img_rgb, dev):
    """Full-resolution inference. Returns (mask prob HxW, class probs [3])."""
    h, w = img_rgb.shape[:2]
    ph, pw = (-h) % 32, (-w) % 32
    x = torch.from_numpy(img_rgb).permute(2, 0, 1).float()[None] / 255
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
        ml, cl = model(x.to(dev))
    mask = torch.sigmoid(ml.float())[0, 0, :h, :w].cpu().numpy()
    probs = torch.softmax(cl.float(), 1)[0].cpu().numpy()
    return mask, probs


def overlay(img_rgb, mask, probs, gt=None):
    heat = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    over = cv2.addWeighted(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), 0.55, heat, 0.45, 0)
    panels = [cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), over]
    if gt is not None:
        panels.append(cv2.cvtColor((gt * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    out = np.hstack(panels)
    txt = "  ".join(f"{c}:{p:.2f}" for c, p in zip(CLASSES, probs))
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, txt, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class MaskStats:
    """Pixel F1 / IoU over tampered images at several thresholds."""

    def __init__(self):
        self.f1 = {t: [] for t in THRS}
        self.iou = {t: [] for t in THRS}

    def add(self, mask, gt):
        for t in THRS:
            iou, f1 = prf(mask >= t, gt)
            self.iou[t].append(iou)
            self.f1[t].append(f1)

    def summary(self):
        if not self.f1[0.5]:
            return {}
        best = max(THRS, key=lambda t: np.mean(self.f1[t]))
        return {"n_images": len(self.f1[0.5]),
                "f1@0.5": float(np.mean(self.f1[0.5])), "iou@0.5": float(np.mean(self.iou[0.5])),
                "best_threshold": best,
                "f1@best": float(np.mean(self.f1[best])), "iou@best": float(np.mean(self.iou[best]))}


def to_uint8(x):
    return (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


# ----------------------------------------------------------------- synthetic
def run_synthetic(a, cfg, model, dev, out):
    from data_stream import ForgeryStream
    it = iter(ForgeryStream(cfg, seed=a.seed))
    conf = np.zeros((3, 3), int)
    ms = MaskStats()
    fp_area = {0: [], 2: []}
    for i in range(a.n):
        x, m, y = next(it)
        img, gt, y = to_uint8(x), m[0].numpy() > 0.5, int(y)
        mask, probs = infer(model, img, dev)
        conf[y, int(probs.argmax())] += 1
        if y == 1:
            ms.add(mask, gt)
        else:
            fp_area[y].append(float((mask >= 0.5).mean()))
        if i < a.overlays:
            cv2.imwrite(str(out / f"synthetic_{i}_true-{CLASSES[y]}.png"),
                        overlay(img, mask, probs, gt.astype(np.uint8)))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{a.n}")
    recall = {CLASSES[k]: float(conf[k, k] / max(conf[k].sum(), 1)) for k in range(3)}
    return {"mode": "synthetic (in-distribution sanity check, not generalization)",
            "samples": a.n, "accuracy": float(np.trace(conf) / conf.sum()), "per_class_recall": recall,
            "confusion_rows_true_cols_pred": conf.tolist(), "class_order": CLASSES,
            "mask_on_edited": ms.summary(),
            "false_positive_area_on_authentic": float(np.mean(fp_area[0])) if fp_area[0] else None,
            "false_positive_area_on_ai": float(np.mean(fp_area[2])) if fp_area[2] else None}


# -------------------------------------------------------------------- folder
def list_images(d, limit):
    fs = sorted(p for p in Path(d).iterdir() if p.suffix.lower() in EXTS)
    return fs[:limit] if limit else fs


def load_rgb(p):
    im = cv2.imread(str(p))
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def run_folder(a, cfg, model, dev, out):
    ms, skipped = MaskStats(), 0
    y_true, s_edit, s_mask, s_any = [], [], [], []
    n_over = 0
    for p in list_images(a.images, a.limit):
        mp = find_mask(a.masks, p.stem)
        img = load_rgb(p)
        if mp is None or img is None or max(img.shape[:2]) > MAX_SIDE:
            skipped += 1
            continue
        gt = cv2.imread(str(mp), 0)
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
        gt = ~gt if a.invert_mask else gt
        mask, probs = infer(model, img, dev)
        ms.add(mask, gt)
        y_true.append(1); s_edit.append(probs[1]); s_mask.append(float(mask.max())); s_any.append(1 - probs[0])
        if n_over < a.overlays:
            cv2.imwrite(str(out / f"folder_{p.stem}.png"), overlay(img, mask, probs, gt.astype(np.uint8)))
            n_over += 1
    fp = []
    if a.authentic:
        for p in list_images(a.authentic, a.limit):
            img = load_rgb(p)
            if img is None or max(img.shape[:2]) > MAX_SIDE:
                skipped += 1
                continue
            mask, probs = infer(model, img, dev)
            y_true.append(0); s_edit.append(probs[1]); s_mask.append(float(mask.max())); s_any.append(1 - probs[0])
            fp.append(float((mask >= 0.5).mean()))
    res = {"mode": "folder (real forgeries, full resolution, no resizing)",
           "mask_on_tampered": ms.summary(), "skipped": skipped}
    if a.authentic and len(set(y_true)) == 2:
        res["image_level_auc"] = {"p_edited": float(roc_auc_score(y_true, s_edit)),
                                  "max_mask_prob": float(roc_auc_score(y_true, s_mask)),
                                  "1-p_authentic": float(roc_auc_score(y_true, s_any))}
        res["false_positive_area_on_authentic"] = float(np.mean(fp))
    elif a.authentic:
        res["note"] = "no authentic images were scored; AUC not computed"
    return res


# ------------------------------------------------------------------- predict
def run_predict(a, cfg, model, dev, out):
    paths = list_images(a.image, 0) if Path(a.image).is_dir() else [Path(a.image)]
    res = {}
    for p in paths:
        img = load_rgb(p)
        if img is None:
            print("cannot read", p)
            continue
        mask, probs = infer(model, img, dev)
        verdict = CLASSES[int(probs.argmax())]
        cv2.imwrite(str(out / f"predict_{p.stem}.png"), overlay(img, mask, probs))
        res[p.name] = {"verdict": verdict, **{c: round(float(q), 3) for c, q in zip(CLASSES, probs)},
                       "pixels_flagged_%": round(100 * float((mask >= 0.5).mean()), 2)}
        print(p.name, res[p.name])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["synthetic", "folder", "predict"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--n", type=int, default=1000, help="synthetic: number of samples")
    ap.add_argument("--seed", type=int, default=12345, help="synthetic: differs from training seeds")
    ap.add_argument("--images")
    ap.add_argument("--masks")
    ap.add_argument("--authentic")
    ap.add_argument("--invert-mask", action="store_true", help="if masks are black on tampered pixels")
    ap.add_argument("--image", help="predict: an image file or a folder")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overlays", type=int, default=12, help="how many example images to save")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.ckpt, cfg, dev)
    if a.mode == "folder" and not (a.images and a.masks):
        ap.error("folder mode needs --images and --masks")
    if a.mode == "predict" and not a.image:
        ap.error("predict mode needs --image")
    res = {"synthetic": run_synthetic, "folder": run_folder, "predict": run_predict}[a.mode](a, cfg, model, dev, out)
    (out / f"{a.mode}_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("saved to", out)


if __name__ == "__main__":
    main()
