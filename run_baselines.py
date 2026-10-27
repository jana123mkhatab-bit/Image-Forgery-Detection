"""Run all classical baselines over a folder of images (+ optional GT masks)
and report pixel-level IoU / F1 and image-level AUC.

Usage:
  python run_baselines.py --images data/casia/Tp --masks data/casia/gt \
         --authentic data/casia/Au --out results/casia --limit 200

Masks are matched by file stem; a trailing "_gt" or "_mask" on the mask name
is ignored (CASIA ground truth uses "_gt"). Authentic folder is optional and
only used for image-level AUC.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

from baselines import METHODS, copy_move

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def load_rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def find_mask(masks_dir, stem):
    for p in Path(masks_dir).iterdir():
        s = p.stem
        for suf in ("_gt", "_mask"):
            if s.endswith(suf):
                s = s[: -len(suf)]
        if s == stem and p.suffix.lower() in EXTS:
            return p
    return None


def prf(pred_bin, gt_bin):
    tp = np.logical_and(pred_bin, gt_bin).sum()
    fp = np.logical_and(pred_bin, ~gt_bin).sum()
    fn = np.logical_and(~pred_bin, gt_bin).sum()
    iou = tp / (tp + fp + fn + 1e-8)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    return iou, f1


def best_thr_scores(pred, gt, thrs=np.linspace(0.05, 0.95, 19)):
    scores = [prf(pred >= t, gt) for t in thrs]
    return max(scores, key=lambda s: s[1])  # best F1 (and its IoU)


def heatmaps(img):
    out = {name: fn(img) for name, fn in METHODS.items()}
    cm_mask, n = copy_move(img)
    out["copy_move"] = cm_mask
    return out, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--authentic", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--save-overlays", type=int, default=10)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    imgs = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in EXTS)
    if a.limit:
        imgs = imgs[: a.limit]

    rows, agg = [], {}
    img_scores = {k: ([], []) for k in list(METHODS) + ["copy_move"]}  # (labels, scores)
    for i, p in enumerate(imgs):
        mp = find_mask(a.masks, p.stem)
        if mp is None:
            continue
        img = load_rgb(p)
        gt = cv2.imread(str(mp), 0)
        gt = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
        hm, n_cm = heatmaps(img)
        row = {"image": p.name, "copy_move_inliers": n_cm}
        for name, h in hm.items():
            iou05, f105 = prf(h >= 0.5, gt)
            f1b_iou, f1b = best_thr_scores(h, gt)
            row.update({f"{name}_iou@0.5": iou05, f"{name}_f1@0.5": f105,
                        f"{name}_iou@best": f1b_iou, f"{name}_f1@best": f1b})
            for k, v in row.items():
                if k.startswith(name):
                    agg.setdefault(k, []).append(v)
            img_scores[name][0].append(1)
            img_scores[name][1].append(float(h.max() if name != "copy_move" else n_cm))
        rows.append(row)
        if i < a.save_overlays:
            panels = [img]
            for h in hm.values():
                heat = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)
                panels.append(cv2.cvtColor(cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), 0.5, heat, 0.5, 0), cv2.COLOR_BGR2RGB))
            panels.append(np.repeat((gt * 255).astype(np.uint8)[..., None], 3, 2))
            cv2.imwrite(str(out / f"overlay_{p.stem}.png"), cv2.cvtColor(np.hstack(panels), cv2.COLOR_RGB2BGR))
        print(f"[{i + 1}/{len(imgs)}] {p.name}")

    # image-level AUC (needs authentic images too)
    if a.authentic:
        au = sorted(p for p in Path(a.authentic).iterdir() if p.suffix.lower() in EXTS)
        if a.limit:
            au = au[: a.limit]
        for p in au:
            img = load_rgb(p)
            hm, n_cm = heatmaps(img)
            for name, h in hm.items():
                img_scores[name][0].append(0)
                img_scores[name][1].append(float(h.max() if name != "copy_move" else n_cm))

    with open(out / "per_image.csv", "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    print("\n=== Summary (mean over images) ===")
    names = list(METHODS) + ["copy_move"]
    print(f"{'method':<12}{'IoU@0.5':>9}{'F1@0.5':>9}{'IoU@best':>10}{'F1@best':>9}{'AUC':>7}")
    for n in names:
        g = lambda s: np.mean(agg.get(f"{n}_{s}", [np.nan]))
        y, s = img_scores[n]
        auc = roc_auc_score(y, s) if len(set(y)) == 2 else float("nan")
        print(f"{n:<12}{g('iou@0.5'):>9.3f}{g('f1@0.5'):>9.3f}{g('iou@best'):>10.3f}{g('f1@best'):>9.3f}{auc:>7.3f}")


if __name__ == "__main__":
    main()
