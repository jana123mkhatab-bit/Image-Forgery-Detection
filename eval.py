"""Evaluate or try the trained forgery model.

  # 1) quick sanity check on freshly generated samples (NOT a generalization test:
  #    same source photos as training, only new random edits)
  python eval.py synthetic --ckpt baseline_final.pt --n 1000

  # 2) real forgeries made by people (CASIA, Columbia, ...): never seen in training
  python eval.py folder --ckpt baseline_final.pt --images Tp/ --masks gt/ [--authentic Au/]

  # 3) try any image(s): verdict + heatmap overlay
  python eval.py predict --ckpt baseline_final.pt --image photo.jpg

  # 4) real-world test set streamed from the Hugging Face Hub (no download, no manual setup)
  python eval.py hub --ckpt baseline_final.pt --preset sofake     # real / tampered / fully AI
  python eval.py hub --ckpt baseline_final.pt --preset khang      # real vs AI-edited, with masks

  # add --tile to any mode to score 256x256 tiles (the training scale) instead of the whole image
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


@torch.no_grad()
def infer_tiled(model, img_rgb, dev, tile=256, stride=192, chunk=16):
    """Diagnostic: run the model on overlapping tile x tile windows (the scale it was trained at)
    and stitch. Mask = mean of overlapping tile probabilities. Class probs = max over tiles of
    P(edited)/P(ai) with authentic = the remainder, i.e. the most suspicious tile decides."""
    h, w = img_rgb.shape[:2]
    if h < tile or w < tile:
        return infer_full(model, img_rgb, dev)
    ys = list(range(0, h - tile + 1, stride))
    xs = list(range(0, w - tile + 1, stride))
    if ys[-1] != h - tile:
        ys.append(h - tile)
    if xs[-1] != w - tile:
        xs.append(w - tile)
    pos = [(y, x) for y in ys for x in xs]
    acc, cnt = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    cls = []
    for i in range(0, len(pos), chunk):
        part = pos[i:i + chunk]
        x = torch.stack([torch.from_numpy(img_rgb[y:y + tile, xx:xx + tile]).permute(2, 0, 1).float() / 255
                         for y, xx in part])
        with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            ml, cl = model(x.to(dev))
        pm = torch.sigmoid(ml.float())[:, 0].cpu().numpy()
        cls.append(torch.softmax(cl.float(), 1).cpu().numpy())
        for (y, xx), m in zip(part, pm):
            acc[y:y + tile, xx:xx + tile] += m
            cnt[y:y + tile, xx:xx + tile] += 1
    cls = np.concatenate(cls)
    p_edit, p_ai = float(cls[:, 1].max()), float(cls[:, 2].max())
    probs = np.array([max(1 - p_edit - p_ai, 0.0), p_edit, p_ai], dtype=np.float32)
    probs /= probs.sum()
    return acc / np.maximum(cnt, 1), probs


infer_full = infer  # whole-image inference; --tile swaps `infer` for infer_tiled


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


def cls_report(conf):
    """Per-class precision / recall / F1 and macro F1 from a confusion matrix (rows = true class)."""
    out, f1s = {}, []
    for k, name in enumerate(CLASSES):
        tp = conf[k, k]
        prec = tp / max(conf[:, k].sum(), 1)
        rec = tp / max(conf[k].sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        out[name] = {"precision": float(prec), "recall": float(rec), "f1": float(f1),
                     "support": int(conf[k].sum())}
        f1s.append(f1)
    out["macro_f1"] = float(np.mean(f1s))
    return out


def binary_report(y, scores, fixed=()):
    """Authentic (0) vs edited (1) from several image-level scores. For each score: AUC, plus
    F1 / precision / recall at the threshold that maximises F1 (picked on this same data, so
    slightly optimistic). Scores named in `fixed` are probabilities: F1 at a plain 0.5 too."""
    y = np.asarray(y)
    out = {}
    if len(set(y.tolist())) < 2:
        return out

    def at(s, t):
        pred = s >= t
        tp, fp, fn = (pred & (y == 1)).sum(), (pred & (y == 0)).sum(), (~pred & (y == 1)).sum()
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        return float(2 * p * r / max(p + r, 1e-8)), float(p), float(r)

    for name, s in scores.items():
        s = np.asarray(s, dtype=float)
        ts = np.unique(np.quantile(s, np.linspace(0, 1, 101)))
        t_best = max(ts, key=lambda t: at(s, t)[0])
        f1, p, r = at(s, t_best)
        out[name] = {"auc": float(roc_auc_score(y, s)), "best_f1": f1, "precision": p, "recall": r,
                     "threshold": float(t_best)}
        if name in fixed:
            out[name]["f1@0.5"] = at(s, 0.5)[0]
    return out


def to_uint8(x):
    return (x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)


# ----------------------------------------------------------------- synthetic
def run_synthetic(a, cfg, model, dev, out):
    from data_stream import ForgeryStream
    it = iter(ForgeryStream(cfg, seed=a.seed))
    conf = np.zeros((3, 3), int)
    ms = MaskStats()
    fp_area = {0: [], 2: []}
    ys, s_cls, s_max, s_area = [], [], [], []  # authentic-vs-edited image-level scores
    for i in range(a.n):
        x, m, y = next(it)
        img, gt, y = to_uint8(x), m[0].numpy() > 0.5, int(y)
        mask, probs = infer(model, img, dev)
        conf[y, int(probs.argmax())] += 1
        if y in (0, 1):
            ys.append(y)
            s_cls.append(float(probs[1] / (probs[0] + probs[1] + 1e-8)))
            s_max.append(float(mask.max()))
            s_area.append(float((mask >= 0.5).mean()))
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
            "classification_f1": cls_report(conf),
            "authentic_vs_edited": binary_report(
                ys, {"class_head_p_edited": s_cls, "mask_max_prob": s_max, "mask_area_frac": s_area},
                fixed=("class_head_p_edited",)),
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
    y_true, s_edit, s_mask, s_any, s_area = [], [], [], [], []
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
        s_area.append(float((mask >= 0.5).mean()))
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
            s_area.append(float((mask >= 0.5).mean()))
            fp.append(s_area[-1])
    res = {"mode": "folder (real forgeries, full resolution, no resizing)",
           "mask_on_tampered": ms.summary(), "skipped": skipped}
    if a.authentic and len(set(y_true)) == 2:
        res["image_level"] = binary_report(
            y_true, {"class_head_p_edited": s_edit, "class_head_1-p_authentic": s_any,
                     "mask_max_prob": s_mask, "mask_area_frac": s_area},
            fixed=("class_head_p_edited", "class_head_1-p_authentic"))
        res["false_positive_area_on_authentic"] = float(np.mean(fp))
    elif a.authentic:
        res["note"] = "no authentic images were scored; image-level metrics not computed"
    return res


# ----------------------------------------------------------------------- hub
# Test sets that load straight from the Hugging Face Hub. Column layouts were read off the
# dataset pages, not verified by running them, so run_hub prints the real columns and fails
# loudly if the label column is missing. Override with --hf-id / --label-col / --mask-col.
HUB_PRESETS = {
    "sofake": dict(id="saberzl/So-Fake-OOD", split="test_image"),   # REAL / TAMPERED / FULL_SYNTHETIC
    "khang": dict(id="KhangTruong/Merged", split="cross_test"),     # real / fake (AI-edited), with masks
}


def norm_label(v, names=None):
    """Map a dataset label to our classes: 0 authentic, 1 edited, 2 fully AI-generated (None = unknown)."""
    if isinstance(v, (int, np.integer)):
        if names:
            v = names[int(v)]
        else:
            return int(v) if int(v) in (0, 1, 2) else None
    s = str(v).upper()
    if "SYNTH" in s:
        return 2
    if "TAMPER" in s or "FAKE" in s or "EDIT" in s:
        return 1
    if "REAL" in s or "AUTHENTIC" in s:
        return 0
    return None


def run_hub(a, cfg, model, dev, out):
    from datasets import load_dataset
    spec = dict(HUB_PRESETS.get(a.preset, {}))
    spec["id"] = a.hf_id or spec.get("id")
    spec["split"] = a.hf_split or spec.get("split", "test")
    if not spec["id"]:
        raise SystemExit("hub mode needs --preset or --hf-id")
    ds = load_dataset(spec["id"], split=spec["split"], streaming=True,
                      **({"name": a.hf_config} if a.hf_config else {}))
    feats = getattr(ds, "features", None)
    names = getattr(feats.get(a.label_col), "names", None) if feats else None
    it = iter(ds.shuffle(seed=a.seed, buffer_size=500))
    counts, scanned, skipped = {0: 0, 1: 0, 2: 0}, 0, {"unknown_label": 0, "too_large": 0, "no_mask": 0}
    conf = np.zeros((3, 3), int)
    ms = MaskStats()
    y_bin, s_any, s_max, s_area = [], [], [], []
    fp_area = {0: [], 2: []}
    saved = {0: 0, 1: 0, 2: 0}
    for rec in it:
        scanned += 1
        if scanned == 1:
            print("columns:", list(rec.keys()))
            for col in (a.image_col, a.label_col):
                if col not in rec:
                    raise SystemExit(f"column '{col}' not found; use --image-col / --label-col with one of {list(rec.keys())}")
        if sum(counts.values()) >= a.n or scanned > a.max_scan:
            break
        cls = norm_label(rec[a.label_col], names)
        if cls is None:
            skipped["unknown_label"] += 1
            continue
        if counts[cls] >= a.per_class:
            continue
        im = rec[a.image_col]
        img = np.array(im.convert("RGB"))
        if max(img.shape[:2]) > MAX_SIDE:
            skipped["too_large"] += 1
            continue
        gt = None
        if cls == 1:
            m = rec.get(a.mask_col)
            if m is None:
                skipped["no_mask"] += 1
            else:
                m = np.array(m.convert("L"))
                gt = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
                if not gt.any():
                    gt = None
                    skipped["no_mask"] += 1
        mask, probs = infer(model, img, dev)
        counts[cls] += 1
        conf[cls, int(probs.argmax())] += 1
        y_bin.append(int(cls != 0))
        s_any.append(float(1 - probs[0]))
        s_max.append(float(mask.max()))
        s_area.append(float((mask >= 0.5).mean()))
        if gt is not None:
            ms.add(mask, gt)
        if cls in fp_area:
            fp_area[cls].append(s_area[-1])
        if saved[cls] < a.overlays:
            cv2.imwrite(str(out / f"hub_{CLASSES[cls]}_{saved[cls]}.png"),
                        overlay(img, mask, probs, None if gt is None else gt.astype(np.uint8)))
            saved[cls] += 1
        if sum(counts.values()) % 50 == 0:
            print(f"{sum(counts.values())}/{a.n}  (scanned {scanned})")
    res = {"mode": f"hub: {spec['id']} [{spec['split']}] (never seen in training)",
           "samples_per_class": {CLASSES[k]: v for k, v in counts.items()}, "scanned": scanned,
           "skipped": skipped, "label_names_from_dataset": names,
           "confusion_rows_true_cols_pred": conf.tolist(), "class_order": CLASSES}
    if all(counts.values()):
        res["classification_f1"] = cls_report(conf)
        res["accuracy"] = float(np.trace(conf) / conf.sum())
    res["flagged_as_not_authentic_by_true_class"] = {
        CLASSES[k]: float(1 - conf[k, 0] / conf[k].sum()) for k in range(3) if conf[k].sum()}
    res["real_vs_not_real"] = binary_report(
        y_bin, {"class_head_1-p_authentic": s_any, "mask_max_prob": s_max, "mask_area_frac": s_area},
        fixed=("class_head_1-p_authentic",))
    res["mask_on_edited"] = ms.summary()
    res["false_positive_area_on_authentic"] = float(np.mean(fp_area[0])) if fp_area[0] else None
    if counts[1] and not ms.f1[0.5]:
        res["note"] = "edited samples had no usable mask column, so pixel F1/IoU could not be computed"
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
    ap.add_argument("mode", choices=["synthetic", "folder", "predict", "hub"])
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
    ap.add_argument("--preset", choices=sorted(HUB_PRESETS), help="hub: a ready-made Hugging Face test set")
    ap.add_argument("--hf-id", help="hub: any dataset id, instead of a preset")
    ap.add_argument("--hf-split")
    ap.add_argument("--hf-config")
    ap.add_argument("--image-col", default="image")
    ap.add_argument("--mask-col", default="mask")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--tile", action="store_true",
                    help="diagnostic: run on 256x256 tiles (training scale) instead of the whole image")
    ap.add_argument("--per-class", type=int, default=200, help="hub: max samples per class")
    ap.add_argument("--max-scan", type=int, default=6000, help="hub: stop after looking at this many records")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.ckpt, cfg, dev)
    if a.tile:
        globals()["infer"] = infer_tiled
    if a.mode == "folder" and not (a.images and a.masks):
        ap.error("folder mode needs --images and --masks")
    if a.mode == "predict" and not a.image:
        ap.error("predict mode needs --image")
    res = {"synthetic": run_synthetic, "folder": run_folder, "predict": run_predict,
           "hub": run_hub}[a.mode](a, cfg, model, dev, out)
    (out / f"{a.mode}_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print("saved to", out)


if __name__ == "__main__":
    main()