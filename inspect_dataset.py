"""Sanity-check a Hugging Face forgery dataset BEFORE training on it (streamed, nothing downloaded).

  python inspect_dataset.py --n 300 --out /content/drive/MyDrive/forgery/results/dataset_check

Prints, per class: counts, image size and file format, mask coverage, empty masks, mask/image
size mismatches, and which generators appear. Flags SHORTCUTS: properties (image format, size,
JPEG quality) that differ so much between classes that a model could use them instead of
looking at the content. Saves a contact sheet with mask overlays so you can eyeball the labels.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from sofake_stream import to_class

NAMES = ["authentic", "edited", "ai_generated"]


def jpeg_quality_guess(im):
    """Rough JPEG quality from the luminance quantization table (None if not a JPEG)."""
    q = getattr(im, "quantization", None)
    if not q:
        return None
    t = q[0]
    return float(np.mean(list(t.values()) if isinstance(t, dict) else list(t)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="saberzl/So-Fake-Set")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=300, help="records to scan")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/dataset_check")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    ds = load_dataset(a.id, split=a.split, streaming=True).shuffle(seed=a.seed, buffer_size=1000)
    st = defaultdict(lambda: defaultdict(list))
    gens, labels, fmts = defaultdict(Counter), Counter(), defaultdict(Counter)
    sheet = defaultdict(list)
    for i, rec in enumerate(ds):
        if i == 0:
            print("columns:", list(rec.keys()))
        if i >= a.n:
            break
        raw = rec["label"]
        labels[str(raw)] += 1
        c = to_class(raw)
        if c is None:
            continue
        im = rec["image"]
        fmts[c][im.format or "decoded"] += 1
        w, h = im.size
        st[c]["w"].append(w); st[c]["h"].append(h); st[c]["mp"].append(w * h / 1e6)
        q = jpeg_quality_guess(im)
        if q is not None:
            st[c]["jpeg_q"].append(q)
        if rec.get("generator"):
            gens[c][str(rec["generator"])] += 1
        m = rec.get("mask")
        if c == 1:
            if m is None:
                st[c]["no_mask"].append(1)
            else:
                ma = np.array(m.convert("L")) > 127
                st[c]["mask_frac"].append(float(ma.mean()))
                st[c]["mask_empty"].append(int(not ma.any()))
                st[c]["mask_size_mismatch"].append(int((ma.shape[1], ma.shape[0]) != im.size))
        elif m is not None:
            ma = np.array(m.convert("L")) > 127
            st[c]["mask_on_nonedited_frac"].append(float(ma.mean()))
        if len(sheet[c]) < 8:
            img = np.array(im.convert("RGB"))
            tile = cv2.resize(img, (256, 256))
            if c == 1 and m is not None:
                mm = cv2.resize((np.array(m.convert("L")) > 127).astype(np.uint8) * 255, (256, 256),
                                interpolation=cv2.INTER_NEAREST)
                heat = np.zeros_like(tile); heat[..., 0] = mm
                tile = cv2.addWeighted(tile, 0.65, heat, 0.6, 0)
            cv2.putText(tile, NAMES[c], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            sheet[c].append(tile)

    rep = {"raw_label_values": dict(labels), "scanned": min(a.n, sum(labels.values()))}
    for c in range(3):
        s = st[c]
        if not s["w"]:
            rep[NAMES[c]] = "no samples"
            continue
        r = {"n": len(s["w"]), "formats": dict(fmts[c]),
             "megapixels_median": round(float(np.median(s["mp"])), 3),
             "width_median": int(np.median(s["w"])), "height_median": int(np.median(s["h"])),
             "pct_exact_256x256": round(100 * float(np.mean([(w == 256 and h == 256)
                                                              for w, h in zip(s["w"], s["h"])])), 1)}
        if s["jpeg_q"]:
            r["jpeg_qtable_mean_median"] = round(float(np.median(s["jpeg_q"])), 2)
            r["pct_jpeg"] = round(100 * len(s["jpeg_q"]) / len(s["w"]), 1)
        if c == 1:
            r["missing_mask"] = len(s["no_mask"])
            r["empty_masks"] = int(sum(s["mask_empty"]))
            r["mask_size_mismatch"] = int(sum(s["mask_size_mismatch"]))
            if s["mask_frac"]:
                r["mask_area_pct_median"] = round(100 * float(np.median(s["mask_frac"])), 2)
                r["mask_area_pct_p10_p90"] = [round(100 * float(np.percentile(s["mask_frac"], p)), 2) for p in (10, 90)]
        elif s["mask_on_nonedited_frac"]:
            r["masks_present_on_nonedited"] = len(s["mask_on_nonedited_frac"])
            r["mask_area_pct_median"] = round(100 * float(np.median(s["mask_on_nonedited_frac"])), 2)
        if gens[c]:
            r["generators"] = dict(gens[c].most_common(12))
        rep[NAMES[c]] = r

    flags = []
    ok = [c for c in range(3) if isinstance(rep[NAMES[c]], dict)]
    if len(ok) >= 2:
        for key, tol in (("pct_exact_256x256", 30), ("pct_jpeg", 30)):
            v = {NAMES[c]: rep[NAMES[c]].get(key) for c in ok if rep[NAMES[c]].get(key) is not None}
            if len(v) >= 2 and max(v.values()) - min(v.values()) > tol:
                flags.append(f"SHORTCUT RISK: {key} differs by class {v}")
        mp = {NAMES[c]: rep[NAMES[c]]["megapixels_median"] for c in ok}
        if max(mp.values()) > 2.5 * max(min(mp.values()), 1e-6):
            flags.append(f"SHORTCUT RISK: median image size differs a lot by class {mp}")
        fm = {NAMES[c]: sorted(rep[NAMES[c]]["formats"]) for c in ok}
        if len({tuple(v) for v in fm.values()}) > 1:
            flags.append(f"SHORTCUT RISK: file formats differ by class {fm}")
    e = rep.get("edited")
    if isinstance(e, dict):
        if e["missing_mask"] or e["empty_masks"]:
            flags.append(f"LABEL PROBLEM: {e['missing_mask']} edited without mask, {e['empty_masks']} with empty mask")
        if e["mask_size_mismatch"]:
            flags.append(f"NOTE: {e['mask_size_mismatch']} masks differ in size from their image (we resize them)")
    rep["flags"] = flags or ["no shortcut or label problem detected in this sample"]

    rows = [np.hstack(sheet[c] + [np.zeros((256, 256, 3), np.uint8)] * (8 - len(sheet[c])))
            for c in range(3) if sheet[c]]
    if rows:
        cv2.imwrite(str(out / "contact_sheet.jpg"), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print("saved to", out)


if __name__ == "__main__":
    main()
