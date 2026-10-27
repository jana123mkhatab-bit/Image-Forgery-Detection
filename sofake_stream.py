"""Training stream for the So-Fake-Set dataset (Hugging Face, streamed, nothing downloaded).

Columns used: image, label ("real" / "tampered" / "full_synthetic"), mask (tampered only).
Maps to our classes: 0 authentic, 1 edited, 2 AI-generated. The first record's columns are printed
and a missing column stops the run with a clear message (the layout was read off the dataset page).

Differences from data_stream.ForgeryStream:
  * classes are balanced by rejection sampling (the dataset is not balanced);
  * crops always keep the edit (forgery_synth.random_crop);
  * no artificial JPEG/blur/noise by default: these are already social-media images;
  * optional `synthetic_frac`: some real images get our own splice / copy-move, so classical
    Photoshop-style edits stay represented.
"""
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

import forgery_synth as fs


def to_class(v):
    s = str(v).upper()
    if "SYNTH" in s:
        return 2
    if "TAMPER" in s or "EDIT" in s:
        return 1
    if "REAL" in s or "AUTHENTIC" in s:
        return 0
    return None


def open_stream(spec, split, seed):
    from datasets import load_dataset
    ds = load_dataset(spec["id"], split=split, streaming=True,
                      **({"name": spec["config"]} if spec.get("config") else {}))
    return iter(ds.shuffle(seed=seed, buffer_size=spec.get("buffer", 1000)))


class SoFakeStream(IterableDataset):
    """Yields (image 3xHxW float 0-1, mask 1xHxW float, label long)."""

    def __init__(self, cfg, split="train", seed=0, train=True, max_items=None):
        self.cfg, self.split, self.seed, self.train, self.max_items = cfg, split, seed, train, max_items
        self.size = cfg["data"]["crop"]

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        d = self.cfg["data"]
        spec = d["sofake"]
        rng = np.random.default_rng(self.seed + wid)
        target = np.array(spec.get("class_mix", [1 / 3, 1 / 3, 1 / 3]), float)
        target /= target.sum()
        seen = np.ones(3)  # running class frequencies of the raw stream
        syn_frac = d.get("synthetic_frac", 0.0) if self.train else 0.0
        min_px = d.get("min_edit_px", 256)
        img_key, lab_key, mask_key = (spec.get("image_col", "image"), spec.get("label_col", "label"),
                                      spec.get("mask_col", "mask"))
        donor, n_out, first = None, 0, True
        limit = None if self.max_items is None else -(-self.max_items // nw)
        for rec in open_stream(spec, self.split, self.seed + wid):
            if first:
                first = False
                if wid == 0:
                    print("columns:", list(rec.keys()))
                for c in (img_key, lab_key):
                    if c not in rec:
                        raise SystemExit(f"column '{c}' missing; columns are {list(rec.keys())}")
            cls = to_class(rec[lab_key])
            if cls is None:
                continue
            seen[cls] += 1
            w = target / (seen / seen.sum())            # how much each class is over/under-represented
            if rng.random() > w[cls] / w.max():
                continue
            img = np.array(rec[img_key].convert("RGB"))
            if max(img.shape[:2]) > 4096:
                continue
            mask = np.zeros(img.shape[:2], np.uint8)
            label = cls
            if cls == 1:
                m = rec.get(mask_key)
                if m is None:
                    continue
                import cv2
                m = np.array(m.convert("L"))
                mask = (cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
                        ).astype(np.uint8) * 255
                if (mask > 127).sum() < min_px:
                    continue
            elif cls == 0 and donor is not None and rng.random() < syn_frac:
                img, mask = (fs.splice(img, donor, rng) if rng.random() < 0.6 else fs.copy_move(img, rng))
                label = 1
            if cls == 0:
                donor = img.copy() if label == 0 else donor
            if self.train and rng.random() < 0.5:                    # horizontal flip
                img, mask = img[:, ::-1], mask[:, ::-1]
            img, mask = fs.random_crop(np.ascontiguousarray(img), np.ascontiguousarray(mask), self.size, rng)
            if label == 1 and int((mask > 127).sum()) < min_px:
                continue
            x = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255
            m = torch.from_numpy((mask > 127).astype(np.float32))[None]
            yield x, m, torch.tensor(label)
            n_out += 1
            if limit is not None and n_out >= limit:
                return
