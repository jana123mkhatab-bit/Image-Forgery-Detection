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
import io
import time

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
    """Stream with image columns left UNDECODED (raw bytes), so one corrupt image in the dataset
    cannot crash the loader: we decode ourselves and skip bad records (see decode_image)."""
    from datasets import Image as HFImage, load_dataset
    ds = load_dataset(spec["id"], split=split, streaming=True,
                      **({"name": spec["config"]} if spec.get("config") else {}))
    for col in (spec.get("image_col", "image"), spec.get("mask_col", "mask")):
        if col in (getattr(ds, "features", None) or {}):
            ds = ds.cast_column(col, HFImage(decode=False))
    return iter(ds.shuffle(seed=seed, buffer_size=spec.get("buffer", 1000)))


def robust_stream(spec, split, seed, max_restarts=30):
    """Like open_stream, but if reading fails (network drop, bad shard) it reopens the stream with a
    new shuffle seed instead of killing the training run."""
    attempt = 0
    while True:
        try:
            yield from open_stream(spec, split, seed + 7919 * attempt)
            return
        except Exception as e:  # noqa: BLE001 - any read error: log, back off, reopen
            attempt += 1
            print(f"[stream] read error ({type(e).__name__}: {str(e)[:120]}); reopening ({attempt}/{max_restarts})")
            if attempt > max_restarts:
                raise
            time.sleep(min(2 * attempt, 30))


def decode_image(x):
    """PIL image from a raw {'bytes','path'} dict (or an already decoded PIL image). Raises on corrupt data."""
    from PIL import Image
    if x is None:
        return None
    if isinstance(x, dict):
        x = Image.open(io.BytesIO(x["bytes"])) if x.get("bytes") is not None else Image.open(x["path"])
    x.load()  # forces full decode so truncated files fail HERE, where we catch them
    return x


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
        syn_frac = d.get("synthetic_frac", 0.0) if self.train else 0.0
        min_px = d.get("min_edit_px", 256)
        img_key, lab_key, mask_key = (spec.get("image_col", "image"), spec.get("label_col", "label"),
                                      spec.get("mask_col", "mask"))
        donor, n_out, first, bad = None, 0, True, 0
        jpeg_frac = spec.get("jpeg_frac")
        acc = np.zeros((3, 2))  # accepted [jpeg, non-jpeg] counts per source class
        self.acc = acc
        wait, max_wait = 0, 400  # safety: after 400 balance-rejections in a row, accept anyway (never stall)
        limit = None if self.max_items is None else -(-self.max_items // nw)
        for rec in robust_stream(spec, self.split, self.seed + wid):
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
            # class balance on what is actually KEPT (acc also counts the format filter below):
            # a class that is ahead of its target share waits for the others to catch up
            kept = acc.sum(1) / target
            if kept[cls] > kept.min() + 8 and wait < max_wait:
                wait += 1
                continue
            try:
                pil = decode_image(rec[img_key])
                is_jpeg = int(pil.format == "JPEG")
                img = np.array(pil.convert("RGB"))
                m = decode_image(rec.get(mask_key)) if cls == 1 else None
            except (OSError, ValueError, SyntaxError, KeyError, TypeError):
                bad += 1                      # corrupt / truncated image: skip it
                if bad in (1, 10, 100, 1000):
                    print(f"[stream] skipped {bad} corrupt records so far")
                continue
            if max(img.shape[:2]) > 4096:
                continue
            # FORMAT SHORTCUT GUARD: in this dataset real images are mostly JPEG and fakes mostly PNG,
            # so "is it a JPEG" would predict the class. Keep the JPEG share of every class near
            # `jpeg_frac` (drops surplus real JPEGs / fake PNGs); `degrade` below then recompresses
            # all classes alike.
            if jpeg_frac is not None:
                a_j, a_p = acc[cls]
                if wait < max_wait and (
                        (is_jpeg and a_j > jpeg_frac / (1 - jpeg_frac) * a_p + 3)
                        or (not is_jpeg and a_p > (1 - jpeg_frac) / jpeg_frac * a_j + 3)):
                    wait += 1
                    continue
                acc[cls][0 if is_jpeg else 1] += 1
            mask = np.zeros(img.shape[:2], np.uint8)
            label = cls
            if cls == 1:
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
            if d.get("degrade", True):
                img = fs.degrade(img, rng)    # same random JPEG/blur/noise for every class, after cropping
            if label == 1 and int((mask > 127).sum()) < min_px:
                continue
            wait = 0
            x = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255
            m = torch.from_numpy((mask > 127).astype(np.float32))[None]
            yield x, m, torch.tensor(label)
            n_out += 1
            if limit is not None and n_out >= limit:
                return