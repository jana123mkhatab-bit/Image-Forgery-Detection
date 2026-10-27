"""Streaming data pipeline: nothing is downloaded to disk up front.

Real images are streamed from the Hugging Face Hub (datasets, streaming=True).
Tampered samples are generated on the fly (forgery_synth.py). AI-generated
images are streamed from a second HF dataset and labelled class 2.

Classes: 0 = authentic, 1 = edited (splice / copy-move), 2 = AI-generated.
Dataset ids live in configs/default.yaml. VERIFY each id on huggingface.co
before a long run (names and availability change).
"""
import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

import forgery_synth as fs


def _to_rgb(x):
    if isinstance(x, dict) and "bytes" in x:
        x = Image.open(io.BytesIO(x["bytes"]))
    return np.array(x.convert("RGB"))


def _local_stream(spec, seed):
    """Endless shuffled iterator over image files in a folder (e.g. COCO val2017
    downloaded to Colab's temporary disk). Yields {image_key: PIL.Image}."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = sorted(p for p in Path(spec["local_dir"]).rglob("*") if p.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"no images found under {spec['local_dir']}")
    key, rng = spec.get("image_key", "image"), np.random.default_rng(seed)
    while True:
        for i in rng.permutation(len(files)):
            yield {key: Image.open(files[i])}


def _stream(spec, seed):
    if spec.get("local_dir"):
        return _local_stream(spec, seed)
    from datasets import load_dataset  # only needed for Hub streaming
    ds = load_dataset(spec["id"], split=spec.get("split", "train"), streaming=True,
                      **({"name": spec["config"]} if spec.get("config") else {}))
    return iter(ds.shuffle(seed=seed, buffer_size=spec.get("buffer", 500)))


class ForgeryStream(IterableDataset):
    """Yields (image CxHxW float 0-1, mask 1xHxW float, label long)."""

    def __init__(self, cfg, seed=0):
        self.cfg, self.seed = cfg, seed
        self.size = cfg["data"]["crop"]

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng(self.seed + wid)
        d = self.cfg["data"]
        real = _stream(d["real"], self.seed + wid)
        fake = _stream(d["ai"], self.seed + 1000 + wid) if d.get("ai") else None
        p_ai, p_auth = d["p_ai"], d["p_authentic"]
        donor = None
        for rec in real:
            key = d["real"].get("image_key", "image")
            img = _to_rgb(rec[key])
            r = rng.random()
            if fake is not None and r < p_ai:
                try:
                    img = _to_rgb(next(fake)[d["ai"].get("image_key", "image")])
                except StopIteration:
                    fake = None
                    continue
                mask, label = np.zeros(img.shape[:2], np.uint8), 2
            elif r < p_ai + p_auth or donor is None:
                mask, label = np.zeros(img.shape[:2], np.uint8), 0
            else:
                img, mask = (fs.splice(img, donor, rng) if rng.random() < 0.6 else fs.copy_move(img, rng))
                label = 1
            donor = _to_rgb(rec[key])
            img = fs.degrade(img, rng)
            img, mask = fs.random_crop(img, mask, self.size, rng)
            # drop edited samples whose edit is (almost) gone after cropping: a wrong label is worse than no sample
            if label == 1 and int((mask > 127).sum()) < d.get("min_edit_px", 256):
                continue
            x = torch.from_numpy(img).permute(2, 0, 1).float() / 255
            m = torch.from_numpy((mask > 127).astype(np.float32))[None]
            yield x, m, torch.tensor(label)