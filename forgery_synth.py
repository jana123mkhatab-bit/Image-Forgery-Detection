"""Synthetic forgery generation (numpy/cv2 only, no downloads, no torch).

Works on any stream of RGB uint8 images, so datasets can be streamed from the
Hugging Face Hub and nothing has to be stored on disk. Every function returns
(forged_rgb, mask_uint8) where mask is 255 on tampered pixels.
"""
import cv2
import numpy as np


def random_blob_mask(h, w, rng, min_frac=0.03, max_frac=0.25):
    """Random irregular blob (union of a few ellipses, warped edge)."""
    area = rng.uniform(min_frac, max_frac) * h * w
    r = np.sqrt(area / np.pi)
    cx, cy = rng.uniform(r, max(w - r, r + 1)), rng.uniform(r, max(h - r, r + 1))
    m = np.zeros((h, w), np.uint8)
    for _ in range(rng.integers(1, 4)):
        ax = int(r * rng.uniform(0.5, 1.2)); ay = int(r * rng.uniform(0.5, 1.2))
        off = rng.normal(0, r * 0.3, 2)
        cv2.ellipse(m, (int(cx + off[0]), int(cy + off[1])), (max(ax, 3), max(ay, 3)),
                    float(rng.uniform(0, 180)), 0, 360, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 2) > 127


def _fit(img, h, w):
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def splice(host, donor, rng, feather=True):
    """Paste a blob from `donor` onto `host` at the same location (optionally
    with scale/rotation jitter and edge feathering)."""
    h, w = host.shape[:2]
    donor = _fit(donor, h, w)
    # random affine jitter on donor so it is not just an aligned crop
    ang, sc = rng.uniform(-20, 20), rng.uniform(0.8, 1.3)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    donor = cv2.warpAffine(donor, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    m = random_blob_mask(h, w, rng)
    alpha = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.5) if feather else m.astype(np.float32)
    out = (host * (1 - alpha[..., None]) + donor * alpha[..., None]).round().astype(np.uint8)
    return out, (m * 255).astype(np.uint8)


def copy_move(img, rng):
    """Copy a blob from one place in the image to another."""
    h, w = img.shape[:2]
    m_src = random_blob_mask(h, w, rng, 0.02, 0.12)
    ys, xs = np.where(m_src)
    if len(ys) == 0:
        return img.copy(), np.zeros((h, w), np.uint8)
    dy, dx = int(rng.integers(-h // 2, h // 2)), int(rng.integers(-w // 2, w // 2))
    if abs(dy) + abs(dx) < 30:
        dy, dx = 40, 40
    ty, tx = np.clip(ys + dy, 0, h - 1), np.clip(xs + dx, 0, w - 1)
    out = img.copy()
    out[ty, tx] = img[ys, xs]
    m = np.zeros((h, w), np.uint8)
    m[ty, tx] = 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return out, m


def degrade(img, rng):
    """Random JPEG / blur / noise so the detector cannot rely on clean pixels."""
    if rng.random() < 0.7:
        q = int(rng.integers(55, 96))
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.cvtColor(cv2.imdecode(buf, 1), cv2.COLOR_BGR2RGB)
    if rng.random() < 0.2:
        img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.3, 1.2)))
    if rng.random() < 0.2:
        img = np.clip(img + rng.normal(0, rng.uniform(1, 4), img.shape), 0, 255).astype(np.uint8)
    return img


def random_crop(img, mask, size, rng):
    h, w = img.shape[:2]
    if h < size or w < size:  # pad rather than resize (resizing destroys traces)
        ph, pw = max(size - h, 0), max(size - w, 0)
        img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
        mask = np.pad(mask, ((0, ph), (0, pw)), mode="constant")
        h, w = img.shape[:2]
    ys, xs = np.nonzero(mask > 127)
    if len(ys) == 0:  # untouched or AI image: plain random crop
        y, x = int(rng.integers(0, h - size + 1)), int(rng.integers(0, w - size + 1))
    else:
        # Tampered image: the crop must keep the edit, otherwise the label says "edited"
        # while no edited pixel is visible. Try random windows anchored on an edited pixel and take
        # the first that keeps at least 256 edited pixels (so edit size stays varied); if none
        # does, keep the best one and let the caller decide.
        best, y, x = -1, 0, 0
        for _ in range(8):
            k = int(rng.integers(0, len(ys)))
            cy, cx = int(ys[k]), int(xs[k])
            y0 = int(rng.integers(max(0, cy - size + 1), min(cy, h - size) + 1))
            x0 = int(rng.integers(max(0, cx - size + 1), min(cx, w - size) + 1))
            area = int((mask[y0:y0 + size, x0:x0 + size] > 127).sum())
            if area > best:
                best, y, x = area, y0, x0
            if area >= 256:
                y, x = y0, x0
                break
    return img[y:y + size, x:x + size], mask[y:y + size, x:x + size]