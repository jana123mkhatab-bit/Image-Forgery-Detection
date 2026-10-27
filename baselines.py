"""Classical image-forgery baselines. Each localization method returns a float
heatmap in [0, 1] with the same HxW as the input (higher = more likely tampered).
Copy-move also returns the number of geometrically consistent matches.
"""
import io

import cv2
import numpy as np
from PIL import Image


def _norm(x):
    x = x.astype(np.float32)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    return np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)


def _jpeg_roundtrip(img_rgb, quality):
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


# ---------------------------------------------------------------- 1. ELA
def ela_map(img_rgb, quality=90, blur=7):
    """Error Level Analysis: re-save as JPEG and look at the residual.
    Regions with a different compression history stand out."""
    diff = np.abs(img_rgb.astype(np.float32) - _jpeg_roundtrip(img_rgb, quality).astype(np.float32))
    m = diff.max(axis=2)
    m = cv2.GaussianBlur(m, (0, 0), blur)
    return _norm(m)


# ------------------------------------------------- 2. Noise inconsistency
def noise_map(img_rgb, block=32):
    """Noise-level inconsistency: estimate local noise std from a high-pass
    residual and flag blocks whose noise deviates from the image median."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    resid = gray - cv2.medianBlur(gray.astype(np.uint8), 3).astype(np.float32)
    h, w = gray.shape
    gh, gw = max(h // block, 1), max(w // block, 1)
    sigma = np.zeros((gh, gw), np.float32)
    for i in range(gh):
        for j in range(gw):
            patch = resid[i * block:(i + 1) * block, j * block:(j + 1) * block]
            # robust std (MAD) so edges do not dominate
            sigma[i, j] = 1.4826 * np.median(np.abs(patch - np.median(patch)))
    dev = np.abs(np.log(sigma + 1e-3) - np.median(np.log(sigma + 1e-3)))
    dev = cv2.resize(dev, (w, h), interpolation=cv2.INTER_CUBIC)
    return _norm(cv2.GaussianBlur(dev, (0, 0), block / 4))


# ------------------------------------------------------- 3. JPEG ghost
def jpeg_ghost_map(img_rgb, qualities=range(50, 100, 5), win=16):
    """JPEG ghost: recompress at many qualities. A region last saved at a
    different quality than the rest has its error minimum at another q.
    Pixels whose argmin-quality differs from the dominant one are flagged."""
    stack = []
    for q in qualities:
        d = ((img_rgb.astype(np.float32) - _jpeg_roundtrip(img_rgb, q).astype(np.float32)) ** 2).mean(axis=2)
        d = cv2.blur(d, (win, win))
        stack.append((d - d.min()) / (d.max() - d.min() + 1e-8))
    stack = np.stack(stack)                      # Q x H x W
    arg = stack.argmin(axis=0)
    mode = np.bincount(arg.ravel()).argmax()
    flagged = (arg != mode).astype(np.float32)
    return cv2.GaussianBlur(flagged, (0, 0), win / 2)


# ------------------------------------------------------ 4. Copy-move
def copy_move(img_rgb, min_dist=40, ratio=0.8, min_inliers=8, n_features=5000):
    """Copy-move via ORB matching of the image against itself + RANSAC.
    Returns (mask float HxW in {0,1}, n_inliers)."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    mask = np.zeros((h, w), np.float32)
    orb = cv2.ORB_create(nfeatures=n_features)
    kps, des = orb.detectAndCompute(gray, None)
    if des is None or len(kps) < 10:
        return mask, 0
    pts = np.float32([k.pt for k in kps])
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des, des, k=3)
    src, dst = [], []
    for m in matches:
        # m[0] is the keypoint matched with itself; use the next two
        if len(m) < 3:
            continue
        a, b = m[1], m[2]
        if a.distance < ratio * b.distance and np.linalg.norm(pts[a.queryIdx] - pts[a.trainIdx]) > min_dist:
            src.append(pts[a.queryIdx])
            dst.append(pts[a.trainIdx])
    if len(src) < min_inliers:
        return mask, 0
    src, dst = np.float32(src), np.float32(dst)
    _, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if inl is None or int(inl.sum()) < min_inliers:
        return mask, 0
    inl = inl.ravel().astype(bool)
    for p in np.vstack([src[inl], dst[inl]]):
        cv2.circle(mask, tuple(int(v) for v in p), 12, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 6)
    return (mask > 0.2).astype(np.float32), int(inl.sum())


METHODS = {"ela": ela_map, "noise": noise_map, "jpeg_ghost": jpeg_ghost_map}
