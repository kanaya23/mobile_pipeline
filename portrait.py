"""Stage 5 — Portrait bokeh: subject segmentation + depth-based background blur."""
from __future__ import annotations

import numpy as np
import cv2

from .config import CONFIG
from .utils import ensure_uint8, run_template


# ── Fallback subject mask (GrabCut) ─────────────────────────────────────────

def simple_subject_mask(img8: np.ndarray) -> np.ndarray:
    """GrabCut with a centre-biased rect — good enough for solo portrait shots."""
    h, w = img8.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(0.15 * w), int(0.10 * h), int(0.70 * w), int(0.80 * h))
    bg = np.zeros((1, 65), np.float64)
    fg = np.zeros((1, 65), np.float64)
    cv2.grabCut(img8, mask, rect, bg, fg, 5, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


# ── Fallback depth from subject mask ─────────────────────────────────────────

def simple_depth_from_subject(subject_mask: np.ndarray) -> np.ndarray:
    """
    Distance transform of the background — pixels far from the subject edge
    get a high "depth" value and therefore more blur.
    Correct direction: background → high value → more blur.
    """
    background = np.where(subject_mask > 0, 0, 1).astype(np.uint8)
    dist = cv2.distanceTransform(background, cv2.DIST_L2, 5)
    if float(dist.max()) <= 0.0:
        return np.zeros_like(subject_mask, dtype=np.uint8)
    return cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# ── External model helper ─────────────────────────────────────────────────────

def _run_map(template: str, img8: np.ndarray, stage: str) -> np.ndarray:
    out = run_template(template, img8, stage, output_name="map.png")
    if out.ndim == 3:
        out = cv2.cvtColor(ensure_uint8(out), cv2.COLOR_BGR2GRAY)
    return out


# ── Layered bokeh compositor ─────────────────────────────────────────────────

def portrait_stage(img: np.ndarray) -> np.ndarray:
    """
    1. Get subject mask (SAM2 or GrabCut fallback)
    2. Get depth map (Depth Anything V2 or distance-transform fallback)
    3. Compute bg = (1-subject) * depth  →  how blurry each pixel should be
    4. Apply layered Gaussian blur keyed to bg intensity bands
    """
    img8 = ensure_uint8(img)

    if CONFIG.sam2_mask_command_template and CONFIG.depth_map_command_template:
        subject = _run_map(CONFIG.sam2_mask_command_template, img8, "SAM2")
        depth   = _run_map(CONFIG.depth_map_command_template, img8, "DepthAnything")
    else:
        if CONFIG.strict_components:
            raise RuntimeError("Portrait requires SAM2 + Depth Anything templates")
        subject = simple_subject_mask(img8)
        depth   = simple_depth_from_subject(subject)

    s  = np.clip(subject.astype(np.float32) / 255.0, 0.0, 1.0)
    d  = np.clip(depth.astype(np.float32)   / 255.0, 0.0, 1.0)
    bg = (1.0 - s) * d   # high where far from subject

    kernels = [k for k in [5, 9, 13, 17, 21, 25, 31]
               if k <= CONFIG.max_bokeh_kernel and k % 2 == 1]
    if not kernels:
        kernels = [5]

    out = img8.astype(np.float32)
    n   = len(kernels)
    for i, k in enumerate(kernels, start=1):
        lo  = (i - 1) / n
        hi  = i / n
        region = ((bg >= lo) & (bg < hi)).astype(np.float32)[..., None]
        blur   = cv2.GaussianBlur(img8, (k, k), 0).astype(np.float32)
        out    = out * (1.0 - region) + blur * region

    return np.clip(out, 0, 255).astype(np.uint8)
