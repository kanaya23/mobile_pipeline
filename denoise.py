"""Stage 2 — Noise reduction. Bilateral fallback; NAFNet/Restormer via command template."""
from __future__ import annotations

import numpy as np
import cv2

from .config import CONFIG
from .utils import to_float01, run_template, ensure_uint8


def denoise_stage(img: np.ndarray) -> np.ndarray:
    """
    Returns float32 [0,1].

    Fast path (no template):  cv2.bilateralFilter — operates in float space,
                               much gentler than fastNlMeansDenoising on detail.
    Full path (template set): delegates to NAFNet or Restormer via subprocess.
    """
    if CONFIG.denoise_command_template:
        out = run_template(CONFIG.denoise_command_template, ensure_uint8(img), "Denoise")
        return to_float01(out)

    if CONFIG.strict_components:
        raise RuntimeError("Set CONFIG.denoise_command_template for NAFNet / Restormer")

    work = to_float01(img)
    # bilateralFilter on float32 needs d=-1 (auto from sigmaSpace)
    den = cv2.bilateralFilter(work, d=0, sigmaColor=0.08, sigmaSpace=5)
    return np.clip(den, 0.0, 1.0).astype(np.float32)
