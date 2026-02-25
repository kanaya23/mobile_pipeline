"""Stage 3 — Super resolution. Real-ESRGAN (fast) or diffusion model (quality)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .config import CONFIG, Mode
from .utils import run_template, ensure_uint8


def _ensure_weights() -> Path:
    p = CONFIG.fast_sr_weights_path
    if p is None:
        raise RuntimeError("CONFIG.fast_sr_weights_path not set")
    if p.exists():
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Downloading Real-ESRGAN weights → {p} ...")
        urllib.request.urlretrieve(CONFIG.fast_sr_weights_url, str(p))
        print("  Done.")
    except Exception as exc:
        if CONFIG.strict_components:
            raise RuntimeError("Failed to download Real-ESRGAN weights") from exc
        print(f"WARN: weight download failed ({exc}); SR will use bicubic fallback.")
    return p


def fast_sr(img: np.ndarray) -> np.ndarray:
    """Real-ESRGAN ×4. Falls back to bicubic if packages unavailable."""
    img8 = ensure_uint8(img)

    try:
        import torch
        from basicsr.archs.srvgg_arch import SRVGGNetCompact
        from realesrgan import RealESRGANer
    except ImportError as exc:
        if CONFIG.strict_components:
            raise RuntimeError("Install torch + realesrgan + basicsr") from exc
        print("WARN: Real-ESRGAN packages missing — bicubic ×4 fallback.")
        h, w = img8.shape[:2]
        return cv2.resize(img8, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)

    try:
        weights = _ensure_weights()
        if not weights.exists():
            raise RuntimeError("Weights unavailable")
        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu"
        )
        up = RealESRGANer(
            scale=4,
            model_path=str(weights),
            model=model,
            tile=512,
            tile_pad=10,
            pre_pad=0,
            half=torch.cuda.is_available(),
        )
        out, _ = up.enhance(img8, outscale=4)
        return out
    except Exception as exc:
        if CONFIG.strict_components:
            raise RuntimeError("Real-ESRGAN inference failed") from exc
        print(f"WARN: Real-ESRGAN failed ({exc}) — bicubic ×4 fallback.")
        h, w = img8.shape[:2]
        return cv2.resize(img8, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)


def quality_sr(img: np.ndarray) -> np.ndarray:
    """DiffBIR / StableSR via command template. Falls back to fast_sr if not configured."""
    if CONFIG.quality_sr_command_template:
        return run_template(CONFIG.quality_sr_command_template, ensure_uint8(img), "Quality SR")
    if CONFIG.strict_components:
        raise RuntimeError("Set CONFIG.quality_sr_command_template for DiffBIR / StableSR")
    print("WARN: quality_sr_command_template not set — falling back to Fast SR.")
    return fast_sr(img)


def sr_stage(img: np.ndarray, mode: Mode) -> np.ndarray:
    return fast_sr(img) if mode == Mode.FAST else quality_sr(img)
