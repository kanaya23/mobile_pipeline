"""Stage 3 — Super resolution. Real-ESRGAN (fast) or diffusion model (quality)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

from .config import CONFIG, Mode
from .utils import run_template, ensure_uint8


# ── GPU helpers ───────────────────────────────────────────────────────────────

def _gpu_info() -> tuple[bool, int]:
    """Returns (cuda_available, vram_mb). Safe to call even without torch."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, 0
        vram = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        return True, vram
    except Exception:
        return False, 0


def _tile_size_for_vram(vram_mb: int) -> int:
    """
    Conservative tile sizes that avoid OOM on T4 (16 GB) / low-VRAM cards.
    Larger tiles = fewer stitching seams and faster on GPU.
    """
    if vram_mb >= 14_000:   # T4 16 GB, A100, etc.
        return 768
    if vram_mb >= 6_000:    # RTX 3060 / 3070, A10G
        return 512
    if vram_mb >= 3_000:    # GTX 1060, entry GPU
        return 256
    return 128              # CPU / integrated / very low VRAM


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
    """
    Real-ESRGAN ×4.

    FIX: was silently falling back to bicubic even with CUDA available because
    the packages were never checked properly. Now:
      - Explicit CUDA detection and half-precision on GPU
      - Tile size adapted to available VRAM (768 on T4 = fewer seams, faster)
      - Helpful error message when packages are missing so user knows to install
    """
    img8 = ensure_uint8(img)
    cuda_ok, vram_mb = _gpu_info()

    try:
        import torch
        from basicsr.archs.srvgg_arch import SRVGGNetCompact
        from realesrgan import RealESRGANer
    except ImportError:
        if CONFIG.strict_components:
            raise RuntimeError(
                "Install Real-ESRGAN: pip install torch basicsr realesrgan\n"
                "For GPU:  pip install torch --index-url https://download.pytorch.org/whl/cu118"
            )
        print("WARN: Real-ESRGAN packages missing — bicubic ×4 fallback.")
        print("      To enable GPU super-resolution run:")
        print("      pip install torch basicsr realesrgan")
        h, w = img8.shape[:2]
        return cv2.resize(img8, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)

    try:
        weights   = _ensure_weights()
        tile_size = _tile_size_for_vram(vram_mb) if cuda_ok else 256
        use_half  = cuda_ok  # FP16 on GPU — ~2× faster, no quality loss for SR

        if cuda_ok:
            print(f"  SR: CUDA ({vram_mb} MB VRAM), tile={tile_size}, half={use_half}")
        else:
            print("  SR: CPU mode (no CUDA) — consider enabling GPU in Kaggle settings")

        model = SRVGGNetCompact(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32,
            upscale=4, act_type="prelu",
        )
        upsampler = RealESRGANer(
            scale=4,
            model_path=str(weights),
            model=model,
            tile=tile_size,
            tile_pad=10,
            pre_pad=0,
            half=use_half,
            device=torch.device("cuda" if cuda_ok else "cpu"),
        )
        out, _ = upsampler.enhance(img8, outscale=4)
        return out

    except Exception as exc:
        if CONFIG.strict_components:
            raise RuntimeError(f"Real-ESRGAN inference failed: {exc}") from exc
        print(f"WARN: Real-ESRGAN failed ({exc}) — bicubic ×4 fallback.")
        h, w = img8.shape[:2]
        return cv2.resize(img8, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)


def quality_sr(img: np.ndarray) -> np.ndarray:
    """
    DiffBIR / StableSR via command template.
    Falls back to fast_sr (Real-ESRGAN) if template not configured.

    Wire up by setting CONFIG.quality_sr_command_template — see config.py
    or run auto_configure_from_models() from the pipeline __init__.
    """
    if CONFIG.quality_sr_command_template:
        return run_template(CONFIG.quality_sr_command_template, ensure_uint8(img), "Quality SR")
    if CONFIG.strict_components:
        raise RuntimeError("Set CONFIG.quality_sr_command_template for DiffBIR / StableSR")
    print("WARN: quality_sr_command_template not set — falling back to Fast SR (Real-ESRGAN).")
    return fast_sr(img)


def sr_stage(img: np.ndarray, mode: Mode) -> np.ndarray:
    """Dispatch to fast (Real-ESRGAN) or quality (DiffBIR/StableSR) super-resolution."""
    return fast_sr(img) if mode == Mode.FAST else quality_sr(img)
