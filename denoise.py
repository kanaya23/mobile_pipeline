"""Stage 2 — Noise reduction. NAFNet/Restormer via command template; bilateral fallback."""
from __future__ import annotations

import numpy as np
import cv2

from .config import CONFIG
from .utils import to_float01, run_template, ensure_uint8


def denoise_stage(img: np.ndarray) -> np.ndarray:
    """
    Returns float32 [0,1].

    Priority:
      1. NAFNet or Restormer via CONFIG.denoise_command_template (best quality)
      2. cv2.bilateralFilter (CPU fallback — gentle, detail-preserving)

    FIX: previous bilateral params (sigmaColor=0.08, sigmaSpace=5, d=0)
         over-smoothed fine textures. Tuned values below preserve micro-detail
         while still reducing sensor noise from the Redmi IMX882 sensor.

    Auto-wiring: if no template is set but model weights are detected on disk,
    this function now attempts to build the template automatically via
    CONFIG._try_auto_wire_denoise().
    """
    # Try auto-wiring if templates haven't been set yet
    if not CONFIG.denoise_command_template:
        _try_auto_wire_denoise()

    if CONFIG.denoise_command_template:
        out = run_template(CONFIG.denoise_command_template, ensure_uint8(img), "Denoise")
        return to_float01(out)

    if CONFIG.strict_components:
        raise RuntimeError(
            "Set CONFIG.denoise_command_template for NAFNet / Restormer.\n"
            "Or call auto_configure_from_models() after setup_model_assets()."
        )

    # ── Bilateral fallback ────────────────────────────────────────────────────
    # FIX: d=9 (explicit neighbourhood), sigmaColor=0.05 (tighter colour range
    #      = sharper edges preserved), sigmaSpace=7 (enough spatial smoothing
    #      for shot noise without smearing fine texture).
    work = to_float01(img)
    den  = cv2.bilateralFilter(
        work,
        d=9,
        sigmaColor=0.05,
        sigmaSpace=7,
    )
    return np.clip(den, 0.0, 1.0).astype(np.float32)


def _try_auto_wire_denoise() -> None:
    """
    If NAFNet or Restormer weights exist on disk and a NAFNet/Restormer inference
    script can be found, auto-set CONFIG.denoise_command_template.
    Called lazily — safe to call many times.
    """
    if CONFIG.denoise_command_template:
        return  # already set

    from .config import STORAGE
    import shutil

    nafnet_w   = STORAGE["models"] / "NAFNet-SIDD-width64.pth"
    restormer_w = STORAGE["models"] / "restormer_denoising.pth"

    # Prefer NAFNet (faster on T4, equal quality to Restormer on SIDD benchmark)
    if nafnet_w.exists():
        script = _find_script(["NAFNet/basicsr/test.py", "NAFNet/predict.py",
                                "/kaggle/working/NAFNet/basicsr/test.py"])
        if script:
            # NAFNet test.py requires a .yml; use the subprocess template wrapper
            CONFIG.denoise_command_template = (
                f"python {script} "
                "--input {{input}} --output {{output}} "
                f"--model_path {nafnet_w}"
            )
            print(f"Auto-wired NAFNet denoiser: {script}")
            return

    if restormer_w.exists():
        script = _find_script(["Restormer/demo.py",
                                "/kaggle/working/Restormer/demo.py"])
        if script:
            CONFIG.denoise_command_template = (
                f"python {script} "
                "--task Real_Denoising "
                "--input_dir {{workdir}} "
                "--result_dir {{workdir}} "
                f"--pretrained_weights {restormer_w}"
            )
            print(f"Auto-wired Restormer denoiser: {script}")
            return


def _find_script(candidates: list[str]) -> str | None:
    from pathlib import Path as _P
    for c in candidates:
        if _P(c).exists():
            return c
    return None
