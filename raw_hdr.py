"""Stage 1 — RAW development (rawpy) and multi-frame HDR fusion (Mertens)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .config import CONFIG
from .utils import read_bgr, to_float01, ensure_uint8


def develop_dng(path: Path) -> np.ndarray:
    """
    Develop a DNG into a 16-bit sRGB BGR array (float32 carrier, [0,1]).

    FIX: gamma=(1,1) produced linear-light data which looks pitch black
         (18% grey ≈ 18/255 in linear vs. ~186/255 after sRGB gamma).
         We now apply the standard sRGB piecewise curve (2.222, 4.5) and
         let rawpy normalise exposure automatically.
    """
    try:
        import rawpy
    except ImportError as exc:
        raise RuntimeError("rawpy not installed — run: pip install rawpy") from exc

    with rawpy.imread(str(path)) as raw:
        # ── KEY FIXES ─────────────────────────────────────────────────────
        # 1. gamma=(2.222, 4.5)  → standard sRGB gamma; was (1,1) = black image
        # 2. no_auto_bright=False → rawpy normalises exposure correctly
        # 3. output_color=rawpy.ColorSpace.sRGB → explicit colour space
        # 4. output_bps=16 → keep full bit-depth, float01 conversion is lossless
        rgb16 = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=False,
            output_bps=16,
            gamma=(2.222, 4.5),
            output_color=rawpy.ColorSpace.sRGB,
        )

    # BGR for the rest of the cv2-based pipeline
    return cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR)


def fuse_hdr(frames: List[np.ndarray]) -> np.ndarray:
    """
    Align and Mertens-fuse burst frames → float32 [0,1].

    Uses AlignMTB (fast, exposure-agnostic) before fusion.
    Mertens fusion handles exposure variation automatically — ideal for
    shooting the same scene at -1EV / 0EV / +1EV.
    """
    if len(frames) == 1:
        return to_float01(frames[0])

    proxies = [ensure_uint8(f) for f in frames]
    aligned = [p.copy() for p in proxies]

    try:
        cv2.createAlignMTB().process(proxies, aligned)
    except cv2.error:
        # alignment can fail on near-identical frames; proceed unaligned
        aligned = proxies

    fused = cv2.createMergeMertens().process(
        [a.astype(np.float32) / 255.0 for a in aligned]
    )
    return np.clip(fused, 0.0, 1.0).astype(np.float32)


def load_primary(primary: Path, burst_paths: Optional[List[Path]] = None) -> np.ndarray:
    """
    Load the primary image (DNG or JPEG/PNG) plus optional burst frames.
    Returns float32 [0,1] BGR.
    """
    first = develop_dng(primary) if primary.suffix.lower() == ".dng" else read_bgr(primary)

    burst_paths = burst_paths or []
    if not CONFIG.use_hdr_fusion or not burst_paths:
        return to_float01(first)

    frames = [first]
    for p in burst_paths:
        frames.append(develop_dng(p) if p.suffix.lower() == ".dng" else read_bgr(p))
    return fuse_hdr(frames)
