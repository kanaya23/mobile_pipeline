"""Stage 4 — Color science: Leica, Pixel, and Blend looks."""
from __future__ import annotations

import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from .config import CONFIG, Look
from .utils import ensure_uint8, read_bgr, write_bgr


# ── LUT application ──────────────────────────────────────────────────────────

def apply_lut_ffmpeg(img: np.ndarray, lut_path: Path) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found — install it or remove leica_lut_path")
    if not lut_path.exists():
        raise RuntimeError(f"LUT file missing: {lut_path}")
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        inp, out = td / "in.png", td / "out.png"
        write_bgr(inp, img)
        vf = f"lut3d=file={shlex.quote(lut_path.as_posix())}"
        cmd = (
            f"ffmpeg -y -hide_banner -loglevel error "
            f"-i {shlex.quote(inp.as_posix())} "
            f"-vf {shlex.quote(vf)} "
            f"{shlex.quote(out.as_posix())}"
        )
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr)
        return ensure_uint8(read_bgr(out))


# ── Leica look ────────────────────────────────────────────────────────────────

def _micro_contrast(img: np.ndarray, amount: float = 1.10) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), 1.2)
    out = cv2.addWeighted(img, amount, blur, -(amount - 1.0), 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def leica_style(img: np.ndarray) -> np.ndarray:
    """
    Leica M-series look:
     - S-curve mid-tone contrast (3D pop)
     - Subtle green desaturation (natural foliage)
     - Warm shadow cast (red/yellow lift)
     - Micro-contrast sharpening
    """
    img8 = ensure_uint8(img)

    # S-curve on L channel (LAB)
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    s = 1.0 / (1.0 + np.exp(-8.0 * (x - 0.5)))
    s = np.clip((s - 0.5) * 1.10 + 0.5, 0.0, 1.0)
    lut = np.clip(s * 255.0, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img8, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.LUT(lab[..., 0], lut)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Green desaturation
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    greens = (hsv[..., 0] >= 35) & (hsv[..., 0] <= 90)
    hsv[..., 1][greens] *= 0.95
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Warm shadow cast
    luma = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    shadow = np.clip((0.55 - luma) / 0.55, 0.0, 1.0)[..., None]
    warm = np.zeros_like(out, dtype=np.float32)
    warm[..., 2] = 8.0   # red
    warm[..., 1] = 4.0   # green
    out = np.clip(out.astype(np.float32) + warm * shadow, 0, 255).astype(np.uint8)

    return _micro_contrast(out, 1.10)


# ── Pixel look ────────────────────────────────────────────────────────────────

def pixel_style(img: np.ndarray) -> np.ndarray:
    """
    Google Pixel look:
     - Neutral greens, boosted blues
     - Highlight roll-off (protect against blow-out)
     - Cool shadow cast
    """
    img8 = ensure_uint8(img)

    hsv = cv2.cvtColor(img8, cv2.COLOR_BGR2HSV).astype(np.float32)
    greens = (hsv[..., 0] >= 35) & (hsv[..., 0] <= 90)
    blues  = (hsv[..., 0] >= 90) & (hsv[..., 0] <= 130)
    hsv[..., 1][greens] *= 0.90
    hsv[..., 1][blues]  *= 1.08
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Highlight roll-off in LAB
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
    hi = np.clip((lab[..., 0] - 180.0) / 75.0, 0.0, 1.0)
    lab[..., 0] = np.clip(lab[..., 0] - 15.0 * hi, 0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # Cool shadow cast
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float32)
    shadow = np.clip((140.0 - gray) / 140.0, 0.0, 1.0)[..., None]
    cool = np.zeros_like(out, dtype=np.float32)
    cool[..., 0] = 8.0   # blue
    out = np.clip(out.astype(np.float32) + cool * shadow, 0, 255).astype(np.uint8)

    return out


# ── Dispatcher ────────────────────────────────────────────────────────────────

def color_stage(img: np.ndarray, look: Look) -> np.ndarray:
    """
    Apply color grading.
    Blend mode: Leica and Pixel are applied to the ORIGINAL independently,
    then alpha-composited — NOT Leica-on-top-of-Pixel.
    """
    img8 = ensure_uint8(img)

    if look == Look.PIXEL:
        return pixel_style(img8)

    if look == Look.LEICA:
        src = img8.copy()
        if CONFIG.leica_lut_path:
            src = apply_lut_ffmpeg(src, CONFIG.leica_lut_path)
        return leica_style(src)

    # BLEND: both from the same img8 original
    pixel = pixel_style(img8)

    leica_src = img8.copy()
    if CONFIG.leica_lut_path:
        leica_src = apply_lut_ffmpeg(leica_src, CONFIG.leica_lut_path)
    leica = leica_style(leica_src)

    return cv2.addWeighted(leica, CONFIG.leica_blend_alpha, pixel, 1.0 - CONFIG.leica_blend_alpha, 0)
