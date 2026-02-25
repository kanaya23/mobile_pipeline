"""Shared utilities: dtype conversion, image IO, JSON helpers, subprocess template runner."""
from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .config import STORAGE


# ── Time ────────────────────────────────────────────────────────────────────

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Dtype helpers ────────────────────────────────────────────────────────────

def to_float01(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 65535.0
    if np.issubdtype(img.dtype, np.floating):
        return np.clip(img.astype(np.float32), 0.0, 1.0)
    raise TypeError(f"Unsupported dtype: {img.dtype}")


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(to_float01(img) * 255.0, 0, 255).astype(np.uint8)


def ensure_uint16(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint16:
        return img
    return np.clip(to_float01(img) * 65535.0, 0, 65535).astype(np.uint16)


# ── Image IO ─────────────────────────────────────────────────────────────────

def read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to load image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def write_bgr(path: Path, img: np.ndarray, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    if ext in {".jpg", ".jpeg"}:
        enc_img = ensure_uint8(img)
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    else:
        enc_img = ensure_uint16(img) if np.issubdtype(img.dtype, np.floating) else img
        if enc_img.dtype not in (np.uint8, np.uint16):
            enc_img = ensure_uint8(enc_img)
        params = []
    ok, buf = cv2.imencode(ext, enc_img, params)
    if not ok:
        raise RuntimeError(f"Encode failed for {path}")
    path.write_bytes(buf.tobytes())


def thumbnail(img: np.ndarray, width: int) -> np.ndarray:
    preview = ensure_uint8(img)
    h, w = preview.shape[:2]
    if w <= width:
        return preview.copy()
    scale = width / float(w)
    return cv2.resize(preview, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


# ── Zoom crop ─────────────────────────────────────────────────────────────────

def parse_zoom_crop(text: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not text:
        return None
    vals = [float(x.strip()) for x in text.split(",")]
    if len(vals) != 4:
        raise ValueError("zoom_crop must be x,y,w,h")
    x, y, w, h = vals
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1):
        raise ValueError("zoom_crop values must be normalised to [0,1]")
    return x, y, w, h


def apply_zoom_crop(img: np.ndarray, crop: Optional[Tuple[float, float, float, float]]) -> np.ndarray:
    if crop is None:
        return img
    x, y, w, h = crop
    ih, iw = img.shape[:2]
    x0, y0 = int(iw * x), int(ih * y)
    x1, y1 = int(iw * min(1.0, x + w)), int(ih * min(1.0, y + h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Invalid crop bounds")
    return img[y0:y1, x0:x1].copy()


# ── External model runner ─────────────────────────────────────────────────────

def run_template(template: str, img: np.ndarray, stage: str, output_name: str = "output.png") -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        inp = td / "input.png"
        out = td / output_name
        write_bgr(inp, img)
        cmd = template.format(input=inp.as_posix(), output=out.as_posix(), workdir=td.as_posix())
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"{stage} failed.\n{proc.stderr}")
        if not out.exists():
            raise RuntimeError(f"{stage} output missing: {out}")
        return read_bgr(out)


# ── JSON helpers ──────────────────────────────────────────────────────────────

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def metadata_path(job_id: str) -> Path:
    return STORAGE["metadata"] / f"{job_id}.json"
