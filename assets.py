"""Model asset registry: download URLs, paths, and setup helper."""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from .config import CONFIG, STORAGE


# ── Registry ─────────────────────────────────────────────────────────────────
#
#  required=True  → downloaded by setup_model_assets() always
#  required=False → downloaded only when download_optional=True
#
#  DiffBIR and StableSR are very large (4-8 GB each). On Kaggle, load them
#  as a Kaggle Dataset instead of downloading at runtime — set the path
#  manually and leave url=None.
#
MODEL_ASSETS: Dict[str, Dict[str, Any]] = {
    "realesr-general-x4v3": {
        "path":        STORAGE["models"] / "realesr-general-x4v3.pth",
        "url":         "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "required":    True,
        "size_note":   "~67 MB",
        "description": "Real-ESRGAN Fast Mode checkpoint",
    },
    "nafnet": {
        "path":        STORAGE["models"] / "NAFNet-SIDD-width64.pth",
        "url":         "https://huggingface.co/nickliqman/NAFNet/resolve/main/NAFNet-SIDD-width64.pth",
        "required":    False,
        "size_note":   "~300 MB",
        "description": "NAFNet denoising (SIDD, width-64)",
    },
    "restormer": {
        "path":        STORAGE["models"] / "restormer_denoising.pth",
        "url":         "https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth",
        "required":    False,
        "size_note":   "~175 MB",
        "description": "Restormer real-world denoising checkpoint",
    },
    "sam2": {
        "path":        STORAGE["models"] / "sam2.1_hiera_small.pt",
        "url":         "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "required":    False,
        "size_note":   "~185 MB",
        "description": "SAM 2.1 Hiera Small — portrait segmentation",
    },
    "depth-anything-v2": {
        "path":        STORAGE["models"] / "depth_anything_v2_vits.pth",
        "url":         "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
        "required":    False,
        "size_note":   "~100 MB",
        "description": "Depth Anything V2 Small — monocular depth",
    },
    # ── Large diffusion models — load from Kaggle Dataset, not downloaded ──
    "diffbir": {
        "path":        STORAGE["models"] / "diffbir_general_swinir_v1.ckpt",
        "url":         None,   # ~5 GB — add Kaggle Dataset + set path manually
        "required":    False,
        "size_note":   "~5 GB  ← load as Kaggle Dataset",
        "description": "DiffBIR Quality Mode checkpoint",
    },
    "stablesr": {
        "path":        STORAGE["models"] / "stablesr_768v_000139.ckpt",
        "url":         None,   # ~8 GB — add Kaggle Dataset + set path manually
        "required":    False,
        "size_note":   "~8 GB  ← load as Kaggle Dataset",
        "description": "StableSR Quality Mode checkpoint",
    },
}


def set_model_url(name: str, url: str) -> None:
    """Override the download URL for a model (e.g. a mirror)."""
    if name not in MODEL_ASSETS:
        raise KeyError(f"Unknown model: {name!r}  — valid: {list(MODEL_ASSETS)}")
    MODEL_ASSETS[name]["url"] = url


def _download(url: str, dst: Path, overwrite: bool = False) -> None:
    if dst.exists() and not overwrite:
        return
    if not url:
        raise RuntimeError(f"No URL configured for {dst.name}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {dst.name} ...")
    urllib.request.urlretrieve(url, str(dst))
    print(f"  ✓ {dst.name}")


def setup_model_assets(
    download_optional: bool = True,
    overwrite: bool = False,
) -> Dict[str, Dict[str, str]]:
    """
    Download model checkpoints.

    Args:
        download_optional: If True, also download optional models (NAFNet,
                           Restormer, SAM2, Depth Anything V2).
                           DiffBIR and StableSR are never auto-downloaded
                           (too large — load as Kaggle Datasets).
        overwrite:         Re-download even if file already exists.

    Returns:
        Per-model status report dict.
    """
    report: Dict[str, Dict[str, str]] = {}

    for name, item in MODEL_ASSETS.items():
        path: Path = item["path"]
        url: Optional[str] = item["url"]
        required: bool = bool(item["required"])
        size_note: str = item.get("size_note", "")

        should_download = required or download_optional

        # Never auto-download models with no URL (too large, must use Dataset)
        if url is None:
            report[name] = {
                "status": "manual",
                "path":   str(path),
                "reason": f"No URL — {size_note}. Add as Kaggle Dataset and set path manually.",
            }
            continue

        if not should_download:
            report[name] = {"status": "skipped", "path": str(path), "reason": "optional"}
            continue

        if path.exists() and not overwrite:
            report[name] = {"status": "ready", "path": str(path), "reason": "already exists"}
            continue

        try:
            _download(url, path, overwrite=overwrite)
            report[name] = {"status": "downloaded", "path": str(path), "size": size_note}
        except Exception as exc:
            report[name] = {"status": "failed", "path": str(path), "reason": str(exc)}
            if required and CONFIG.strict_components:
                raise RuntimeError(f"Required model download failed: {name}") from exc

    # Keep fast SR config in sync with registry path
    CONFIG.fast_sr_weights_path = MODEL_ASSETS["realesr-general-x4v3"]["path"]

    return report
