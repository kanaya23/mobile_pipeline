"""Model asset registry: download URLs, paths, and setup helper."""
from __future__ import annotations

import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from .config import CONFIG, STORAGE

# ── Storage dirs ──────────────────────────────────────────────────────────────
# Kaggle working dir has ~20 GB.  Large diffusion models (5-8 GB each) go to
# /kaggle/tmp which has ~100 GB but is NOT persisted across sessions.
_KAGGLE_TMP = Path("/kaggle/tmp")
_ON_KAGGLE   = Path("/kaggle/working").exists()

TMP_MODELS_DIR = (_KAGGLE_TMP / "models") if _ON_KAGGLE else (STORAGE["models"] / "large")
TMP_MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── Registry ──────────────────────────────────────────────────────────────────
#
#  required=True  → downloaded by setup_model_assets() always
#  required=False → downloaded only when download_optional=True
#  large=True     → saved to TMP_MODELS_DIR (/kaggle/tmp/models) instead of
#                   STORAGE["models"] (/kaggle/working/storage/models)
#
MODEL_ASSETS: Dict[str, Dict[str, Any]] = {
    # ── Always-required ───────────────────────────────────────────────────────
    "realesr-general-x4v3": {
        "path":        STORAGE["models"] / "realesr-general-x4v3.pth",
        "url":         "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "required":    True,
        "large":       False,
        "size_note":   "~67 MB",
        "description": "Real-ESRGAN Fast Mode checkpoint",
        "downloader":  "wget",
    },
    # ── Optional small/medium models (~100-465 MB, fit in /kaggle/working) ────
    "nafnet": {
        "path":        STORAGE["models"] / "NAFNet-SIDD-width64.pth",
        # Google Drive file ID — downloaded via gdown
        "url":         "14Fht1QQJ2gMlk4N1ERCRuElg8JfjrWWR",
        "required":    False,
        "large":       False,
        "size_note":   "~464 MB",
        "description": "NAFNet denoising (SIDD, width-64) — best denoising model",
        "downloader":  "gdown",
    },
    "restormer": {
        "path":        STORAGE["models"] / "restormer_denoising.pth",
        "url":         "https://github.com/swz30/Restormer/releases/download/v1.0/real_denoising.pth",
        "required":    False,
        "large":       False,
        "size_note":   "~175 MB",
        "description": "Restormer real-world denoising checkpoint",
        "downloader":  "wget",
    },
    "sam2": {
        "path":        STORAGE["models"] / "sam2.1_hiera_small.pt",
        "url":         "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "required":    False,
        "large":       False,
        "size_note":   "~185 MB",
        "description": "SAM 2.1 Hiera Small — portrait segmentation",
        "downloader":  "wget",
    },
    "depth-anything-v2": {
        "path":        STORAGE["models"] / "depth_anything_v2_vits.pth",
        "url":         "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth",
        "required":    False,
        "large":       False,
        "size_note":   "~100 MB",
        "description": "Depth Anything V2 Small — monocular depth",
        "downloader":  "wget",
    },
    # ── Large diffusion models (~5-8 GB) — routed to /kaggle/tmp/models/ ──────
    #    /kaggle/tmp has ~100 GB but is NOT persisted; re-download each session.
    "diffbir-swinir": {
        "path":        TMP_MODELS_DIR / "general_swinir_v1.ckpt",
        "url":         "https://huggingface.co/lxq007/DiffBIR/resolve/main/general_swinir_v1.ckpt",
        "required":    False,
        "large":       True,
        "size_note":   "~400 MB",
        "description": "DiffBIR SwinIR helper encoder",
        "downloader":  "wget",
    },
    "diffbir-full": {
        "path":        TMP_MODELS_DIR / "general_full_v1.ckpt",
        "url":         "https://huggingface.co/lxq007/DiffBIR/resolve/main/general_full_v1.ckpt",
        "required":    False,
        "large":       True,
        "size_note":   "~5.1 GB",
        "description": "DiffBIR full model — quality SR (very large)",
        "downloader":  "wget",
    },
    "stablesr": {
        "path":        TMP_MODELS_DIR / "webui_768v_139.ckpt",
        "url":         "https://huggingface.co/Iceclear/StableSR/resolve/main/webui_768v_139.ckpt",
        "required":    False,
        "large":       True,
        "size_note":   "~7.7 GB",
        "description": "StableSR Quality Mode checkpoint (very large)",
        "downloader":  "wget",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def set_model_url(name: str, url: str) -> None:
    """Override the download URL for a model (e.g. a mirror)."""
    if name not in MODEL_ASSETS:
        raise KeyError(f"Unknown model: {name!r}  — valid: {list(MODEL_ASSETS)}")
    MODEL_ASSETS[name]["url"] = url


def _has_tool(name: str) -> bool:
    return shutil.which(name) is not None


def _download_wget(url: str, dst: Path) -> None:
    """Download via wget with progress bar."""
    subprocess.run(
        ["wget", "--show-progress", "-q", str(url), "-O", str(dst)],
        check=True,
    )


def _download_gdown(file_id: str, dst: Path) -> None:
    """Download from Google Drive via gdown (installs gdown if missing)."""
    if not _has_tool("gdown"):
        print("   Installing gdown ...")
        subprocess.run(["pip", "install", "-q", "gdown"], check=True)
    subprocess.run(["gdown", "--fuzzy", file_id, "-O", str(dst)], check=True)


def _download_urllib(url: str, dst: Path) -> None:
    """Fallback: plain urllib download (no progress bar)."""
    print("   (wget not found — using urllib, no progress bar)")
    urllib.request.urlretrieve(url, str(dst))


def _download(
    name: str,
    url: str,
    dst: Path,
    downloader: str = "wget",
    overwrite: bool = False,
) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"   ⬇️  Downloading {dst.name} ...")
    try:
        if downloader == "gdown":
            _download_gdown(url, dst)
        elif downloader == "wget" and _has_tool("wget"):
            _download_wget(url, dst)
        else:
            _download_urllib(url, dst)
    except Exception:
        # Clean up zero-byte partial file so a retry works cleanly
        if dst.exists() and dst.stat().st_size == 0:
            dst.unlink(missing_ok=True)
        raise
    print(f"   ✓ Saved → {dst}")


# ── Public API ────────────────────────────────────────────────────────────────

def setup_model_assets(
    download_optional: bool = True,
    download_large: bool = True,
    overwrite: bool = False,
) -> Dict[str, Dict[str, str]]:
    """
    Download model checkpoints.

    Args:
        download_optional: If True, also download optional small/medium models
                           (NAFNet ~464 MB, Restormer ~175 MB, SAM2 ~185 MB,
                            Depth Anything V2 ~100 MB).
        download_large:    If True, also auto-download the large diffusion models
                           (DiffBIR ~5.5 GB total, StableSR ~7.7 GB).
                           These are saved to /kaggle/tmp/models/ to avoid
                           filling the 20 GB /kaggle/working quota.
                           NOTE: /kaggle/tmp is NOT persisted — re-download
                           each session.
        overwrite:         Re-download even if the file already exists.

    Returns:
        Per-model status report dict.
    """
    report: Dict[str, Dict[str, str]] = {}

    for name, item in MODEL_ASSETS.items():
        path: Path         = item["path"]
        url: Optional[str] = item.get("url")
        required: bool     = bool(item["required"])
        is_large: bool     = bool(item.get("large", False))
        size_note: str     = item.get("size_note", "")
        downloader: str    = item.get("downloader", "wget")

        # Decide whether to attempt this model
        if not (required or download_optional):
            report[name] = {"status": "skipped", "path": str(path), "reason": "optional"}
            continue
        if is_large and not download_large:
            report[name] = {
                "status": "skipped",
                "path":   str(path),
                "reason": f"large model — pass download_large=True to fetch ({size_note})",
            }
            continue
        if url is None:
            report[name] = {
                "status": "manual",
                "path":   str(path),
                "reason": "No URL configured — set manually via set_model_url()",
            }
            continue
        if path.exists() and not overwrite:
            report[name] = {"status": "ready", "path": str(path), "size": size_note}
            continue

        try:
            _download(name, url, path, downloader=downloader, overwrite=overwrite)
            report[name] = {"status": "downloaded", "path": str(path), "size": size_note}
        except Exception as exc:
            report[name] = {"status": "failed", "path": str(path), "reason": str(exc)}
            if required and CONFIG.strict_components:
                raise RuntimeError(f"Required model download failed: {name}") from exc

    # Keep fast SR config in sync with registry path
    CONFIG.fast_sr_weights_path = MODEL_ASSETS["realesr-general-x4v3"]["path"]

    return report


def print_download_summary(report: Dict[str, Dict[str, str]]) -> None:
    """Pretty-print the result of setup_model_assets()."""
    print("\n" + "=" * 65)
    print("🎉  MODEL DOWNLOAD SUMMARY")
    print("=" * 65)

    icons = {
        "ready":      "✓",
        "downloaded": "✓",
        "skipped":    "⏭",
        "manual":     "📋",
        "failed":     "✗",
    }
    for name, info in report.items():
        status = info["status"]
        icon   = icons.get(status, "?")
        size   = info.get("size", "")
        reason = info.get("reason", "")
        note   = size or reason
        print(f"  {icon}  {name:<28s}  {status:<12s}  {note}")

    failed = [n for n, i in report.items() if i["status"] == "failed"]
    if failed:
        print(f"\n  ⚠  Failed: {', '.join(failed)}")
        print("     → Check internet access and re-run setup_model_assets(overwrite=True)")

    print()
    print(f"  📁  Small/medium models → {STORAGE['models']}")
    print(f"  📁  Large models        → {TMP_MODELS_DIR}  (⚠ /kaggle/tmp is NOT persisted!)")
    print("=" * 65 + "\n")
