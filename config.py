"""Central configuration: enums, PipelineConfig dataclass, storage directories."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class Mode(str, Enum):
    FAST    = "fast"
    QUALITY = "quality"


class Look(str, Enum):
    LEICA = "leica"
    PIXEL = "pixel"
    BLEND = "blend"


@dataclass
class PipelineConfig:
    base_dir: Path

    # Pipeline behaviour
    strict_components:   bool          = False
    use_hdr_fusion:      bool          = True
    leica_blend_alpha:   float         = 0.65
    leica_lut_path:      Optional[Path] = None
    thumbnail_width:     int           = 768
    max_bokeh_kernel:    int           = 31

    # External model command templates (set these to enable heavy models)
    denoise_command_template:    Optional[str] = None
    quality_sr_command_template: Optional[str] = None
    sam2_mask_command_template:  Optional[str] = None
    depth_map_command_template:  Optional[str] = None

    # Fast SR weights
    fast_sr_weights_path: Optional[Path] = None
    fast_sr_weights_url:  str            = (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
    )

    # Celery / Redis (optional)
    enable_celery: bool = False
    redis_url:     str  = "redis://127.0.0.1:6379/0"

    # Snapshot paths
    snapshot_export_path: Optional[Path] = None
    snapshot_import_path: Optional[Path] = None


# ── Singleton + storage dirs ─────────────────────────────────────────────────

_base  = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
CONFIG = PipelineConfig(base_dir=_base)

ROOT    = CONFIG.base_dir / "storage"
STORAGE: Dict[str, Path] = {
    "uploads":    ROOT / "uploads",
    "processed":  ROOT / "processed",
    "thumbnails": ROOT / "thumbnails",
    "metadata":   ROOT / "metadata",
    "models":     ROOT / "models",
}
for _d in STORAGE.values():
    _d.mkdir(parents=True, exist_ok=True)

if CONFIG.fast_sr_weights_path is None:
    CONFIG.fast_sr_weights_path = STORAGE["models"] / "realesr-general-x4v3.pth"

if CONFIG.snapshot_export_path is None:
    CONFIG.snapshot_export_path = CONFIG.base_dir / "storage_snapshot.zip"

# In-memory job-status cache
JOB_STATUS: Dict[str, Dict[str, Any]] = {}


# ── Auto-configure helper ─────────────────────────────────────────────────────

def auto_configure_from_models(
    tmp_models_dir: Optional[Path] = None,
    diffbir_repo: Optional[str] = None,
    nafnet_repo: Optional[str] = None,
    restormer_repo: Optional[str] = None,
    sam2_repo: Optional[str] = None,
    depth_repo: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    Scan disk for downloaded model weights and inference scripts,
    then auto-set CONFIG command templates so heavy models are actually used.

    Call this AFTER setup_model_assets() in your notebook setup cell.

    Args:
        tmp_models_dir:  Where large models live (default: /kaggle/tmp/models)
        diffbir_repo:    Path to DiffBIR repo clone (default: auto-detect)
        nafnet_repo:     Path to NAFNet repo clone  (default: auto-detect)
        restormer_repo:  Path to Restormer repo     (default: auto-detect)
        sam2_repo:       Path to sam2 repo          (default: auto-detect)
        depth_repo:      Path to Depth-Anything-V2  (default: auto-detect)
    """
    _on_kaggle = Path("/kaggle/working").exists()
    _tmp       = tmp_models_dir or (
        Path("/kaggle/tmp/models") if _on_kaggle else STORAGE["models"] / "large"
    )
    _wdir      = Path("/kaggle/working") if _on_kaggle else CONFIG.base_dir

    def _find(candidates: list) -> Optional[Path]:
        for c in candidates:
            p = Path(c)
            if p.exists():
                return p
        return None

    # ── Denoiser ─────────────────────────────────────────────────────────────
    if not CONFIG.denoise_command_template:
        nafnet_w   = STORAGE["models"] / "NAFNet-SIDD-width64.pth"
        restormer_w = STORAGE["models"] / "restormer_denoising.pth"

        nafnet_script = _find([
            nafnet_repo or "",
            str(_wdir / "NAFNet" / "predict.py"),
            str(_wdir / "NAFNet" / "basicsr" / "test.py"),
        ])
        restormer_script = _find([
            restormer_repo or "",
            str(_wdir / "Restormer" / "demo.py"),
        ])

        if nafnet_w.exists() and nafnet_script:
            CONFIG.denoise_command_template = (
                f"python {nafnet_script} "
                "--input {input} --output {output} "
                f"--model_path {nafnet_w}"
            )
            if verbose: print(f"✓ Denoise  → NAFNet  ({nafnet_script})")
        elif restormer_w.exists() and restormer_script:
            CONFIG.denoise_command_template = (
                f"python {restormer_script} "
                "--task Real_Denoising "
                "--input_dir {workdir} --result_dir {workdir} "
                f"--pretrained_weights {restormer_w}"
            )
            if verbose: print(f"✓ Denoise  → Restormer ({restormer_script})")
        elif verbose:
            print("⚠ Denoise  → bilateral fallback (NAFNet/Restormer script not found)")

    # ── Quality SR ───────────────────────────────────────────────────────────
    if not CONFIG.quality_sr_command_template:
        diffbir_swinir = _tmp / "general_swinir_v1.ckpt"
        diffbir_full   = _tmp / "general_full_v1.ckpt"
        stablesr_ckpt  = _tmp / "webui_768v_139.ckpt"

        diffbir_script = _find([
            diffbir_repo or "",
            str(_wdir / "DiffBIR" / "inference.py"),
            str(_wdir / "DiffBIR" / "run_pipeline.py"),
        ])

        if diffbir_full.exists() and diffbir_swinir.exists() and diffbir_script:
            CONFIG.quality_sr_command_template = (
                f"python {diffbir_script} "
                "--input {input} --output {output} "
                f"--ckpt_swinir {diffbir_swinir} "
                f"--ckpt {diffbir_full} "
                "--sr_scale 4 --device cuda"
            )
            if verbose: print(f"✓ QualitySR → DiffBIR ({diffbir_script})")
        elif stablesr_ckpt.exists() and verbose:
            print("⚠ QualitySR → StableSR found but needs separate wiring (see README)")
        elif verbose:
            print("⚠ QualitySR → Real-ESRGAN fallback (DiffBIR not found)")

    # ── Portrait: SAM2 ───────────────────────────────────────────────────────
    if not CONFIG.sam2_mask_command_template:
        sam2_w      = STORAGE["models"] / "sam2.1_hiera_small.pt"
        sam2_script = _find([
            sam2_repo or "",
            str(_wdir / "sam2" / "scripts" / "run_segment.py"),
            str(_wdir / "segment-anything-2" / "scripts" / "run_segment.py"),
        ])
        if sam2_w.exists() and sam2_script:
            CONFIG.sam2_mask_command_template = (
                f"python {sam2_script} "
                "--input {input} --output {output} "
                f"--checkpoint {sam2_w} --model-cfg sam2.1_hiera_s.yaml"
            )
            if verbose: print(f"✓ Portrait → SAM2 ({sam2_script})")
        elif verbose:
            print("⚠ Portrait → GrabCut fallback (SAM2 script not found)")

    # ── Portrait: Depth Anything V2 ──────────────────────────────────────────
    if not CONFIG.depth_map_command_template:
        depth_w      = STORAGE["models"] / "depth_anything_v2_vits.pth"
        depth_script = _find([
            depth_repo or "",
            str(_wdir / "Depth-Anything-V2" / "run.py"),
            str(_wdir / "depth-anything-v2" / "run.py"),
        ])
        if depth_w.exists() and depth_script:
            CONFIG.depth_map_command_template = (
                f"python {depth_script} "
                "--encoder vits "
                "--img-path {input} --outdir {workdir} "
                f"--pretrained-from {depth_w}"
            )
            if verbose: print(f"✓ Depth    → DepthAnythingV2 ({depth_script})")
        elif verbose:
            print("⚠ Depth    → distance-transform fallback (script not found)")

    if verbose:
        print("\n── GPU Status ───────────────────────────────────────")
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
                print(f"  CUDA ✓  {name}  ({vram} MB VRAM)")
            else:
                print("  CUDA ✗  No GPU detected — all stages run on CPU (slow)")
                print("  → Kaggle: Settings ▸ Accelerator ▸ GPU T4 x2")
        except ImportError:
            print("  torch not installed — pip install torch")

