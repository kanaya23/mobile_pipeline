"""Central configuration: enums, PipelineConfig dataclass, storage directories."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class Mode(str, Enum):
    FAST = "fast"
    QUALITY = "quality"


class Look(str, Enum):
    LEICA = "leica"
    PIXEL = "pixel"
    BLEND = "blend"


@dataclass
class PipelineConfig:
    base_dir: Path

    # Pipeline behaviour
    strict_components: bool = False
    use_hdr_fusion: bool = True
    leica_blend_alpha: float = 0.65
    leica_lut_path: Optional[Path] = None
    thumbnail_width: int = 768
    max_bokeh_kernel: int = 31

    # External model command templates (set these to enable heavy models)
    denoise_command_template: Optional[str] = None
    quality_sr_command_template: Optional[str] = None
    sam2_mask_command_template: Optional[str] = None
    depth_map_command_template: Optional[str] = None

    # Fast SR weights
    fast_sr_weights_path: Optional[Path] = None
    fast_sr_weights_url: str = (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
    )

    # Celery / Redis (optional)
    enable_celery: bool = False
    redis_url: str = "redis://127.0.0.1:6379/0"

    # Snapshot paths
    snapshot_export_path: Optional[Path] = None
    snapshot_import_path: Optional[Path] = None


# ── Singleton + storage dirs ────────────────────────────────────────────────

_base = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
CONFIG = PipelineConfig(base_dir=_base)

ROOT = CONFIG.base_dir / "storage"
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
