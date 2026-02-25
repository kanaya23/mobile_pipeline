"""Gallery persistence: zip snapshot export/import for Kaggle session restarts."""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Optional

from .config import CONFIG, ROOT, STORAGE, JOB_STATUS
from .utils import read_json


def export_snapshot(dest: Optional[Path] = None) -> Path:
    """
    Zip the entire storage folder.

    Usage:
        from mobile_pipeline.snapshot import export_snapshot
        export_snapshot()       # → base_dir/storage_snapshot.zip
    Then download from Kaggle output tab, or save as a Kaggle Dataset named
    "mobile-pipeline-storage" to auto-restore on next session.
    """
    out = dest or CONFIG.snapshot_export_path
    if out is None:
        raise RuntimeError("Set CONFIG.snapshot_export_path first")
    out.parent.mkdir(parents=True, exist_ok=True)

    file_count = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in ROOT.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(ROOT))
                file_count += 1

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"✓ Snapshot saved → {out}  ({size_mb:.1f} MB, {file_count} files)")
    return out


def import_snapshot(src: Optional[Path] = None) -> bool:
    """
    Extract a snapshot ZIP back into storage and rebuild in-memory job status.

    Usage:
        from mobile_pipeline.snapshot import import_snapshot
        import_snapshot()       # uses CONFIG.snapshot_import_path
        import_snapshot(Path('/kaggle/input/mobile-pipeline-storage/storage_snapshot.zip'))
    """
    path = src or CONFIG.snapshot_import_path
    if path is None or not path.exists():
        print(f"Snapshot not found: {path}")
        return False

    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(ROOT)

    restored = 0
    for mf in STORAGE["metadata"].glob("*.json"):
        try:
            meta = read_json(mf)
            if "status" in meta and "job_id" in meta:
                JOB_STATUS[meta["job_id"]] = meta["status"]
                restored += 1
        except Exception:
            pass

    print(f"✓ Snapshot restored from {path}  ({restored} job(s) reloaded into memory)")
    return True


def auto_import() -> None:
    """Called at startup — imports snapshot if one exists at a known location."""
    default = Path("/kaggle/input/mobile-pipeline-storage/storage_snapshot.zip")
    local   = CONFIG.base_dir / "storage_snapshot.zip"

    if default.exists():
        CONFIG.snapshot_import_path = default
        import_snapshot()
    elif local.exists():
        CONFIG.snapshot_import_path = local
        import_snapshot()
