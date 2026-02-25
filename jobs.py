"""Job management: request dataclass, process pipeline, queue, gallery helpers."""
from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import CONFIG, STORAGE, JOB_STATUS, Mode, Look
from .utils import (
    utc_now, parse_zoom_crop, apply_zoom_crop,
    write_bgr, thumbnail, read_json, write_json, metadata_path
)
from .raw_hdr import load_primary
from .denoise import denoise_stage
from .sr import sr_stage
from .color import color_stage
from .portrait import portrait_stage


# ── Status helpers ────────────────────────────────────────────────────────────

def set_status(
    job_id: str,
    stage: str,
    percent: int,
    status: str = "processing",
    detail: Optional[str] = None,
) -> None:
    payload: Dict[str, Any] = {
        "job_id":     job_id,
        "status":     status,
        "stage":      stage,
        "percent":    int(percent),
        "detail":     detail,
        "updated_at": utc_now(),
    }
    JOB_STATUS[job_id] = payload
    mf = metadata_path(job_id)
    if mf.exists():
        meta = read_json(mf)
        meta["status"] = payload
        write_json(mf, meta)


# ── JobRequest ────────────────────────────────────────────────────────────────

@dataclass
class JobRequest:
    mode:      Mode
    look:      Look
    portrait:  bool = False
    zoom_crop: Optional[Tuple[float, float, float, float]] = None

    def to_payload(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mode"] = self.mode.value
        d["look"] = self.look.value
        return d

    @staticmethod
    def from_payload(p: Dict[str, Any]) -> "JobRequest":
        z = tuple(p["zoom_crop"]) if p.get("zoom_crop") else None
        return JobRequest(
            mode=Mode(p["mode"]),
            look=Look(p["look"]),
            portrait=bool(p.get("portrait", False)),
            zoom_crop=z,
        )


# ── Metadata helpers ──────────────────────────────────────────────────────────

def init_meta(job_id: str, original: Path, req: JobRequest) -> None:
    meta: Dict[str, Any] = {
        "job_id":     job_id,
        "created_at": utc_now(),
        "request":    req.to_payload(),
        "files":      {"original": original.as_posix(), "processed": None, "thumbnail": None},
        "status":     {"job_id": job_id, "status": "queued", "stage": "queued",
                       "percent": 0, "detail": None, "updated_at": utc_now()},
        "error":      None,
    }
    write_json(metadata_path(job_id), meta)
    JOB_STATUS[job_id] = meta["status"]


def finalize_meta(job_id: str, processed: Path, thumb: Path) -> Dict[str, Any]:
    meta = read_json(metadata_path(job_id))
    meta["files"]["processed"] = processed.as_posix()
    meta["files"]["thumbnail"] = thumb.as_posix()
    meta["completed_at"] = utc_now()
    meta["status"] = {
        "job_id": job_id, "status": "done", "stage": "completed",
        "percent": 100, "detail": None, "updated_at": utc_now(),
    }
    write_json(metadata_path(job_id), meta)
    JOB_STATUS[job_id] = meta["status"]
    return meta


def fail_meta(job_id: str, exc: Exception) -> None:
    pct = JOB_STATUS.get(job_id, {}).get("percent", 0)
    set_status(job_id, "failed", pct, status="failed", detail=str(exc))
    mf = metadata_path(job_id)
    if mf.exists():
        meta = read_json(mf)
        meta["error"]  = str(exc)
        meta["status"] = JOB_STATUS[job_id]
        write_json(mf, meta)


# ── Core processing ───────────────────────────────────────────────────────────

def process_job(
    job_id: str,
    uploaded: Path,
    req: JobRequest,
    burst_paths: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    try:
        set_status(job_id, "developing_raw",   10)
        img = load_primary(uploaded, burst_paths)
        img = apply_zoom_crop(img, req.zoom_crop)

        set_status(job_id, "reducing_noise",   35)
        img = denoise_stage(img)

        set_status(job_id, "enhancing_detail", 65)
        img = sr_stage(img, req.mode)

        set_status(job_id, "applying_color",   82)
        img = color_stage(img, req.look)

        if req.portrait:
            set_status(job_id, "applying_portrait", 94)
            img = portrait_stage(img)

        out = STORAGE["processed"]  / f"{job_id}.jpg"
        th  = STORAGE["thumbnails"] / f"{job_id}.jpg"
        write_bgr(out, img, quality=96)
        write_bgr(th,  thumbnail(img, CONFIG.thumbnail_width), quality=90)
        return finalize_meta(job_id, out, th)

    except Exception as exc:
        fail_meta(job_id, exc)
        raise


# ── Queue ─────────────────────────────────────────────────────────────────────

def queue_job(
    input_path: Path,
    req: JobRequest,
    burst_paths: Optional[List[Path]] = None,
    run_sync: bool = True,
) -> Dict[str, Any]:
    """
    Copy inputs to persistent storage, create metadata, then either:
      run_sync=True  — process immediately (blocks; good for notebook testing)
      run_sync=False — return immediately so the caller can schedule as
                       a background task (used by FastAPI endpoints)
    """
    job_id = str(uuid.uuid4())
    saved  = STORAGE["uploads"] / f"{job_id}{input_path.suffix.lower()}"
    shutil.copy2(input_path, saved)

    bursts: List[Path] = []
    for i, p in enumerate(burst_paths or []):
        dst = STORAGE["uploads"] / f"{job_id}_burst_{i}{p.suffix.lower()}"
        shutil.copy2(p, dst)
        bursts.append(dst)

    init_meta(job_id, saved, req)

    # ── Celery path ───────────────────────────────────────────────────────
    if CONFIG.enable_celery:
        from .celery_worker import celery_process_job, CELERY_APP
        if CELERY_APP is None:
            raise RuntimeError("Celery enabled but app not initialised")
        t = celery_process_job.delay(
            job_id, saved.as_posix(), req.to_payload(),
            [x.as_posix() for x in bursts],
        )
        set_status(job_id, "queued", 2, status="queued", detail=f"task_id={t.id}")
        return {"job_id": job_id, "task_id": t.id, "status": JOB_STATUS[job_id]}

    # ── Sync path (notebook testing / direct calls) ───────────────────────
    if run_sync:
        meta = process_job(job_id, saved, req, bursts)
        return {"job_id": job_id, "status": meta["status"]}

    # ── Async path (FastAPI BackgroundTask) ───────────────────────────────
    set_status(job_id, "queued", 2, status="queued", detail="background_task")
    return {
        "job_id":       job_id,
        "status":       JOB_STATUS[job_id],
        "saved_input":  saved.as_posix(),
        "saved_bursts": [x.as_posix() for x in bursts],
    }


# ── Gallery ───────────────────────────────────────────────────────────────────

def gallery_items() -> List[Dict[str, Any]]:
    items = []
    for f in STORAGE["metadata"].glob("*.json"):
        try:
            meta = read_json(f)
            if meta.get("files", {}).get("processed"):
                items.append(meta)
        except Exception:
            pass
    items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return items


def image_paths(job_id: str) -> Dict[str, Optional[Path]]:
    mf = metadata_path(job_id)
    if not mf.exists():
        raise FileNotFoundError(job_id)
    meta  = read_json(mf)
    files = meta["files"]
    return {
        "original":  Path(files["original"]),
        "processed": Path(files["processed"]) if files.get("processed") else None,
        "thumbnail": Path(files["thumbnail"]) if files.get("thumbnail") else None,
        "meta":      mf,
    }
