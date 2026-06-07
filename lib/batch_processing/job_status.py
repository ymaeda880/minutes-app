# -*- coding: utf-8 -*-
# minutes_app/lib/batch_processing/job_status.py
# ============================================================
# job.json status 更新
# ============================================================

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .io_utils import write_json


# ============================================================
# status update
# ============================================================
def update_job_status(job_root: Path, key: str, value: str) -> None:
    p = job_root / "job.json"
    if not p.exists():
        return

    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return

    status = obj.get("status")
    if not isinstance(status, dict):
        status = {}

    status[key] = value
    obj["status"] = status
    obj["updated_at"] = datetime.now().isoformat(timespec="seconds")

    write_json(p, obj)