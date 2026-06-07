# -*- coding: utf-8 -*-
# minutes_app/lib/batch_processing/paths.py
# ============================================================
# 一括処理 path / job 作成
# ============================================================

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .io_utils import safe_mkdir


# ============================================================
# filename / job id
# ============================================================
def now_job_id() -> str:
    return "job_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_filename(name: str) -> str:
    s = (name or "audio").strip()
    s = s.replace("\\", "_").replace("/", "_").replace(":", "_")
    s = re.sub(r"\s+", "_", s)
    return s or "audio"


# ============================================================
# job root
# ============================================================
def create_job_root_for_uploaded_file(
    *,
    user_root: Path,
    uploaded_name: str,
) -> tuple[str, Path, Path, Path, Path, Path, Path, Path]:
    today_dir = datetime.now().strftime("%Y-%m-%d")
    job_id = now_job_id()
    job_root = user_root / today_dir / job_id

    original_dir = job_root / "original"
    split_dir = job_root / "split"
    transcript_dir = job_root / "transcript"
    transcript_combined_dir = job_root / "transcript_combined"
    speaker_sep_dir = job_root / "transcript_speaker_separated"
    speaker_combined_dir = job_root / "transcript_speaker_separated_combined"
    marked_dir = job_root / "transcript_marked"
    logs_dir = job_root / "logs"

    for d in (
        original_dir,
        split_dir,
        transcript_dir,
        transcript_combined_dir,
        speaker_sep_dir,
        speaker_combined_dir,
        marked_dir,
        logs_dir,
    ):
        safe_mkdir(d)

    return (
        job_id,
        job_root,
        original_dir,
        split_dir,
        transcript_dir,
        transcript_combined_dir,
        speaker_sep_dir,
        speaker_combined_dir,
    )