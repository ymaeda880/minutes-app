# -*- coding: utf-8 -*-
# minutes_app/lib/batch_processing/overlap_step.py
# ============================================================
# 一括処理：重複検出ステップ
# ============================================================

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .constants import (
    BEGIN_TAG,
    DEFAULT_MIN_MATCH_SIZE,
    DEFAULT_USE_AUTOJUNK,
    HEAD_CHARS,
    HEAD_SENTENCES,
    HEAD_SHIFT_TRIES,
    OVERLAP_CHARS,
)
from .io_utils import append_log, read_text_safely, safe_mkdir, write_json, write_text
from .overlap_utils import build_merged_text


# ============================================================
# run overlap step
# ============================================================
def run_overlap_step(
    *,
    job_root: Path,
    combined_path: Path,
    log_path: Path,
    current_user: str,
) -> Path:
    marked_dir = job_root / "transcript_marked"
    safe_mkdir(marked_dir)

    text = read_text_safely(combined_path)

    merged, logs = build_merged_text(
        text,
        int(DEFAULT_MIN_MATCH_SIZE),
        bool(DEFAULT_USE_AUTOJUNK),
    )

    out_txt = marked_dir / f"{combined_path.stem}_marked.txt"
    out_log = marked_dir / f"{combined_path.stem}_detect_log.json"

    write_text(out_txt, merged)

    write_json(
        out_log,
        {
            "input": str(combined_path),
            "output_text": str(out_txt),
            "output_log": str(out_log),
            "params": {
                "OVERLAP_CHARS": OVERLAP_CHARS,
                "HEAD_CHARS": HEAD_CHARS,
                "HEAD_SENTENCES": HEAD_SENTENCES,
                "HEAD_SHIFT_TRIES": HEAD_SHIFT_TRIES,
                "min_match_size": int(DEFAULT_MIN_MATCH_SIZE),
                "autojunk": bool(DEFAULT_USE_AUTOJUNK),
                "BEGIN_TAG": BEGIN_TAG,
            },
            "counts": {
                "markers": len(logs),
                "has_markers": bool(logs),
            },
            "logs": logs,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "user": str(current_user),
            "job_dir": str(job_root),
        },
    )

    append_log(log_path, "OVERLAP DETECT START")
    append_log(log_path, f"input={combined_path.name}")
    append_log(log_path, f"output={out_txt.name}")
    append_log(log_path, f"log={out_log.name}")
    append_log(log_path, f"min_match_size={DEFAULT_MIN_MATCH_SIZE} autojunk={DEFAULT_USE_AUTOJUNK}")
    append_log(log_path, "OVERLAP DETECT DONE")

    return out_txt