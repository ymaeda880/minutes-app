# -*- coding: utf-8 -*-
# minutes_app/lib/batch_processing/io_utils.py
# ============================================================
# 一括処理 I/O ユーティリティ
# ============================================================

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


# ============================================================
# directory
# ============================================================
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ============================================================
# text / json
# ============================================================
def write_text(p: Path, s: str) -> None:
    safe_mkdir(p.parent)
    p.write_text(s, encoding="utf-8")


def write_json(p: Path, obj: Any) -> None:
    safe_mkdir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_text_safely(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="cp932", errors="replace")


# ============================================================
# log
# ============================================================
def append_log(log_path: Path, msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_mkdir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")