# -*- coding: utf-8 -*-
# minutes_app/lib/batch_processing/audio_split.py
# ============================================================
# 一括処理：音声分割
# ============================================================

from __future__ import annotations

import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydub import AudioSegment

from .constants import (
    AUDIO_EXTS,
    DEFAULT_ABSORB_TINY_TAIL,
    DEFAULT_CHUNK_MIN,
    DEFAULT_EXPORT_FMT,
    DEFAULT_FADE_MS,
    DEFAULT_OVERLAP_MIN,
    DEFAULT_TARGET_BITRATE,
)
from .io_utils import append_log, write_json
from .paths import safe_filename


# ============================================================
# format / time
# ============================================================
def hhmmss(ms: int) -> str:
    return str(timedelta(milliseconds=ms)).split(".")[0]


def _guess_format_from_suffix(suffix: str) -> str:
    suf = (suffix or "").lower()
    if suf == ".mp3":
        return "mp3"
    if suf == ".wav":
        return "wav"
    if suf in {".mp4", ".m4a"}:
        return "mp4"
    if suf == ".webm":
        return "webm"
    if suf == ".ogg":
        return "ogg"
    return suf.lstrip(".")


def guess_mime_from_suffix(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".mp3":
        return "audio/mpeg"
    if suf == ".wav":
        return "audio/wav"
    if suf == ".m4a":
        return "audio/mp4"
    if suf == ".mp4":
        return "video/mp4"
    if suf == ".webm":
        return "audio/webm"
    if suf == ".ogg":
        return "audio/ogg"
    return "application/octet-stream"


# ============================================================
# split
# ============================================================
def split_with_overlap(
    audio: AudioSegment,
    chunk_ms: int,
    overlap_ms: int,
    fade_ms: int,
    absorb_tiny_tail: bool,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    n = len(audio)

    if chunk_ms <= 0:
        raise ValueError("chunk_ms must be > 0")
    if overlap_ms < 0:
        raise ValueError("overlap_ms must be >= 0")
    if overlap_ms >= chunk_ms:
        raise ValueError("overlap_ms must be < chunk_ms")

    step = max(1, chunk_ms - overlap_ms)
    start = 0

    while start < n:
        end = min(start + chunk_ms, n)
        seg = audio[start:end]

        if absorb_tiny_tail and start > 0 and end == n:
            tail_len = end - start
            if tail_len < overlap_ms and results:
                prev = results[-1]
                prev["end_ms"] = n
                prev["segment"] = audio[prev["start_ms"] : n]
                break

        if fade_ms > 0 and len(seg) > fade_ms * 2:
            seg = seg.fade_in(fade_ms).fade_out(fade_ms)

        results.append(
            {
                "start_ms": start,
                "end_ms": end,
                "segment": seg,
            }
        )

        if end == n:
            break

        start += step

    return results


# ============================================================
# run audio split
# ============================================================
def run_audio_split_step(
    *,
    uploaded_name: str,
    uploaded_bytes: bytes,
    job_id: str,
    job_root: Path,
    original_dir: Path,
    split_dir: Path,
    log_path: Path,
    current_user: str,
    username_dir: str,
    transcript_model: str,
) -> list[Path]:
    suffix = Path(uploaded_name).suffix.lower()

    if suffix not in AUDIO_EXTS:
        raise ValueError("対応していない拡張子です。")

    load_fmt = _guess_format_from_suffix(suffix)
    audio = AudioSegment.from_file(io.BytesIO(uploaded_bytes), format=load_fmt)

    #chunk_ms = int(DEFAULT_CHUNK_MIN * 60_000)
    # ============================================================
    # chunk length
    # - Gemini文字起こしでは20分チャンクでMALFORMED_RESPONSEが出ることがあるため15分にする
    # - OpenAI/Whisperは従来通り DEFAULT_CHUNK_MIN を使う
    # ============================================================
    chunk_min = 15 if str(transcript_model).startswith("gemini") else int(DEFAULT_CHUNK_MIN)
    chunk_ms = int(chunk_min * 60_000)

    overlap_ms = int(float(DEFAULT_OVERLAP_MIN) * 60_000)

    parts = split_with_overlap(
        audio=audio,
        chunk_ms=chunk_ms,
        overlap_ms=overlap_ms,
        fade_ms=int(DEFAULT_FADE_MS),
        absorb_tiny_tail=bool(DEFAULT_ABSORB_TINY_TAIL),
    )

    original_path = original_dir / safe_filename(uploaded_name)
    original_path.write_bytes(uploaded_bytes)

    base_name = (Path(uploaded_name).stem or "audio").replace(" ", "_")

    if DEFAULT_EXPORT_FMT == "wav":
        out_ext = "wav"
        export_kwargs = {"format": "wav"}
    else:
        out_ext = "mp3"
        export_kwargs = {"format": "mp3"}
        if DEFAULT_TARGET_BITRATE:
            export_kwargs["bitrate"] = DEFAULT_TARGET_BITRATE

    index_rows: list[dict[str, Any]] = []
    split_paths: list[Path] = []

    for i, p in enumerate(parts):
        start_tag = hhmmss(p["start_ms"]).replace(":", "")
        end_tag = hhmmss(p["end_ms"]).replace(":", "")
        filename = f"{base_name}_part{i:03d}_{start_tag}-{end_tag}.{out_ext}"
        out_path = split_dir / filename

        buf = io.BytesIO()
        p["segment"].export(buf, **export_kwargs)
        out_path.write_bytes(buf.getvalue())

        split_paths.append(out_path)

        index_rows.append(
            {
                "part": i,
                "start_ms": p["start_ms"],
                "end_ms": p["end_ms"],
                "start_hhmmss": hhmmss(p["start_ms"]),
                "end_hhmmss": hhmmss(p["end_ms"]),
                "file": filename,
            }
        )

    job_json = {
        "job_id": job_id,
        "user": str(current_user),
        "user_dir": str(username_dir),
        "date": job_root.parent.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "batch_35",
        "paths": {
            "job_root": str(job_root),
            "original_dir": str(original_dir),
            "original": str(original_path),
            "split_dir": str(split_dir),
            "transcript_dir": str(job_root / "transcript"),
            "transcript_combined_dir": str(job_root / "transcript_combined"),
            "transcript_speaker_separated_dir": str(job_root / "transcript_speaker_separated"),
            "transcript_speaker_separated_combined_dir": str(job_root / "transcript_speaker_separated_combined"),
            "transcript_marked_dir": str(job_root / "transcript_marked"),
            "logs_dir": str(job_root / "logs"),
        },
        "config": {
            "chunk_min": int(chunk_min),
            "overlap_min": float(DEFAULT_OVERLAP_MIN),
            "export_fmt": DEFAULT_EXPORT_FMT,
            "target_bitrate": DEFAULT_TARGET_BITRATE,
            "fade_ms": int(DEFAULT_FADE_MS),
            "absorb_tiny_tail": bool(DEFAULT_ABSORB_TINY_TAIL),
        },
        "status": {
            "split": "done",
            "transcribe": "not_started",
            "speaker_separate": "not_started",
            "dedup": "not_started",
        },
        "split_index": index_rows,
        "audio_total_ms": int(len(audio)),
    }

    write_json(job_root / "job.json", job_json)

    append_log(log_path, "AUDIO SPLIT DONE")
    append_log(log_path, f"split_count={len(split_paths)}")

    return split_paths