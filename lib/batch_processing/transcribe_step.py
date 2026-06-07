# -*- coding: utf-8 -*-
# minutes_app/lib/batch_processing/transcribe_step.py
# ============================================================
# 一括処理：文字起こし
# ============================================================

from __future__ import annotations

import io
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

from common_lib.ai.routing import transcribe_audio
from common_lib.ai.types import TranscribeResult
from common_lib.busy import busy_run

from lib.audio import get_audio_duration_seconds

from .audio_split import guess_mime_from_suffix
from .constants import (
    BRACKET_TAG_PATTERN,
    DEFAULT_STRIP_BRACKET_TAGS,
    DEFAULT_TRANSCRIBE_PROMPT,
    DEFAULT_TRANSCRIBE_RESPONSE_FORMAT,
)
from .io_utils import append_log, safe_mkdir, write_text


# ============================================================
# text cleanup
# ============================================================
def strip_bracket_tags(text: str) -> str:
    if not text:
        return text
    return BRACKET_TAG_PATTERN.sub("", text)


# ============================================================
# run transcribe
# ============================================================
def run_transcribe_step(
    *,
    split_paths: list[Path],
    job_root: Path,
    transcript_dir: Path,
    transcript_model: str,
    language: str,
    log_path: Path,
    projects_root: Any,
    user_sub: str,
    app_name: str,
    page_name: str,
    progress_callback: Optional[Callable[[int, int, Path], None]] = None,
) -> Path:
    safe_mkdir(transcript_dir)

    provider = "gemini" if str(transcript_model).startswith("gemini") else "openai"
    combined_parts: list[str] = []
    # ============================================================
    # chunk results
    # - page側で各チャンクの文字起こし結果を確認する
    # ============================================================
    chunk_results: list[dict[str, Any]] = []

    append_log(log_path, "TRANSCRIBE START")
    append_log(log_path, f"model={transcript_model}")
    append_log(log_path, f"language={language}")

    with busy_run(
        projects_root=projects_root,
        user_sub=str(user_sub),
        app_name=str(app_name),
        page_name=str(page_name),
        task_type="transcribe",
        provider=str(provider),
        model=str(transcript_model),
        meta={
            "feature": "minutes_batch_35",
            "action": "transcribe",
            "job_dir": str(job_root),
            "chunks": len(split_paths),
            "language": language,
        },
    ) as br:
        total_usd: Optional[float] = 0.0
        total_jpy: Optional[float] = 0.0

        for idx, path in enumerate(split_paths, start=1):

            # ============================================================
            # progress callback
            # - page側に chunk 進捗を通知する
            # ============================================================
            if progress_callback is not None:
                progress_callback(int(idx), int(len(split_paths)), path)

            file_bytes = path.read_bytes()
            mime = guess_mime_from_suffix(path)

            audio_sec: Optional[float] = None
            try:
                audio_sec = float(get_audio_duration_seconds(io.BytesIO(file_bytes)))
            except Exception:
                audio_sec = None

            lang_arg = language.strip() if language and language.strip() else None
            prompt_arg = DEFAULT_TRANSCRIBE_PROMPT.strip() if DEFAULT_TRANSCRIBE_PROMPT.strip() else None


            # ============================================================
            # transcribe with retry
            # - Gemini で MALFORMED_RESPONSE 等により空文字が返る場合がある
            # - 空文字は正常成果物として保存しない
            # - Gemini の場合だけ同一チャンクをリトライする
            # ============================================================
            max_attempts = 3 if provider == "gemini" else 1
            retry_sleep_sec = 3.0

            tr: TranscribeResult | None = None
            text = ""
            elapsed = 0.0

            for attempt in range(1, max_attempts + 1):
                t0 = time.perf_counter()

                tr = transcribe_audio(
                    provider=provider,
                    model=str(transcript_model),
                    audio_bytes=file_bytes,
                    mime_type=mime,
                    filename=path.name,
                    audio_seconds=float(audio_sec) if audio_sec is not None else None,
                    response_format=str(DEFAULT_TRANSCRIBE_RESPONSE_FORMAT),
                    language=lang_arg,
                    prompt=prompt_arg,
                    timeout_sec=600,
                    extra={
                        "page": str(page_name),
                        "job_dir": str(job_root),
                        "chunk_name": str(path.name),
                        "attempt": int(attempt),
                    },
                )

                elapsed += time.perf_counter() - t0

                text = tr.text or ""
                if DEFAULT_STRIP_BRACKET_TAGS:
                    text = strip_bracket_tags(text)

                # ------------------------------------------------------------
                # 空でなければ成功
                # ------------------------------------------------------------
                if text.strip():
                    break

                # ------------------------------------------------------------
                # 空文字の場合は保存せず、Geminiだけリトライ
                # ------------------------------------------------------------
                append_log(
                    log_path,
                    f"TRANSCRIBE EMPTY chunk={path.name} attempt={attempt}/{max_attempts}",
                )

                if attempt < max_attempts:
                    time.sleep(retry_sleep_sec)

            # ============================================================
            # empty guard
            # - リトライ後も空なら保存せず停止する
            # ============================================================
            if not text.strip():
                raise RuntimeError(
                    "文字起こし結果が空です。"
                    f" chunk={path.name}"
                    f" provider={provider}"
                    f" model={transcript_model}"
                    f" attempts={max_attempts}"
                )

            assert tr is not None



            out_txt = transcript_dir / f"{idx:03d}_{path.stem}.txt"
            write_text(out_txt, text)

            # ============================================================
            # chunk result
            # - DEBUG表示用に各チャンクの結果を返す
            # ============================================================
            chunk_results.append(
                {
                    "index": int(idx),
                    "source_name": str(path.name),
                    "out_txt": str(out_txt),
                    "text": str(text),
                    "text_len": int(len(text)),
                    "elapsed_sec": float(elapsed),
                    "audio_sec": float(audio_sec) if audio_sec is not None else None,
                    "request_id": str(tr.request_id or ""),
                }
            )

            append_log(
                log_path,
                f"SAVED transcript {path.name} -> {out_txt.name} elapsed={elapsed:.2f}s",
            )

            if tr.cost is not None:
                if total_usd is not None:
                    total_usd += float(tr.cost.usd)
                if total_jpy is not None:
                    total_jpy += float(tr.cost.jpy)
            else:
                total_usd = None
                total_jpy = None

            combined_parts.append(text or "")

            if idx < len(split_paths):
                combined_parts.append(
                    f"\n\n----- ここがつなぎ目です（{path.name} と次のファイルの間）-----\n\n"
                )

        if (total_usd is not None) and (total_jpy is not None):
            try:
                br.set_cost(float(total_usd), float(total_jpy))
            except Exception:
                pass

        br.add_finish_meta(
            note="ok",
            chunks_done=len(split_paths),
            total_usd=(round(float(total_usd), 6) if total_usd is not None else None),
            total_jpy=(round(float(total_jpy), 2) if total_jpy is not None else None),
        )

    combined_dir = job_root / "transcript_combined"
    safe_mkdir(combined_dir)

    first_stem = split_paths[0].stem if split_paths else "audio"
    base_name = first_stem.split("_part", 1)[0] if "_part" in first_stem else first_stem

    combined_path = combined_dir / f"{base_name}_combined.txt"
    write_text(combined_path, "".join(combined_parts))

    append_log(log_path, f"SAVED combined transcript -> {combined_path.name}")
    append_log(log_path, "TRANSCRIBE DONE")

    return combined_path, chunk_results