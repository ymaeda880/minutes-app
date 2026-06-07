# -*- coding: utf-8 -*-
# minutes_app/lib/batch_processing/speaker_step.py
# ============================================================
# 一括処理：話者分離
# ============================================================

from __future__ import annotations

import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

from common_lib.ai.routing import call_text
from common_lib.busy import busy_run

from lib.prompts import (
    SPEAKER_MANDATORY,
    SPEAKER_MANDATORY_LIGHT,
    SPEAKER_MANDATORY_LIGHTER,
    SPEAKER_PREP,
    build_prompt,
    get_group,
)

from .constants import DEFAULT_SPEAKER_PROMPT_LEVEL
from .io_utils import append_log, read_text_safely, safe_mkdir, write_text


# ============================================================
# transcript list
# ============================================================
_PART_RE = re.compile(r"_part(\d+)_", re.IGNORECASE)


def _sort_key_transcript(p: Path) -> tuple[int, str]:
    m = _PART_RE.search(p.name)
    if m:
        return (int(m.group(1)), p.name.lower())
    return (10**9, p.name.lower())


def list_transcript_txts(job_dir: Path) -> list[Path]:
    transcript_dir = job_dir / "transcript"
    if not transcript_dir.exists():
        return []

    files: list[Path] = []
    for p in transcript_dir.glob("*.txt"):
        n = p.name.lower()
        if "speaker" in n:
            continue
        if "marked" in n:
            continue
        files.append(p)

    return sorted(files, key=_sort_key_transcript)


# ============================================================
# prompt
# ============================================================
def make_connector_line(prev_name: str) -> str:
    return f"----- ここがつなぎ目です（{prev_name} と次のファイルの間）-----"


def get_speaker_prompts(prompt_level: str) -> tuple[str, str, str]:
    group = get_group(SPEAKER_PREP)

    if prompt_level == "標準（精度優先）":
        mandatory = SPEAKER_MANDATORY
    elif prompt_level == "軽量（タイムアウト低減）":
        mandatory = SPEAKER_MANDATORY_LIGHT
    else:
        mandatory = SPEAKER_MANDATORY_LIGHTER

    preset_label = group.label_for_key(group.default_preset_key)
    preset_text = group.body_for_label(preset_label)
    extra_text = ""

    return mandatory, preset_text, extra_text


# ============================================================
# ai call
# ============================================================
def run_speaker_prep_one(
    *,
    prompt: str,
    provider: str,
    model: str,
) -> Any:
    return call_text(
        provider=str(provider),
        model=str(model),
        prompt=str(prompt),
        system=None,
        temperature=None,
        max_output_tokens=None,
        extra=None,
    )


# ============================================================
# run speaker step
# ============================================================
def run_speaker_step(
    *,
    job_root: Path,
    speaker_sep_dir: Path,
    speaker_combined_dir: Path,
    speaker_model_key: str,
    prompt_level: str,
    log_path: Path,
    projects_root: Any,
    user_sub: str,
    app_name: str,
    page_name: str,
    progress_callback: Optional[Callable[[int, int, Path], None]] = None,
) -> Path:
    transcript_files = list_transcript_txts(job_root)

    if not transcript_files:
        raise RuntimeError("transcript/*.txt が見つかりません。")

    provider = str(speaker_model_key).split(":", 1)[0]
    model = str(speaker_model_key).split(":", 1)[1] if ":" in str(speaker_model_key) else str(speaker_model_key)

    mandatory_prompt, preset_text, extra_text = get_speaker_prompts(prompt_level)

    safe_mkdir(speaker_sep_dir)
    safe_mkdir(speaker_combined_dir)

    append_log(log_path, "SPEAKER PREP START")
    append_log(log_path, f"model_key={speaker_model_key}")
    append_log(log_path, f"provider={provider}")
    append_log(log_path, f"model={model}")
    append_log(log_path, f"prompt_level={prompt_level}")

    combined_chunks: list[str] = []
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_path = speaker_combined_dir / f"transcript_speaker_separated_combined_{ts_tag}.txt"

    with busy_run(
        projects_root=projects_root,
        user_sub=str(user_sub),
        app_name=str(app_name),
        page_name=str(page_name),
        task_type="text",
        provider=str(provider),
        model=str(model),
        meta={
            "feature": "minutes_batch_35",
            "action": "speaker_prep",
            "job_dir": str(job_root),
            "file_count": len(transcript_files),
            "prompt_level": prompt_level,
        },
    ) as br:
        total_in = 0
        total_out = 0
        total_usd: Optional[float] = 0.0
        total_jpy: Optional[float] = 0.0
        processed_count = 0
        total_elapsed = 0.0

        for i, src_path in enumerate(transcript_files, start=1):

            # ============================================================
            # progress callback
            # - page側に chunk 進捗を通知する
            # ============================================================
            if progress_callback is not None:
                progress_callback(int(i), int(len(transcript_files)), src_path)

            src = read_text_safely(src_path).strip()

            if not src:
                continue

            prompt = build_prompt(
                mandatory_prompt,
                preset_text,
                extra_text,
                src,
            )

            t0 = time.perf_counter()
            res = run_speaker_prep_one(
                prompt=prompt,
                provider=str(provider),
                model=str(model),
            )
            elapsed = time.perf_counter() - t0

            text = getattr(res, "text", "") or ""

            out_path = speaker_sep_dir / f"{i:03d}_{src_path.stem}_speaker.txt"
            write_text(out_path, text)

            combined_chunks.append(text or "")

            if i < len(transcript_files):
                combined_chunks.append(make_connector_line(src_path.name))

            usage = getattr(res, "usage", None)
            in_tok = getattr(usage, "input_tokens", None) if usage is not None else None
            out_tok = getattr(usage, "output_tokens", None) if usage is not None else None

            if isinstance(in_tok, int):
                total_in += int(in_tok)
            if isinstance(out_tok, int):
                total_out += int(out_tok)

            cost = getattr(res, "cost", None)
            usd = getattr(cost, "usd", None) if cost is not None else None
            jpy = getattr(cost, "jpy", None) if cost is not None else None

            if (usd is not None) and (total_usd is not None):
                total_usd += float(usd)
            elif usd is None:
                total_usd = None

            if (jpy is not None) and (total_jpy is not None):
                total_jpy += float(jpy)
            elif jpy is None:
                total_jpy = None

            total_elapsed += float(elapsed)
            processed_count += 1

            append_log(
                log_path,
                f"SPEAKER DONE {src_path.name} -> {out_path.name} "
                f"in={in_tok} out={out_tok} usd={usd} jpy={jpy} elapsed={elapsed:.2f}s",
            )

        combined_text = "\n\n".join(combined_chunks)
        write_text(combined_path, combined_text)

        try:
            br.set_usage(int(total_in), int(total_out))
        except Exception:
            pass

        try:
            if (total_usd is not None) and (total_jpy is not None):
                br.set_cost(float(total_usd), float(total_jpy))
        except Exception:
            pass

        br.add_finish_meta(
            note="ok",
            processed_count=int(processed_count),
            total_elapsed_sec=round(float(total_elapsed), 3),
            total_tokens_in=int(total_in),
            total_tokens_out=int(total_out),
            total_usd=(round(float(total_usd), 6) if total_usd is not None else None),
            total_jpy=(round(float(total_jpy), 2) if total_jpy is not None else None),
            combined_path=str(combined_path),
            speaker_sep_dir=str(speaker_sep_dir),
        )

    append_log(log_path, f"SPEAKER COMBINED -> {combined_path.name}")
    append_log(log_path, "SPEAKER PREP DONE")

    return combined_path