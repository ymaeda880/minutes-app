# -*- coding: utf-8 -*-
# minutes_app/pages/35_音声ファイル一括処理.py
# ============================================================
# 🎧 音声ファイル一括処理
#
# 機能：
# - 音声ファイルをアップロードする
# - 音声分割
# - 文字起こし
# - 話者分離
# - 話者分離済みテキストを結合
# - 重複箇所にマーカーを挿入
# - 逐語録作成用の準備テキストとしてダウンロード
#
# 方針：
# - 中間テキストの表示・ダウンロードはしない
# - 各処理の完了だけを表示する
# - 最後に「逐語録作成用の準備テキスト」だけをダウンロードする
# - st.form は使わない
# - st.button / st.download_button に width 引数は使わない
# - use_container_width は使わない
# ============================================================

from __future__ import annotations

# ============================================================
# 標準ライブラリ
# ============================================================
import io
import json
import re
import sys
import time
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# サードパーティ
# ============================================================
import streamlit as st
from pydub import AudioSegment

# ============================================================
# sys.path（common_lib を import できるように）
# ============================================================
_THIS = Path(__file__).resolve()
APP_DIR = _THIS.parents[1]
PROJ_DIR = _THIS.parents[2]
MONO_ROOT = _THIS.parents[3]

for p in (MONO_ROOT, PROJ_DIR, APP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

PROJECTS_ROOT = MONO_ROOT
APP_NAME = _THIS.parents[1].name
PAGE_NAME = _THIS.stem

# ============================================================
# common_lib（正本）
# ============================================================
from common_lib.ui.page_header import render_standard_page_header
from common_lib.storage.external_ssd_root import resolve_storage_subdir_root
from common_lib.busy import busy_run

# ============================================================
# AI 正本
# ============================================================
from common_lib.ai.routing import transcribe_audio
from common_lib.ai.routing import call_text
from common_lib.ai.types import TranscribeResult

# ============================================================
# モデル正本
# ============================================================
from common_lib.ai.models import TRANSCRIBE_MODELS
from common_lib.ai.models import TEXT_MODEL_CATALOG
from common_lib.ai.models import DEFAULT_TEXT_MODEL_KEY

# ============================================================
# モデル選択UI
# ============================================================
from common_lib.ui.model_picker import render_text_model_picker

# ============================================================
# 既存アプリ資産
# ============================================================
from lib.audio import get_audio_duration_seconds

from lib.prompts import SPEAKER_PREP
from lib.prompts import SPEAKER_MANDATORY
from lib.prompts import SPEAKER_MANDATORY_LIGHT
from lib.prompts import SPEAKER_MANDATORY_LIGHTER
from lib.prompts import get_group
from lib.prompts import build_prompt

# ============================================================
# ページ説明
# ============================================================
from lib.batch_processing.explanation import (
    render_batch_processing_page_intro,
    render_batch_processing_help_expander,
)

# ============================================================
# ページ設定
# ============================================================
st.set_page_config(
    page_title="Minutes Maker",
    page_icon="🎧",
    layout="wide",
)

# ============================================================
# Storage root
# ============================================================
STORAGE_ROOT = resolve_storage_subdir_root(
    PROJECTS_ROOT,
    subdir="Storages",
)

# ============================================================
# 一括処理デフォルト設定
# ============================================================

# ------------------------------------------------------------
# 音声分割設定
# ------------------------------------------------------------
DEFAULT_CHUNK_MIN = 20
DEFAULT_OVERLAP_MIN = 1.0
DEFAULT_EXPORT_FMT = "mp3"
DEFAULT_TARGET_BITRATE = "160k"
DEFAULT_FADE_MS = 0
DEFAULT_ABSORB_TINY_TAIL = True

# ------------------------------------------------------------
# 文字起こし設定
# ------------------------------------------------------------
DEFAULT_TRANSCRIBE_MODEL = "whisper-1"
DEFAULT_TRANSCRIBE_RESPONSE_FORMAT = "json"
DEFAULT_LANGUAGE = "ja"
DEFAULT_TRANSCRIBE_PROMPT = ""
DEFAULT_STRIP_BRACKET_TAGS = True

# ------------------------------------------------------------
# 話者分離設定
# ------------------------------------------------------------
DEFAULT_SPEAKER_PROMPT_LEVEL = "標準（精度優先）"

# ------------------------------------------------------------
# 重複検出設定
# ------------------------------------------------------------
OVERLAP_CHARS = 700
HEAD_CHARS = 400
HEAD_SENTENCES = 3
HEAD_SHIFT_TRIES = 3
DEFAULT_MIN_MATCH_SIZE = 20
DEFAULT_USE_AUTOJUNK = True
BEGIN_TAG = "-----ここから重複-----"

MARKER_PATTERN = re.compile(
    r"^-{3,}\s*ここがつなぎ目です（(.*?)）.*$",
    re.MULTILINE,
)

BRACKET_TAG_PATTERN = re.compile(r"【[^】]*】")

AUDIO_EXTS = {".mp3", ".wav", ".mp4", ".m4a", ".webm", ".ogg"}

# ============================================================
# page key
# ============================================================
PAGE_KEY_PREFIX = PAGE_NAME


def k(name: str) -> str:
    return f"{PAGE_KEY_PREFIX}::{name}"


# ============================================================
# Gemini 利用可否
# ============================================================
def _gemini_available() -> bool:
    try:
        from google import genai

        _ = genai
        return True
    except Exception:
        return False


# ============================================================
# 共通ヘッダー
# ============================================================
sub, theme, BANNER_KEY, settings = render_standard_page_header(
    st_module=st,
    projects_root=PROJECTS_ROOT,
    app_dir=APP_DIR,
    app_name=APP_NAME,
    page_name=PAGE_NAME,
    title="🎧 音声ファイル一括処理",
    subtitle_text="音声分割・文字起こし・話者分離・重複検出をまとめて実行",
    default_banner_key="light_green",
)

# ============================================================
# ページ説明
# ============================================================
render_batch_processing_page_intro()

render_batch_processing_help_expander(
    theme=theme,
    banner_key="light_green",
)

# ============================================================
# ユーザー名のフォルダ安全化
# ============================================================
def _sanitize_username_for_path(username: str) -> str:
    u = (username or "").strip()
    if not u:
        return "anonymous"
    u = re.sub(r"[^0-9A-Za-z_-]+", "_", u).strip("_")
    return u or "anonymous"


current_user = sub
USERNAME_DIR = _sanitize_username_for_path(str(current_user))
USER_ROOT = STORAGE_ROOT / USERNAME_DIR / "minutes_app"

# ============================================================
# ファイルI/O
# ============================================================
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_text(p: Path, s: str) -> None:
    safe_mkdir(p.parent)
    p.write_text(s, encoding="utf-8")


def write_json(p: Path, obj: Any) -> None:
    safe_mkdir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_log(log_path: Path, msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_mkdir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def now_job_id() -> str:
    return "job_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def hhmmss(ms: int) -> str:
    return str(timedelta(milliseconds=ms)).split(".")[0]


def safe_filename(name: str) -> str:
    s = (name or "audio").strip()
    s = s.replace("\\", "_").replace("/", "_").replace(":", "_")
    s = re.sub(r"\s+", "_", s)
    return s or "audio"


def read_text_safely(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="cp932", errors="replace")


# ============================================================
# 音声分割ユーティリティ
# ============================================================
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
            if tail_len < overlap_ms:
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
# 文字起こしユーティリティ
# ============================================================
def strip_bracket_tags(text: str) -> str:
    if not text:
        return text
    return BRACKET_TAG_PATTERN.sub("", text)


# ============================================================
# 話者分離ユーティリティ
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
# 重複検出ユーティリティ
# ============================================================
SPEAKER_PREFIX_PATTERN = re.compile(
    r"""^(
        \s*
        (?:司会|ＭＣ|MC|進行)
        \s*[:：]\s*
      |
        \s*\[?\s*[sS]\s*\d+\s*\]?\s*[:：]\s*
    )""",
    re.VERBOSE,
)


def split_by_markers(text: str) -> Tuple[List[str], List[Dict[str, str]]]:
    segments: List[str] = []
    markers: List[Dict[str, str]] = []
    prev_end = 0

    for m in MARKER_PATTERN.finditer(text):
        seg = text[prev_end : m.start()]
        segments.append(seg)
        markers.append(
            {
                "file_name": m.group(1),
                "marker_text": m.group(0),
            }
        )
        prev_end = m.end()

    segments.append(text[prev_end:])
    return segments, markers


def strip_leading_speaker_labels(text: str) -> str:
    if not text:
        return ""
    s = text.lstrip("\ufeff")
    for _ in range(5):
        m = SPEAKER_PREFIX_PATTERN.match(s)
        if not m:
            break
        s = s[m.end() :]
    return s


def extract_head_phrase(next_seg: str) -> str:
    if not next_seg:
        return ""

    next_seg = strip_leading_speaker_labels(next_seg)

    s = next_seg[:HEAD_CHARS]
    count = 0
    end = len(s)

    for i, ch in enumerate(s):
        if ch in "。？！\n":
            count += 1
            if count >= HEAD_SENTENCES:
                end = i + 1
                break

    return s[:end]


def normalize_text(s: str) -> str:
    s = s.replace("　", "")
    s = s.replace("\n", "")
    s = re.sub(r"[、。！？,.!?]", "", s)
    s = s.replace(" ", "")
    return s


def _match_with_phrase(
    prev_seg: str,
    phrase: str,
    min_match_size: int,
    use_autojunk: bool,
) -> Tuple[int, int]:
    if not prev_seg or not phrase:
        return -1, 0
    if len(phrase) < min_match_size:
        return -1, 0

    prev_tail = prev_seg[-OVERLAP_CHARS:]
    norm_head = normalize_text(phrase)
    norm_prev = normalize_text(prev_tail)

    if len(norm_head) < min_match_size:
        return -1, 0

    sm = SequenceMatcher(None, norm_head, norm_prev, autojunk=use_autojunk)
    blocks = sm.get_matching_blocks()

    cand = [(b.a, b.b, b.size) for b in blocks if b.size >= min_match_size]
    if not cand:
        return -1, 0

    a, b, size = sorted(cand, key=lambda t: (t[0], t[1], -t[2]))[0]

    def build_index_map(raw: str, norm: str):
        mapping = []
        j = 0
        for i, ch in enumerate(raw):
            ch_norm = normalize_text(ch)
            if ch_norm == "":
                continue
            if j < len(norm):
                mapping.append((j, i))
                j += 1
        return mapping

    head_map = build_index_map(phrase, norm_head)
    prev_map = build_index_map(prev_tail, norm_prev)

    def mapped_index(mapping, idx_norm):
        candidates = [raw_idx for norm_idx, raw_idx in mapping if norm_idx == idx_norm]
        if candidates:
            return candidates[0]

        nearest = None
        best_dist = 10**9
        for norm_idx, raw_idx in mapping:
            d = abs(norm_idx - idx_norm)
            if d < best_dist:
                best_dist = d
                nearest = raw_idx
        return nearest

    head_raw_start = mapped_index(head_map, a)
    prev_raw_start = mapped_index(prev_map, b)

    start_in_tail = max(0, int(prev_raw_start) - int(head_raw_start))
    global_prev_start = len(prev_seg) - len(prev_tail) + start_in_tail

    return global_prev_start, int(size)


def find_overlap_start(
    prev_seg: str,
    next_seg: str,
    min_match_size: int,
    use_autojunk: bool,
) -> Tuple[int, int, str, List[str], str]:
    if not prev_seg or not next_seg:
        return -1, 0, "", [], ""

    raw_head = extract_head_phrase(next_seg)
    if not raw_head:
        return -1, 0, "", [], ""

    base_head = raw_head.strip().rstrip("。？！!？，、,.")
    if not base_head:
        return -1, 0, "", [], ""

    candidates: List[str] = []
    seen: set[str] = set()
    shifted_heads: List[str] = []

    def add_candidate(s: str, is_shifted: bool = False):
        if not s:
            return
        if s in seen:
            return
        seen.add(s)
        candidates.append(s)
        if is_shifted:
            shifted_heads.append(s)

    add_candidate(base_head, is_shifted=False)

    current = base_head
    for _ in range(HEAD_SHIFT_TRIES):
        if len(current) <= 1:
            break
        current = current[1:]
        add_candidate(current, is_shifted=True)

    matched_phrase = ""
    for phrase in candidates:
        start_pos, size = _match_with_phrase(
            prev_seg,
            phrase,
            min_match_size,
            use_autojunk,
        )
        if start_pos >= 0 and size > 0:
            matched_phrase = phrase
            return start_pos, size, base_head, shifted_heads, matched_phrase

    return -1, 0, base_head, shifted_heads, ""


def build_merged_text(
    text: str,
    min_match_size: int,
    use_autojunk: bool,
) -> Tuple[str, List[Dict[str, Any]]]:
    segments, markers = split_by_markers(text)

    if len(segments) <= 1 or not markers:
        return text, []

    merged: List[str] = []
    logs: List[Dict[str, Any]] = []

    merged.append(segments[0])

    for idx, marker in enumerate(markers):
        prev_seg = segments[idx]
        next_seg = segments[idx + 1]

        start_pos, size, base_head, shifted_heads, matched_phrase = find_overlap_start(
            prev_seg,
            next_seg,
            min_match_size,
            use_autojunk,
        )

        if start_pos < 0 or size <= 0:
            logs.append(
                {
                    "つなぎ目番号": idx,
                    "ファイル名": marker.get("file_name", ""),
                    "検出結果": "見つからず",
                    "開始位置": None,
                    "一致文字数": 0,
                    "head_phrase": base_head,
                    "shifted_phrases": shifted_heads,
                    "matched_phrase": "",
                }
            )
            merged.append("\n" + marker["marker_text"] + "\n")
            merged.append(next_seg)
            continue

        logs.append(
            {
                "つなぎ目番号": idx,
                "ファイル名": marker.get("file_name", ""),
                "検出結果": "検出",
                "開始位置": start_pos,
                "一致文字数": size,
                "head_phrase": base_head,
                "shifted_phrases": shifted_heads,
                "matched_phrase": matched_phrase,
            }
        )

        new_prev = prev_seg[:start_pos] + "\n" + BEGIN_TAG + "\n" + prev_seg[start_pos:]
        merged[-1] = new_prev

        merged.append("\n" + marker["marker_text"] + "\n")
        merged.append(next_seg)

    return "".join(merged), logs


# ============================================================
# job作成
# ============================================================
def create_job_root_for_uploaded_file(
    *,
    uploaded_name: str,
) -> tuple[str, Path, Path, Path, Path, Path, Path, Path]:
    today_dir = datetime.now().strftime("%Y-%m-%d")
    job_id = now_job_id()
    job_root = USER_ROOT / today_dir / job_id

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


# ============================================================
# 1ファイル処理：音声分割
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
) -> list[Path]:
    suffix = Path(uploaded_name).suffix.lower()

    if suffix not in AUDIO_EXTS:
        raise ValueError("対応していない拡張子です。")

    load_fmt = _guess_format_from_suffix(suffix)
    audio = AudioSegment.from_file(io.BytesIO(uploaded_bytes), format=load_fmt)

    chunk_ms = int(DEFAULT_CHUNK_MIN * 60_000)
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
        chunk_bytes = buf.getvalue()
        out_path.write_bytes(chunk_bytes)

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
        "user_dir": USERNAME_DIR,
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
            "chunk_min": int(DEFAULT_CHUNK_MIN),
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


# ============================================================
# 1ファイル処理：文字起こし
# ============================================================
def run_transcribe_step(
    *,
    split_paths: list[Path],
    job_root: Path,
    transcript_dir: Path,
    transcript_model: str,
    language: str,
    log_path: Path,
) -> Path:
    safe_mkdir(transcript_dir)

    provider = "gemini" if str(transcript_model).startswith("gemini") else "openai"
    combined_parts: list[str] = []

    append_log(log_path, "TRANSCRIBE START")
    append_log(log_path, f"model={transcript_model}")
    append_log(log_path, f"language={language}")

    with busy_run(
        projects_root=PROJECTS_ROOT,
        user_sub=str(sub),
        app_name=str(APP_NAME),
        page_name=str(PAGE_NAME),
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
            file_bytes = path.read_bytes()
            mime = guess_mime_from_suffix(path)

            audio_sec: Optional[float] = None
            try:
                audio_sec = float(get_audio_duration_seconds(io.BytesIO(file_bytes)))
            except Exception:
                audio_sec = None

            lang_arg = language.strip() if language and language.strip() else None
            prompt_arg = DEFAULT_TRANSCRIBE_PROMPT.strip() if DEFAULT_TRANSCRIBE_PROMPT.strip() else None

            t0 = time.perf_counter()

            tr: TranscribeResult = transcribe_audio(
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
                    "page": str(PAGE_NAME),
                    "job_dir": str(job_root),
                    "chunk_name": str(path.name),
                },
            )

            elapsed = time.perf_counter() - t0

            text = tr.text or ""
            if DEFAULT_STRIP_BRACKET_TAGS:
                text = strip_bracket_tags(text)

            out_txt = transcript_dir / f"{idx:03d}_{path.stem}.txt"
            write_text(out_txt, text)

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

    return combined_path


# ============================================================
# 1ファイル処理：話者分離
# ============================================================
def run_speaker_step(
    *,
    job_root: Path,
    speaker_sep_dir: Path,
    speaker_combined_dir: Path,
    speaker_model_key: str,
    prompt_level: str,
    log_path: Path,
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
        projects_root=PROJECTS_ROOT,
        user_sub=str(sub),
        app_name=str(APP_NAME),
        page_name=str(PAGE_NAME),
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


# ============================================================
# 1ファイル処理：重複検出
# ============================================================
def run_overlap_step(
    *,
    job_root: Path,
    combined_path: Path,
    log_path: Path,
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


# ============================================================
# job.json status更新
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


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("モデル設定")

    st.session_state.setdefault(k("transcribe_model"), DEFAULT_TRANSCRIBE_MODEL)

    transcribe_model = st.radio(
        "文字起こしモデル",
        options=TRANSCRIBE_MODELS,
        index=TRANSCRIBE_MODELS.index(DEFAULT_TRANSCRIBE_MODEL)
        if DEFAULT_TRANSCRIBE_MODEL in TRANSCRIBE_MODELS
        else 0,
        key=k("transcribe_model"),
    )

    st.divider()

    st.session_state.setdefault(k("speaker_model_key"), DEFAULT_TEXT_MODEL_KEY)

    speaker_model_key = render_text_model_picker(
        title="話者分離モデル",
        catalog=TEXT_MODEL_CATALOG,
        session_key=k("speaker_model_key"),
        default_key=DEFAULT_TEXT_MODEL_KEY,
        page_name=PAGE_NAME,
        gemini_available=_gemini_available(),
    )

    st.divider()

    st.header("言語")

    language = st.selectbox(
        "文字起こし言語コード",
        options=["ja", "en", ""],
        index=0,
        key=k("language"),
        help="空欄を選ぶと自動判定に近い扱いになります。",
    )

# ============================================================
# メインUI
# ============================================================
st.markdown("### ① 音声ファイルのアップロード")

uploaded_files = st.file_uploader(
    "音声ファイルをアップロード（複数可）",
    type=["mp3", "wav", "mp4", "m4a", "webm", "ogg"],
    accept_multiple_files=True,
    key=k("uploader_audio"),
)

if not uploaded_files:
    st.info("音声ファイルを1つ以上アップロードしてください。")
    st.stop()

st.caption("中間テキストは表示せず、最後に逐語録作成用の準備テキストだけをダウンロードします。")

with st.expander("処理対象ファイル", expanded=False):
    for i, f in enumerate(uploaded_files, start=1):
        st.markdown(f"- {i}. `{getattr(f, 'name', '(no name)')}`")

# ============================================================
# セッションキー
# ============================================================
K_LAST_MARKED_TEXT = k("last_marked_text")
K_LAST_DOWNLOAD_NAME = k("last_download_name")
K_LAST_MARKED_PATH = k("last_marked_path")

st.session_state.setdefault(K_LAST_MARKED_TEXT, "")
st.session_state.setdefault(K_LAST_DOWNLOAD_NAME, "")
st.session_state.setdefault(K_LAST_MARKED_PATH, "")

# ============================================================
# 実行
# ============================================================
st.divider()
st.markdown("### ② 一括処理")

run_btn = st.button(
    "▶️ 一括処理を開始",
    type="primary",
    key=k("run_batch"),
)

if run_btn:
    all_final_texts: list[str] = []
    final_paths: list[Path] = []

    overall_progress = st.progress(0, text="準備中...")
    status_area = st.container()

    total_files = len(uploaded_files)

    for file_index, uf in enumerate(uploaded_files, start=1):
        uploaded_name = safe_filename(getattr(uf, "name", f"audio_{file_index}.mp3"))
        uploaded_bytes = uf.getvalue()

        with status_area:
            st.markdown(f"#### {file_index}/{total_files}：{uploaded_name}")

        (
            job_id,
            job_root,
            original_dir,
            split_dir,
            transcript_dir,
            transcript_combined_dir,
            speaker_sep_dir,
            speaker_combined_dir,
        ) = create_job_root_for_uploaded_file(uploaded_name=uploaded_name)

        log_path = job_root / "logs" / "process.log"

        append_log(log_path, "BATCH 35 START")
        append_log(log_path, f"uploaded_name={uploaded_name}")
        append_log(log_path, f"user_display={current_user}")
        append_log(log_path, f"user_dir={USERNAME_DIR}")

        try:
            # ------------------------------------------------------------
            # 1. 音声分割
            # ------------------------------------------------------------
            overall_progress.progress(
                int(((file_index - 1) / total_files) * 100),
                text=f"{file_index}/{total_files} 音声分割中：{uploaded_name}",
            )

            split_paths = run_audio_split_step(
                uploaded_name=uploaded_name,
                uploaded_bytes=uploaded_bytes,
                job_id=job_id,
                job_root=job_root,
                original_dir=original_dir,
                split_dir=split_dir,
                log_path=log_path,
            )

            update_job_status(job_root, "split", "done")

            with status_area:
                st.success(f"✓ 音声分割完了（{len(split_paths)}チャンク）")

            # ------------------------------------------------------------
            # 2. 文字起こし
            # ------------------------------------------------------------
            overall_progress.progress(
                int(((file_index - 1 + 0.25) / total_files) * 100),
                text=f"{file_index}/{total_files} 文字起こし中：{uploaded_name}",
            )

            _combined_transcript_path = run_transcribe_step(
                split_paths=split_paths,
                job_root=job_root,
                transcript_dir=transcript_dir,
                transcript_model=str(transcribe_model),
                language=str(language),
                log_path=log_path,
            )

            update_job_status(job_root, "transcribe", "done")

            with status_area:
                st.success("✓ 文字起こし完了")



            # ============================================================
            # DEBUG : stop after transcribe
            # ============================================================
            st.warning("DEBUG: 文字起こし完了で停止")
            st.stop()
            # ============================================================
            # DEBUG END
            # ============================================================

            

            # ------------------------------------------------------------
            # 3. 話者分離
            # ------------------------------------------------------------
            overall_progress.progress(
                int(((file_index - 1 + 0.50) / total_files) * 100),
                text=f"{file_index}/{total_files} 話者分離中：{uploaded_name}",
            )

            speaker_combined_path = run_speaker_step(
                job_root=job_root,
                speaker_sep_dir=speaker_sep_dir,
                speaker_combined_dir=speaker_combined_dir,
                speaker_model_key=str(speaker_model_key),
                prompt_level=DEFAULT_SPEAKER_PROMPT_LEVEL,
                log_path=log_path,
            )

            update_job_status(job_root, "speaker_separate", "done")

            with status_area:
                st.success("✓ 話者分離完了")

            # ------------------------------------------------------------
            # 4. 重複検出
            # ------------------------------------------------------------
            overall_progress.progress(
                int(((file_index - 1 + 0.75) / total_files) * 100),
                text=f"{file_index}/{total_files} 重複箇所検出中：{uploaded_name}",
            )

            marked_path = run_overlap_step(
                job_root=job_root,
                combined_path=speaker_combined_path,
                log_path=log_path,
            )

            update_job_status(job_root, "dedup", "done")

            final_text = read_text_safely(marked_path)
            all_final_texts.append(final_text)
            final_paths.append(marked_path)

            append_log(log_path, f"FINAL MARKED -> {marked_path}")
            append_log(log_path, "BATCH 35 DONE")

            with status_area:
                st.success("✓ 重複箇所検出完了")

        except Exception as e:
            append_log(log_path, f"BATCH 35 ERROR: {e}")
            with status_area:
                st.error(f"処理に失敗しました：{uploaded_name}")
                st.exception(e)
            st.stop()

    overall_progress.progress(100, text="全処理が完了しました。")

    # ============================================================
    # 最終テキストの作成
    # - 複数ファイルの場合も1つの準備テキストに結合する
    # ============================================================
    if len(all_final_texts) == 1:
        download_text = all_final_texts[0]
    else:
        joined_parts: list[str] = []
        for i, text in enumerate(all_final_texts, start=1):
            src_name = safe_filename(getattr(uploaded_files[i - 1], "name", f"audio_{i}"))
            joined_parts.append(f"\n\n===== 入力音声 {i}: {src_name} =====\n\n")
            joined_parts.append(text)
        download_text = "".join(joined_parts).strip()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if len(uploaded_files) == 1:
        first_name = Path(safe_filename(getattr(uploaded_files[0], "name", "audio"))).stem
        download_name = f"{first_name}_逐語録作成用_重複マーク付き準備テキスト_{ts}.txt"
    else:
        download_name = f"逐語録作成用_重複マーク付き準備テキスト_{ts}.txt"

    st.session_state[K_LAST_MARKED_TEXT] = download_text
    st.session_state[K_LAST_DOWNLOAD_NAME] = download_name
    st.session_state[K_LAST_MARKED_PATH] = str(final_paths[-1]) if final_paths else ""

    st.success("全ての処理が完了しました。")

# ============================================================
# ダウンロード
# ============================================================
st.divider()
st.markdown("### ③ ダウンロード")

marked_text = st.session_state.get(K_LAST_MARKED_TEXT, "")
download_name = st.session_state.get(K_LAST_DOWNLOAD_NAME, "")

if not marked_text:
    st.info("一括処理が完了すると、ここにダウンロードボタンが表示されます。")
else:
    st.download_button(
        "📥 逐語録作成用の準備テキストをダウンロード",
        data=str(marked_text).encode("utf-8"),
        file_name=str(download_name or "逐語録作成用_重複マーク付き準備テキスト.txt"),
        mime="text/plain",
        key=k("download_final_marked"),
    )