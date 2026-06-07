# -*- coding: utf-8 -*-
# minutes_app/pages/21_文字起こし_storage対応.py
# ============================================================
# 🧪 文字起こし（ストレージ対応 / AI実行 + busy 記録 / 正本経由）
#
# 目的：
#   (A) 既存ジョブ（Storages/.../minutes_app/.../job_.../split/）のチャンクを選び、
#       common_lib.ai 正本（routing.transcribe_audio）経由で連続文字起こし。
#   (B) 新規アップロード（drop）した複数音声を「split/ として」新規 job に保存し、
#       その順番のまま連続文字起こし。
#
# UI 方針（指定 by Y MAEDA）：
#   ①ジョブ選択 → メイン（中央）
#   ②チャンク選択・各種設定 → サイドバー
#   実行ボタン → メイン（使いやすさ優先）
#   結果表示 → メイン
#
# テンプレ準拠：
# - page_session_heartbeat をログイン判定の正本とする（require_login は使わない）
# - busy_run（ai_runs.db）で実行を必ず記録（バッチ=1run）
# - AI実行は common_lib.ai.routing.transcribe_audio（正本）を必ず使用
# - pages で APIキー取得 / requests / genai 直叩きは禁止
# - pages 側で「推計」しない（tokens推定・分単価換算は禁止）
#   ※ cost/usage は TranscribeResult が返せる範囲で busy/UI に反映するのみ
# - busy への反映は「最後に1回だけ（合計）」を基本とする
#
# 注意：
# - st.button()/st.download_button() に width 引数は使わない（環境で未対応）
# ============================================================

# ============================================================
# 標準ライブラリ
# ============================================================
from __future__ import annotations

import io
import json
import re
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# ============================================================
# サードパーティ
# ============================================================
import streamlit as st

# ============================================================
# アプリ内モジュール（ページ資産）
# ============================================================
from lib.audio import get_audio_duration_seconds

# ============================================================
# sys.path（common_lib を import できるように）
# ============================================================
import sys

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
# common_lib（正本：sessions / ui / busy / storage）
# ============================================================
from common_lib.ui.page_header import render_standard_page_header
from lib.transcribe.explanation import (
    render_transcribe_page_intro,
    render_transcribe_help_expander,
)

from common_lib.busy import busy_run, get_run
from common_lib.ui.time_format import format_jst_iso_ja
from common_lib.storage.external_ssd_root import resolve_storage_subdir_root

# ============================================================
# common_lib.ai（正本：routing）
# ============================================================
from common_lib.ai.routing import transcribe_audio
from common_lib.ai.types import TranscribeResult

# ============================================================
# common_lib.ai.models（正本：モデル一覧）
# ============================================================
from common_lib.ai.models import TRANSCRIBE_MODELS

# ============================================================
# cost UI（表示専用）
# - 計算しない（CostResult を受け取って表示するだけ）
# ============================================================
from common_lib.ai.costs.ui import render_transcribe_cost_summary

# ============================================================
# common_lib.ui（run summary：Transcribe 専用）
# ============================================================
from common_lib.ui.run_summary import render_run_summary_transcribe_compact_v2


# ============================================================
# ページ内：定数
# ============================================================
BRACKET_TAG_PATTERN = re.compile(r"【[^】]*】")

PROMPT_OPTIONS = [
    "",
    "出力に話者名や【】などのラベルを入れない。音声に無い単語は書かない。",
    "人名やプロジェクト名は正確に出力してください。専門用語はカタカナで。",
    "句読点を正しく付与し、自然な文章にしてください。",
]

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".mp4", ".webm", ".ogg"}

# ============================================================
# ページ設定（必須・統一）
# ============================================================
st.set_page_config(page_title="文字起こし — Storage Jobs", page_icon="🧪", layout="wide")


# ============================================================
# 共通ヘッダー
# - settings.toml から BANNER_KEY を取得
# - banner / theme / intro CSS を描画
# - page_session_heartbeat を実行
# - title / subtitle / ログイン状態を描画
# ============================================================
sub, theme, BANNER_KEY, settings = render_standard_page_header(
    st_module=st,
    projects_root=PROJECTS_ROOT,
    app_dir=APP_DIR,
    app_name=APP_NAME,
    page_name=PAGE_NAME,
    title="🎙️ 文字起こし",
    subtitle_text="音声ファイルを文字起こし",
    default_banner_key="light_green",
)

# ============================================================
# ページ説明
# ============================================================
render_transcribe_page_intro()

# ============================================================
# 詳細説明
# ============================================================
render_transcribe_help_expander(
    theme=theme,
    banner_key=BANNER_KEY,
)

# ============================================================
# Storage root（PROJECTS_ROOT 基準）
# ============================================================
STORAGE_ROOT = resolve_storage_subdir_root(
    PROJECTS_ROOT,
    subdir="Storages",
)

# ============================================================
# セッションキー（最低限）
# ============================================================
K_LAST_RUN_ID = f"{PAGE_NAME}__last_run_id"
K_LAST_RUN_ACTION = f"{PAGE_NAME}__last_run_action"
K_LAST_RESULT = f"{PAGE_NAME}__last_result"

st.session_state.setdefault(K_LAST_RUN_ID, "")
st.session_state.setdefault(K_LAST_RUN_ACTION, "")
st.session_state.setdefault(K_LAST_RESULT, None)

# ------------------------------------------------------------
# drop（アップロード）用：session keys（ページ専用・rerun耐性）
# ------------------------------------------------------------
K_UP_LAST_SIG = f"{PAGE_NAME}_upload_last_sig"
K_UP_JOB_ID = f"{PAGE_NAME}_upload_job_id"
K_UP_JOB_ROOT = f"{PAGE_NAME}_upload_job_root"
K_UP_LOCKED = f"{PAGE_NAME}_upload_job_locked"

st.session_state.setdefault(K_UP_LAST_SIG, None)
st.session_state.setdefault(K_UP_JOB_ID, None)
st.session_state.setdefault(K_UP_JOB_ROOT, None)
st.session_state.setdefault(K_UP_LOCKED, False)

# ============================================================
# ユーザー名のフォルダ安全化（pages/20 と合わせる）
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
# テキスト後処理（任意）
# ============================================================
def strip_bracket_tags(text: str) -> str:
    if not text:
        return text
    return BRACKET_TAG_PATTERN.sub("", text)

# ============================================================
# Job / split スキャン
# ============================================================
@dataclass
class JobInfo:
    date_dir: str
    job_id: str
    job_dir: Path
    split_dir: Path
    transcript_dir: Path
    transcript_marked_dir: Path
    logs_dir: Path
    job_json: Optional[dict]

    @property
    def label(self) -> str:
        base = f"{self.date_dir}/{self.job_id}"
        try:
            if self.job_json:
                orig = self.job_json.get("paths", {}).get("original")
                if orig:
                    return f"{base}  （original: {Path(orig).name}）"
        except Exception:
            pass
        return base

# ============================================================
# ファイルI/O（ページ内）
# ============================================================
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def append_log(log_path: Path, msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_mkdir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")

def read_json_if_exists(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

def scan_jobs(user_root: Path) -> List[JobInfo]:
    jobs: List[JobInfo] = []
    if not user_root.exists():
        return jobs

    for date_dir in sorted([d for d in user_root.iterdir() if d.is_dir()], reverse=True):
        for job_dir in sorted(
            [j for j in date_dir.iterdir() if j.is_dir() and j.name.startswith("job_")],
            reverse=True,
        ):
            split_dir = job_dir / "split"
            transcript_dir = job_dir / "transcript"
            transcript_marked_dir = job_dir / "transcript_marked"
            logs_dir = job_dir / "logs"
            job_json = read_json_if_exists(job_dir / "job.json")

            jobs.append(
                JobInfo(
                    date_dir=date_dir.name,
                    job_id=job_dir.name,
                    job_dir=job_dir,
                    split_dir=split_dir,
                    transcript_dir=transcript_dir,
                    transcript_marked_dir=transcript_marked_dir,
                    logs_dir=logs_dir,
                    job_json=job_json,
                )
            )
    return jobs

def list_split_audio_files(split_dir: Path) -> List[Path]:
    if not split_dir.exists():
        return []
    out: List[Path] = []
    for p in sorted(split_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            out.append(p)
    return out

def now_job_id() -> str:
    return "job_" + datetime.now().strftime("%Y%m%d_%H%M%S")

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

def save_text(p: Path, text: str) -> None:
    safe_mkdir(p.parent)
    p.write_text(text, encoding="utf-8")

def safe_filename(name: str) -> str:
    s = (name or "audio").strip()
    s = s.replace("\\", "_").replace("/", "_").replace(":", "_")
    s = re.sub(r"\s+", " ", s)
    return s

# ============================================================
# 古いジョブを物理的に消去（直近5件だけ残す）
# - ページ読み込み時に自動実行
# ============================================================
MAX_JOBS = 5
jobs_all_for_cleanup = scan_jobs(USER_ROOT)
jobs_delete_for_cleanup = jobs_all_for_cleanup[MAX_JOBS:]

for j in jobs_delete_for_cleanup:
    try:
        shutil.rmtree(j.job_dir)
        date_dir = j.job_dir.parent
        if date_dir.exists() and not any(date_dir.iterdir()):
            date_dir.rmdir()
    except Exception as e:
        st.warning(f"削除失敗: {j.job_dir} ({e})")

# ============================================================
# 入力方式（事故の少ない radio）
# ============================================================
st.subheader("① 音声ファイルの設定")
st.caption("音声ファイルを読み込む先をここで指定します．"
    "「音声ファイル分割」を行った時にサーバー内部に自動保存されたファイルを"
    "使用するときは「既存ジョブ」を選択してください．")
st.caption("**音声ファイル分割から議事録を作成する一連の作業**を，ここでは「ジョブ」と呼んでいます．"
           "ここでの処理では，「ジョブ」という言葉で，分割された**音声ファイルの集合体**を指していることになります．")
input_mode = st.radio(
    "音声ファイルを読み込む先をここで指定します．",
    options=["既存ジョブ（サーバー内部のストレージより読み込み）", "新規アップロード"],
    index=0,
    label_visibility="collapsed",
    #help="「新規アップロード」は、アップロードした複数ファイルをストレージに保存し、その順番で連続文字起こしします。",
)

# ============================================================
# (A) 既存ジョブ（storage）を選ぶ
# ============================================================
picked_job: Optional[JobInfo] = None
split_files: List[Path] = []

if input_mode.startswith("既存ジョブ"):
    jobs = scan_jobs(USER_ROOT)

    st.markdown("#### ジョブ選択（サーバー内部の保存ファイル）")
    st.write("直近の5つのジョブより古いものは自動的に消去されます")

    if not jobs:
        st.warning(f"ジョブが見つかりません: {USER_ROOT}\n先に「音声分割（storage）」で job を作成してください。")
        st.stop()

    job_labels = [j.label for j in jobs]
    picked_label = st.radio(
        "対象ジョブ",
        options=job_labels,
        index=0,
        help="音声分割で作成された job フォルダを選びます（split/ を参照）。",
    )

    picked_job = jobs[job_labels.index(picked_label)]
    split_files = list_split_audio_files(picked_job.split_dir)

    st.caption(f"選択中の job: {picked_job.job_dir}")
    if not split_files:
        st.warning("この job には split/ の音声チャンクが見つかりません。")
        st.stop()

# ============================================================
# (B) 新規アップロード（drop）→ 新規 job 作成 → split/ に保存
# ============================================================
else:
    st.subheader("新規アップロード（drop）")

    uploaded_files = st.file_uploader(
        "音声ファイルをドロップ/選択（複数可・アップロード順＝連続順）",
        type=["mp3", "wav", "m4a", "mp4", "webm", "ogg"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("音声ファイルを1つ以上アップロードしてください。")
        st.stop()

    sig_parts = [f"{getattr(f, 'name', 'file')}:{getattr(f, 'size', 0)}" for f in uploaded_files]
    upload_sig = "|".join(sig_parts)

    if st.session_state.get(K_UP_LAST_SIG) != upload_sig:
        st.session_state[K_UP_LAST_SIG] = upload_sig
        st.session_state[K_UP_JOB_ID] = None
        st.session_state[K_UP_JOB_ROOT] = None
        st.session_state[K_UP_LOCKED] = False

    st.caption("※ アップロード順をそのまま連続処理の順番として使います。")
    st.write({"files": [getattr(f, "name", "(no name)") for f in uploaded_files]})

    existing_job_root = st.session_state.get(K_UP_JOB_ROOT)
    existing_job_id = st.session_state.get(K_UP_JOB_ID)

    if existing_job_root and existing_job_id and st.session_state.get(K_UP_LAST_SIG) == upload_sig:
        job_root = Path(existing_job_root)
        job_id = str(existing_job_id)

        picked_job = JobInfo(
            date_dir=job_root.parent.name,
            job_id=job_id,
            job_dir=job_root,
            split_dir=job_root / "split",
            transcript_dir=job_root / "transcript",
            transcript_marked_dir=job_root / "transcript_marked",
            logs_dir=job_root / "logs",
            job_json=read_json_if_exists(job_root / "job.json"),
        )
        split_files = list_split_audio_files(picked_job.split_dir)

        st.success(f"✅ 作成済みジョブを復元しました: {job_root}")
        st.caption("※ rerun してもこの状態は維持されます（「文字起こしを実行」を押してOK）。")

        if not split_files:
            st.error("split/ に音声が見つかりません。保存を確認してください。")
            st.stop()

    else:
        create_job_clicked = st.button(
            "📦 アップロードを split/ として保存して新規ジョブを作成",
            type="primary",
            disabled=bool(st.session_state.get(K_UP_LOCKED, False)),
            help="押した時点で Storages に job_YYYYMMDD_HHMMSS を作り、split/ に保存します（事故防止のためボタン確定制）。",
        )

        if not create_job_clicked:
            st.stop()

        st.session_state[K_UP_LOCKED] = True

        try:
            with st.spinner("新規ジョブを作成して split/ に保存しています…"):
                today_dir = datetime.now().strftime("%Y-%m-%d")
                job_id = now_job_id()
                job_root = USER_ROOT / today_dir / job_id

                split_dir = job_root / "split"
                transcript_dir = job_root / "transcript"
                transcript_marked_dir = job_root / "transcript_marked"
                logs_dir = job_root / "logs"
                for d in (split_dir, transcript_dir, transcript_marked_dir, logs_dir):
                    safe_mkdir(d)

                job_log_path = logs_dir / "process.log"
                append_log(job_log_path, "UPLOAD->JOB START")
                append_log(job_log_path, f"user_display={current_user}")
                append_log(job_log_path, f"user_dir={USERNAME_DIR}")
                append_log(job_log_path, f"job={job_root}")

                split_index: List[dict] = []

                for i, uf in enumerate(uploaded_files, start=1):
                    name = safe_filename(getattr(uf, "name", f"audio_{i}"))
                    out_name = f"{i:03d}_{name}"
                    out_path = split_dir / out_name
                    b = uf.getvalue()
                    out_path.write_bytes(b)

                    split_index.append(
                        {
                            "order": i,
                            "original_name": name,
                            "saved_name": out_name,
                            "saved_path": str(out_path),
                            "size_bytes": int(len(b)),
                        }
                    )
                    append_log(job_log_path, f"saved split -> {out_name} ({len(b)} bytes)")

                job_json = {
                    "job_id": job_id,
                    "user": str(current_user),
                    "user_dir": USERNAME_DIR,
                    "date": today_dir,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "source": "upload_drop",
                    "paths": {
                        "job_root": str(job_root),
                        "split_dir": str(split_dir),
                        "transcript_dir": str(transcript_dir),
                        "transcript_marked_dir": str(transcript_marked_dir),
                        "logs_dir": str(logs_dir),
                    },
                    "status": {
                        "split": "done",
                        "transcribe": "not_started",
                        "merge": "not_started",
                        "dedup": "not_started",
                    },
                    "split_index": split_index,
                }
                (job_root / "job.json").write_text(
                    json.dumps(job_json, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                append_log(job_log_path, "UPLOAD->JOB DONE")

            st.session_state[K_UP_JOB_ID] = job_id
            st.session_state[K_UP_JOB_ROOT] = str(job_root)

            st.success(f"✅ 新規ジョブを作成しました: {job_root}")
            st.caption(f"保存先: {job_root}")

            picked_job = JobInfo(
                date_dir=today_dir,
                job_id=job_id,
                job_dir=job_root,
                split_dir=job_root / "split",
                transcript_dir=job_root / "transcript",
                transcript_marked_dir=job_root / "transcript_marked",
                logs_dir=job_root / "logs",
                job_json=job_json,
            )
            split_files = list_split_audio_files(picked_job.split_dir)

            if not split_files:
                st.error("split/ への保存後に音声が見つかりません。拡張子や保存を確認してください。")
                st.stop()

        finally:
            st.session_state[K_UP_LOCKED] = False

# ============================================================
# サイドバー：チャンク選択・モデル・パラメータ
# ============================================================
with st.sidebar:
    # ------------------------------------------------------------
    # チャンク選択（split/）
    # ------------------------------------------------------------
    st.header("音声ファイル選択（split/）")

    options = [p.name for p in split_files]
    selected_names = st.multiselect(
        "処理する音声ファイル（選択順に連続文字起こし）",
        options=options,
        default=options,
        help="drop の場合は 001_,002_... の番号順がそのまま連続順になります。",
    )

    st.divider()

    # ------------------------------------------------------------
    # モデル（正本：TRANSCRIBE_MODELS）
    # ------------------------------------------------------------
    st.header("モデル")
    st.session_state.setdefault("model_picker", "whisper-1")
    st.radio(
        "モデル",
        options=TRANSCRIBE_MODELS,
        key="model_picker",
    )
    model = str(st.session_state["model_picker"])

    st.divider()

    # ------------------------------------------------------------
    # 返却形式・言語・プロンプト
    # ------------------------------------------------------------
    st.header("返却形式・言語・プロンプト")
    fmt = st.selectbox("返却形式（OpenAI response_format）", ["json", "text", "srt", "vtt"], index=0)
    language = st.text_input("言語コード（未指定なら自動判定）", value="ja")
    prompt_hint = st.selectbox("Transcribeプロンプト（省略可）", options=PROMPT_OPTIONS, index=0)
    do_strip_brackets = st.checkbox("書き起こし後に【…】を除去する", value=True)

# ============================================================
# メイン：実行ボタン
# ============================================================
st.divider()
st.subheader("② 文字起こしの実行")
st.caption(f"処理対象ジョブ: {picked_job.job_dir if picked_job else '(none)'}")
go = st.button("▶️ 文字起こしを実行 + ストレージ保存", type="primary")
out_area = st.container()

# ============================================================
# 実行部（AI実行 + busy 記録）
# ============================================================
if go:
    # ------------------------------------------------------------
    # ガード（選択）
    # ------------------------------------------------------------
    if not selected_names:
        st.warning("音声ファイルを1つ以上選んでください。")
        st.stop()

    assert picked_job is not None

    # ------------------------------------------------------------
    # 保存先
    # ------------------------------------------------------------
    safe_mkdir(picked_job.transcript_dir)
    safe_mkdir(picked_job.transcript_marked_dir)
    safe_mkdir(picked_job.logs_dir)

    job_log_path = picked_job.logs_dir / "process.log"
    append_log(job_log_path, "TRANSCRIBE START")
    append_log(job_log_path, f"job={picked_job.job_dir}")
    append_log(job_log_path, f"model={model} fmt={fmt} language={language!r}")
    append_log(job_log_path, f"selected={selected_names}")

    # ------------------------------------------------------------
    # 対象チャンク
    # ------------------------------------------------------------
    name_to_path = {p.name: p for p in split_files}
    targets = [name_to_path[n] for n in selected_names if n in name_to_path]
    if not targets:
        st.error("選択音声ファイルが見つかりません。split/ の状態を確認してください。")
        st.stop()

    provider = "gemini" if str(model).startswith("gemini") else "openai"
    action_name = "transcribe_continuous"

    meta_start = {
        "feature": "minutes_transcribe_storage",
        "action": action_name,
        "input_mode": ("job" if input_mode.startswith("既存") else "upload"),
        "job_dir": str(picked_job.job_dir),
        "chunks_selected": len(targets),
        "model": model,
        "provider": provider,
        "response_format": fmt,
        "language": language,
        "prompt_chars": len(prompt_hint or ""),
        "strip_brackets": bool(do_strip_brackets),
    }

    # ------------------------------------------------------------
    # busy（ai_runs.db）：1バッチ=1run
    # - cost は「合計を最後に1回だけ」
    # ------------------------------------------------------------
    try:
        with busy_run(
            projects_root=PROJECTS_ROOT,
            user_sub=str(sub),
            app_name=str(APP_NAME),
            page_name=str(PAGE_NAME),
            task_type="transcribe",
            provider=provider,
            model=str(model),
            meta=meta_start,
        ) as br:
            with st.spinner("文字起こし実行中…"):
                progress = st.progress(0, text="準備中…")

                per_file_results: List[dict] = []
                combined_parts: List[str] = []
                total_elapsed = 0.0

                # ------------------------------------------------------------
                # 合計（推計しない：返ってきた cost を足すだけ）
                # - 1つでも cost が None の回があれば合計は None
                # ------------------------------------------------------------
                total_usd: Optional[float] = 0.0
                total_jpy: Optional[float] = 0.0

                for idx, path in enumerate(targets, start=1):
                    progress.progress((idx - 1) / len(targets), text=f"{idx}/{len(targets)} 処理中: {path.name}")

                    file_bytes = path.read_bytes()
                    mime = guess_mime_from_suffix(path)

                    # ------------------------------------------------------------
                    # audio_seconds（主役：可能なら計測して渡す）
                    # ------------------------------------------------------------
                    audio_sec: Optional[float] = None
                    try:
                        audio_sec = float(get_audio_duration_seconds(io.BytesIO(file_bytes)))
                    except Exception:
                        audio_sec = None

                    # ------------------------------------------------------------
                    # routing 正本：transcribe_audio
                    # ------------------------------------------------------------
                    t0 = time.perf_counter()
                    extra: Dict[str, Any] = {
                        "page": str(PAGE_NAME),
                        "job_dir": str(picked_job.job_dir),
                        "chunk_name": str(path.name),
                    }

                    lang_arg = (language.strip() if language and language.strip() else None)
                    prompt_arg = (prompt_hint.strip() if prompt_hint and prompt_hint.strip() else None)

                    tr: TranscribeResult = transcribe_audio(
                        provider=provider,
                        model=str(model),
                        audio_bytes=file_bytes,
                        mime_type=mime,
                        filename=path.name,
                        audio_seconds=float(audio_sec) if audio_sec is not None else None,
                        response_format=str(fmt),
                        language=lang_arg,
                        prompt=prompt_arg,
                        timeout_sec=600,
                        extra=extra,
                    )

                    elapsed = time.perf_counter() - t0
                    total_elapsed += elapsed

                    # ------------------------------------------------------------
                    # テキスト後処理（任意）
                    # ------------------------------------------------------------
                    text = tr.text or ""
                    if do_strip_brackets and text:
                        text = strip_bracket_tags(text)

                    # ------------------------------------------------------------
                    # cost 合算（推計しない）
                    # ------------------------------------------------------------
                    if tr.cost is not None:
                        if total_usd is not None:
                            total_usd += float(tr.cost.usd)
                        if total_jpy is not None:
                            total_jpy += float(tr.cost.jpy)
                    else:
                        total_usd = None
                        total_jpy = None

                    # ------------------------------------------------------------
                    # 保存（個別）
                    # ------------------------------------------------------------
                    out_txt = picked_job.transcript_dir / f"{idx:03d}_{path.stem}.txt"
                    save_text(out_txt, text)
                    append_log(job_log_path, f"SAVED transcript {path.name} -> {out_txt.name}")

                    # ------------------------------------------------------------
                    # 画面表示（1チャンク）
                    # ------------------------------------------------------------
                    with out_area:
                        st.markdown(f"### 📁 {idx}. {path.name}")
                        st.text_area("テキスト（個別）", value=text, height=220, key=f"ta_{idx}")

                        # render_transcribe_cost_summary(
                        #     title="概算（この音声ファイル）",
                        #     model=str(model),
                        #     audio_sec=audio_sec,
                        #     cost=tr.cost,
                        #     notes=None,
                        # )

                        render_transcribe_cost_summary(
                            title="概算（この音声ファイル）",
                            model=str(model),
                            audio_sec=audio_sec,
                            in_tokens=(
                                int(tr.usage.input_tokens)
                                if getattr(tr, "usage", None) is not None
                                and tr.usage.input_tokens is not None
                                else None
                            ),
                            out_tokens=(
                                int(tr.usage.output_tokens)
                                if getattr(tr, "usage", None) is not None
                                and tr.usage.output_tokens is not None
                                else None
                            ),
                            elapsed_sec=float(elapsed),
                            cost=tr.cost,
                            notes=None,
                        )

                        # ============================================================
                        # compact meta
                        # ============================================================
                        meta_cols = st.columns([2, 2, 2])

                        with meta_cols[0]:
                            st.caption(f"モデル：{model}")

                        with meta_cols[1]:
                            st.caption(f"処理時間：{elapsed:.2f} 秒")

                        with meta_cols[2]:
                            if audio_sec is not None:
                                st.caption(f"音声長：{audio_sec:.1f} 秒")
                            else:
                                st.caption("音声長：—")

                        #st.caption(f"request-id：{tr.request_id or '—'}")
                        #st.caption(f"保存先：{str(out_txt)}")

                    per_file_results.append(
                        dict(
                            name=path.name,
                            text=text,
                            sec=audio_sec,
                            elapsed=elapsed,
                            out_txt=str(out_txt),
                            request_id=tr.request_id,
                            cost=tr.cost,
                        )
                    )

                    # ------------------------------------------------------------
                    # 連結テキスト
                    # ------------------------------------------------------------
                    combined_parts.append(text or "")
                    if idx < len(targets):
                        combined_parts.append(f"\n\n----- ここがつなぎ目です（{path.name} と次のファイルの間）-----\n\n")

                progress.progress(1.0, text="完了")

                # ------------------------------------------------------------
                # combined 保存
                # ------------------------------------------------------------
                combined_text = "".join(combined_parts)

                if targets:
                    first_stem = targets[0].stem
                    base_name = first_stem.split("_part", 1)[0] if "_part" in first_stem else first_stem
                else:
                    base_name = "audio"

                combined_dir = picked_job.job_dir / "transcript_combined"
                safe_mkdir(combined_dir)

                combined_name = f"{base_name}_combined.txt"
                combined_txt_path = combined_dir / combined_name

                save_text(combined_txt_path, combined_text)
                append_log(job_log_path, f"SAVED combined -> {combined_txt_path}")
                append_log(job_log_path, "TRANSCRIBE DONE")

                # ------------------------------------------------------------
                # busy への反映（最後に1回だけ）
                # ------------------------------------------------------------
                if (total_usd is not None) and (total_jpy is not None):
                    br.set_cost(float(total_usd), float(total_jpy))

                    # ============================================================
                    # UI用：合計cost（推計しない）
                    # - 合計USD/JPYは「足した結果」
                    # - usd_jpy / fx_source は最後に取れた cost_obj の値を流用（換算はしない）
                    # ============================================================
                    try:
                        from common_lib.ai.types import CostResult  # 正本型

                        if last_cost_obj is not None:
                            st.session_state["total_cost_obj"] = CostResult(
                                usd=float(total_usd),
                                jpy=float(total_jpy),
                                usd_jpy=float(getattr(last_cost_obj, "usd_jpy")),
                                fx_source=str(getattr(last_cost_obj, "fx_source")),
                            )
                        else:
                            st.session_state["total_cost_obj"] = None
                    except Exception:
                        st.session_state["total_cost_obj"] = None


                    br.add_finish_meta(note="ok_cost_sum")
                else:
                    br.add_finish_meta(note="no_cost_sum")

                br.add_finish_meta(
                    total_elapsed_sec=round(float(total_elapsed), 3),
                    combined_path=str(combined_txt_path),
                    chunks_done=len(per_file_results),
                    total_usd=(round(float(total_usd), 6) if total_usd is not None else None),
                    total_jpy=(round(float(total_jpy), 2) if total_jpy is not None else None),
                )

                # ------------------------------------------------------------
                # ページ表示用 state
                # ------------------------------------------------------------
                st.session_state[K_LAST_RUN_ID] = br.run_id
                st.session_state[K_LAST_RUN_ACTION] = action_name
                st.session_state[K_LAST_RESULT] = {
                    "job_dir": str(picked_job.job_dir),
                    "combined_txt_path": str(combined_txt_path),
                    "combined_name": combined_name,
                    "chunks_done": len(per_file_results),
                }

                st.success(f"✅ ストレージに保存しました: {combined_txt_path.name}")
                st.caption(f"保存先: {combined_txt_path}")

                with out_area:
                    st.markdown("---")
                    st.subheader("🔗 連結テキスト（全音声ファイル）")
                    st.text_area("テキスト（連結済み）", value=combined_text, height=350)

                    st.download_button(
                        "🧩 連結テキスト（.txt）を（パソコンに）ダウンロード",
                        data=(combined_text or "").encode("utf-8"),
                        file_name="transcripts_combined.txt",
                        mime="text/plain",
                    )

                    st.subheader("📊 料金の概要（合算）")
                    if total_usd is None or total_jpy is None:
                        st.caption("※ cost が取得できない音声ファイルがあるため、合算は表示しません。")
                    else:
                        st.caption(f"合計（USD）：${float(total_usd):,.6f}")
                        st.caption(f"合計（JPY）：{float(total_jpy):,.2f} 円")

                st.session_state["transcribed_texts"] = [r["text"] for r in per_file_results]
                st.session_state["transcribed_text"] = combined_text
                st.session_state["picked_job_dir"] = str(picked_job.job_dir)
                st.session_state["combined_txt_path"] = str(combined_txt_path)

    except Exception as e:
        st.error(f"実行に失敗しました: {e}")
        st.stop()

# ============================================================
# 📊 実行サマリ（Transcribe：テンプレ準拠）
# - 音声時間 / 費用 / AI使用時間 を1ブロック表示
# ============================================================
raw_run_id: str = str(st.session_state.get(K_LAST_RUN_ID) or "")
last_run_id: str = raw_run_id.strip()
has_run: bool = bool(last_run_id)

if has_run:
    # ------------------------------------------------------------
    # 合計 cost（ページ内で足した合計を session_state に入れてある前提）
    # - もし未設定なら None（費用は出ない）
    # ------------------------------------------------------------
    total_cost = st.session_state.get("total_cost_obj", None)

    # ------------------------------------------------------------
    # 合計 audio_seconds（合計秒を足しているなら渡す、無ければ None）
    # ------------------------------------------------------------
    total_audio_sec = st.session_state.get("total_audio_sec", None)

    render_run_summary_transcribe_compact_v2(
        projects_root=PROJECTS_ROOT,
        run_id=last_run_id,
        model=str(model),
        audio_sec=(float(total_audio_sec) if isinstance(total_audio_sec, (int, float)) else None),
        cost=total_cost,
        note="",
        show_divider=True,
    )
