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
import re
import sys
from datetime import datetime
from pathlib import Path

# ============================================================
# サードパーティ
# ============================================================
import streamlit as st

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
# batch_processing（ページ外へ切り出した処理）
# ============================================================
from lib.batch_processing.constants import (
    DEFAULT_LANGUAGE,
    DEFAULT_SPEAKER_PROMPT_LEVEL,
    DEFAULT_TRANSCRIBE_MODEL,
)

from lib.batch_processing.io_utils import (
    append_log,
    read_text_safely,
)

from lib.batch_processing.paths import (
    create_job_root_for_uploaded_file,
    safe_filename,
)

from lib.batch_processing.audio_split import run_audio_split_step
from lib.batch_processing.transcribe_step import run_transcribe_step
from lib.batch_processing.speaker_step import run_speaker_step
from lib.batch_processing.overlap_step import run_overlap_step
from lib.batch_processing.job_status import update_job_status

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
    banner_key=BANNER_KEY,
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
# Sidebar
# ============================================================
with st.sidebar:
    # ------------------------------------------------------------
    # モデル設定
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # 言語
    # ------------------------------------------------------------
    st.header("言語")

    language = st.selectbox(
        "文字起こし言語コード",
        options=["ja", "en", ""],
        index=0 if DEFAULT_LANGUAGE == "ja" else 0,
        key=k("language"),
        help="空欄を選ぶと自動判定に近い扱いになります。",
    )

# ============================================================
# メインUI：アップロード
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
    # ------------------------------------------------------------
    # 実行結果保持
    # ------------------------------------------------------------
    all_final_texts: list[str] = []
    final_paths: list[Path] = []

    overall_progress = st.progress(0, text="一括処理を準備しています...")
    status_area = st.container()

    total_files = len(uploaded_files)

    # ------------------------------------------------------------
    # 入力ファイルごとの一括処理
    # ------------------------------------------------------------
    for file_index, uf in enumerate(uploaded_files, start=1):
        uploaded_name = safe_filename(getattr(uf, "name", f"audio_{file_index}.mp3"))
        uploaded_bytes = uf.getvalue()

        with status_area:
            st.markdown(f"#### 処理対象：{uploaded_name}")

        (
            job_id,
            job_root,
            original_dir,
            split_dir,
            transcript_dir,
            transcript_combined_dir,
            speaker_sep_dir,
            speaker_combined_dir,
        ) = create_job_root_for_uploaded_file(
            user_root=USER_ROOT,
            uploaded_name=uploaded_name,
        )

        log_path = job_root / "logs" / "process.log"


        # ============================================================
        # DEBUG : job create
        # ============================================================
        # append_log(log_path, "DEBUG JOB START")
        # append_log(log_path, f"job_id={job_id}")
        # append_log(log_path, f"job_root={job_root}")
        # append_log(log_path, f"uploaded_name={uploaded_name}")

        # print("=" * 60)
        # print("BATCH35_JOB_DEBUG")
        # print(f"uploaded_name = {uploaded_name}")
        # print(f"job_id        = {job_id}")
        # print(f"job_root      = {job_root}")
        # print("=" * 60)
        # ============================================================
        # DEBUG END
        # ============================================================

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
                text=f"音声分割中：{uploaded_name}",
            )

            # ============================================================
            # 音声分割実行
            # - 長時間処理のため spinner を表示する
            # ============================================================
            with st.spinner("音声分割中..."):
                split_paths = run_audio_split_step(
                    uploaded_name=uploaded_name,
                    uploaded_bytes=uploaded_bytes,
                    job_id=job_id,
                    job_root=job_root,
                    original_dir=original_dir,
                    split_dir=split_dir,
                    log_path=log_path,
                    current_user=str(current_user),
                    username_dir=str(USERNAME_DIR),
                    transcript_model=str(transcribe_model),
                )

            # ============================================================
            # DEBUG : split input
            # ============================================================
            # print("=" * 60)
            # print("BATCH35_SPLIT_DEBUG")
            # print(f"job_root  = {job_root}")
            # print(f"split_dir = {split_dir}")
            # print("=" * 60)
            # ============================================================
            # DEBUG END
            # ============================================================

            update_job_status(job_root, "split", "done")

            with status_area:
                st.success(f"✓ 音声分割完了（{len(split_paths)}チャンク）")

            # ------------------------------------------------------------
            # 2. 文字起こし
            # ------------------------------------------------------------
            overall_progress.progress(
                int(((file_index - 1 + 0.25) / total_files) * 100),
                text=f"文字起こし中：{uploaded_name}",
            )

            # ============================================================
            # progress UI : transcribe chunks
            # ============================================================
            transcribe_progress_area = st.empty()

            def show_transcribe_progress(chunk_i: int, chunk_total: int, chunk_path: Path) -> None:
                transcribe_progress_area.info(
                    f"文字起こし中：chunk {chunk_i} / {chunk_total}　{chunk_path.name}"
                )           

            # _combined_transcript_path, transcribe_chunk_results = run_transcribe_step(
            #     split_paths=split_paths,
            #     job_root=job_root,
            #     transcript_dir=transcript_dir,
            #     transcript_model=str(transcribe_model),
            #     language=str(language),
            #     log_path=log_path,
            #     projects_root=PROJECTS_ROOT,
            #     user_sub=str(sub),
            #     app_name=str(APP_NAME),
            #     page_name=str(PAGE_NAME),
            #     progress_callback=show_transcribe_progress,
            # )

            with st.spinner("文字起こし中..."):
                _combined_transcript_path, transcribe_chunk_results = run_transcribe_step(
                    split_paths=split_paths,
                    job_root=job_root,
                    transcript_dir=transcript_dir,
                    transcript_model=str(transcribe_model),
                    language=str(language),
                    log_path=log_path,
                    projects_root=PROJECTS_ROOT,
                    user_sub=str(sub),
                    app_name=str(APP_NAME),
                    page_name=str(PAGE_NAME),
                    progress_callback=show_transcribe_progress,
                )

            update_job_status(job_root, "transcribe", "done")

            transcribe_progress_area.success("✓ 文字起こし完了")

            # ============================================================
            # DEBUG : transcribe chunk texts
            # - 各チャンクの文字起こし結果を画面で確認する
            # ============================================================
            with st.expander(f"DEBUG：文字起こし結果（{uploaded_name}）", expanded=True):
                for item in transcribe_chunk_results:
                    idx = item.get("index", "—")
                    source_name = item.get("source_name", "—")
                    text = str(item.get("text", "") or "")
                    text_len = int(item.get("text_len", 0) or 0)
                    elapsed_sec = item.get("elapsed_sec", None)
                    audio_sec = item.get("audio_sec", None)

                    st.markdown(f"##### chunk {idx}: `{source_name}`")

                    elapsed_label = (
                        f"{float(elapsed_sec):.2f}s"
                        if isinstance(elapsed_sec, (int, float))
                        else "—"
                    )
                    audio_label = (
                        f"{float(audio_sec):.1f}s"
                        if isinstance(audio_sec, (int, float))
                        else "—"
                    )

                    st.caption(
                        f"text_len={text_len}　"
                        f"elapsed={elapsed_label}　"
                        f"audio_sec={audio_label}"
                    )

                    if text_len <= 0:
                        st.error("このチャンクの文字起こし結果は空です。")
                    else:
                        st.text_area(
                            "文字起こし結果",
                            value=text,
                            height=180,
                            key=k(f"debug_transcribe_text_{file_index}_{idx}"),
                        )
            # ============================================================
            # DEBUG END
            # ============================================================

            # ============================================================
            # DEBUG : stop after transcribe
            # ============================================================
            # st.warning("DEBUG: 文字起こし完了で停止")
            # st.stop()
            # ============================================================
            # DEBUG END
            # ============================================================

            # ------------------------------------------------------------
            # 3. 話者分離
            # ------------------------------------------------------------
            overall_progress.progress(
                int(((file_index - 1 + 0.50) / total_files) * 100),
                text=f"話者分離中：{uploaded_name}",
            )

            # ============================================================
            # progress UI : speaker chunks
            # ============================================================
            speaker_progress_area = st.empty()

            def show_speaker_progress(chunk_i: int, chunk_total: int, chunk_path: Path) -> None:
                speaker_progress_area.info(
                    f"話者分離中：chunk {chunk_i} / {chunk_total}　{chunk_path.name}"
                )

            with st.spinner("話者分離中..."):
                speaker_combined_path = run_speaker_step(
                    job_root=job_root,
                    speaker_sep_dir=speaker_sep_dir,
                    speaker_combined_dir=speaker_combined_dir,
                    speaker_model_key=str(speaker_model_key),
                    prompt_level=DEFAULT_SPEAKER_PROMPT_LEVEL,
                    log_path=log_path,
                    projects_root=PROJECTS_ROOT,
                    user_sub=str(sub),
                    app_name=str(APP_NAME),
                    page_name=str(PAGE_NAME),
                    progress_callback=show_speaker_progress,
                )


            update_job_status(job_root, "speaker_separate", "done")

            speaker_progress_area.success("✓ 話者分離完了")

            # ------------------------------------------------------------
            # 4. 重複検出
            # ------------------------------------------------------------
            overall_progress.progress(
                int(((file_index - 1 + 0.75) / total_files) * 100),
                text=f"重複箇所検出中：{uploaded_name}",
            )

            marked_path = run_overlap_step(
                job_root=job_root,
                combined_path=speaker_combined_path,
                log_path=log_path,
                current_user=str(current_user),
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