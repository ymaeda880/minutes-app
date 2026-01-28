# -*- coding: utf-8 -*-
# minutes_app/pages/20_音声ファイル分割_ストレージ対応.py
# ============================================================
# 🎧 音声ファイル分割（ストレージ対応 / 非AI / セッション記録のみ）
#
# 目的：
# - page_session_heartbeat により利用状況を記録（ログイン判定もここが正本）
# - 音声（MP3/WAV/MP4）を一定長さで分割し、隣接チャンクにオーバーラップを付与
# - 「保存」ボタン押下時のみ Storages に保存（original / split / logs / job.json）
# - ZIP を作成し、（PCへ）ダウンロードもできるようにする
#
# 方針：
# - AIは一切使わない（busy / 実行時間測定は不要）
# - st.form は使わない
# - use_container_width は使わない
# - テンプレ準拠：page_session_heartbeat / タイトル帯 / 入力→実行→結果 の構造
# ============================================================

from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st
from pydub import AudioSegment

from lib.explanation import render_audio_split_expander

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
from common_lib.sessions.page_entry import page_session_heartbeat
from common_lib.ui.ui_basics import subtitle
from common_lib.storage.external_ssd_root import resolve_storage_subdir_root
from common_lib.ui.banner_lines import render_banner_line_by_key

# ============================================================
# Storage config（PROJECTS_ROOT 基準）
# ============================================================
STORAGE_ROOT = resolve_storage_subdir_root(
    PROJECTS_ROOT,
    subdir="Storages",
)

# ============================================================
# 関数群（ユーティリティ）
# ============================================================
def _sanitize_username_for_path(username: str) -> str:
    """
    Storages/<user>/... の <user> に使う安全な文字列にする。
    """
    u = (username or "").strip()
    if not u:
        return "anonymous"
    u = re.sub(r"[^0-9A-Za-z_-]+", "_", u).strip("_")
    return u or "anonymous"


def now_job_id() -> str:
    """
    job_YYYYmmdd_HHMMSS 形式のジョブID
    """
    return "job_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def hhmmss(ms: int) -> str:
    """
    ミリ秒 → HH:MM:SS
    """
    return str(timedelta(milliseconds=ms)).split(".")[0]


def safe_mkdir(p: Path) -> None:
    """
    mkdir -p
    """
    p.mkdir(parents=True, exist_ok=True)


def append_log(log_path: Path, msg: str) -> None:
    """
    logs/process.log に追記
    """
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_mkdir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def write_json(p: Path, obj: dict) -> None:
    """
    JSON（整形）で保存
    """
    safe_mkdir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _guess_format_from_suffix(suffix: str) -> str:
    """
    pydub / ffmpeg の format 指定用に拡張子から推定
    """
    suf = (suffix or "").lower()
    if suf == ".mp3":
        return "mp3"
    if suf == ".wav":
        return "wav"
    if suf in {".mp4", ".m4a"}:
        return "mp4"
    return suf.lstrip(".")


def split_with_overlap(
    audio: AudioSegment,
    chunk_ms: int,
    overlap_ms: int,
    fade_ms: int,
    absorb_tiny_tail: bool,
):
    """
    overlap を含めて分割。最後の短すぎる尻尾は吸収（オプション）。
    戻り値: list[dict] with keys {start_ms, end_ms, segment(AudioSegment)}
    """
    results = []
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

        # 短すぎる最後の尻尾（例： overlap 以下）を前チャンクに吸収して重複を増やさない
        if absorb_tiny_tail and start > 0 and end == n:
            tail_len = end - start
            if tail_len < overlap_ms:
                prev = results[-1]
                prev["end_ms"] = n
                prev["segment"] = audio[prev["start_ms"] : n]
                break

        # フェード（任意）
        if fade_ms > 0 and len(seg) > fade_ms * 2:
            seg = seg.fade_in(fade_ms).fade_out(fade_ms)

        results.append({"start_ms": start, "end_ms": end, "segment": seg})

        if end == n:
            break
        start += step

    return results


# ============================================================
# セッションキー（ジョブ固定＆結果保持）
# ============================================================
K_LAST_UPLOAD_SIG = f"{PAGE_NAME}__last_upload_sig"
K_JOB_ID = f"{PAGE_NAME}__job_id"
K_JOB_ROOT = f"{PAGE_NAME}__job_root"
K_JOB_LOCKED = f"{PAGE_NAME}__job_locked"

K_AUDIO_LEN_MS = f"{PAGE_NAME}__audio_len_ms"
K_PARTS = f"{PAGE_NAME}__parts"
K_ROWS = f"{PAGE_NAME}__rows"

K_ZIP_BYTES = f"{PAGE_NAME}__zip_bytes"
K_ZIP_NAME = f"{PAGE_NAME}__zip_name"

# 既定値は「型」を維持して初期化（Noneで上書きしない）
st.session_state.setdefault(K_LAST_UPLOAD_SIG, None)
st.session_state.setdefault(K_JOB_ID, None)
st.session_state.setdefault(K_JOB_ROOT, None)
st.session_state.setdefault(K_JOB_LOCKED, False)

st.session_state.setdefault(K_AUDIO_LEN_MS, None)
st.session_state.setdefault(K_PARTS, None)
st.session_state.setdefault(K_ROWS, None)

st.session_state.setdefault(K_ZIP_BYTES, None)
st.session_state.setdefault(K_ZIP_NAME, None)


def _reset_job_lock_on_new_upload(upload_sig: str) -> None:
    """
    アップロードが変わったらジョブ固定を解除する（“2つできる”防止の制御をリセット）
    """
    if st.session_state[K_LAST_UPLOAD_SIG] != upload_sig:
        st.session_state[K_LAST_UPLOAD_SIG] = upload_sig
        st.session_state[K_JOB_ID] = None
        st.session_state[K_JOB_ROOT] = None
        st.session_state[K_JOB_LOCKED] = False
        st.session_state[K_ZIP_BYTES] = None
        st.session_state[K_ZIP_NAME] = None


# ============================================================
# ページ設定（必須・統一）
# ============================================================
st.set_page_config(
    page_title="音声分割ツール（ストレージ対応）",
    page_icon="🎧",
    layout="wide",
)

render_banner_line_by_key("light_green")

# ============================================================
# セッション記録（ログイン判定の正本）
# ============================================================
sub = page_session_heartbeat(
    st,
    PROJECTS_ROOT,
    app_name=APP_NAME,
    page_name=PAGE_NAME,
)

# 未ログインなら止める（テンプレ準拠）
if not sub:
    st.warning("ログインしていません。ポータルからログインしてください。")
    st.stop()

# ============================================================
# タイトル帯（テンプレ準拠）
# ============================================================
left, right = st.columns([2, 1])
with left:
    st.title("🎧 音声分割ツール")
with right:
    st.success(f"✅ ログイン中: **{sub}**")

subtitle("ストレージ対応")
st.caption("音声を分割（オーバーラップ付）し、ZIP化してダウンロード＋Storagesへ保存します。")

# ============================================================
# ページ説明
# ============================================================
st.write(
    "アップロードした音声（MP3/WAV/MP4）を一定長さで分割し、隣接チャンクに重なり（オーバーラップ）をつけます。"
    "文字起こし（transcription）前の前処理として使ってください。"
)

st.write("サイドバーの設定は，特に変更する必要はありません．そのままお使いください．")
render_audio_split_expander()

# ============================================================
# 保存パス用ユーザー名（sanitized）
# ============================================================
current_user = sub
username = _sanitize_username_for_path(str(current_user))

# ============================================================
# Sidebar（設定）
# ============================================================
with st.sidebar:
    st.header("設定")

    chunk_min = st.selectbox("チャンク長（分）", [3, 5, 10, 15, 20, 30], index=4)
    overlap_min = st.number_input(
        "オーバーラップ（分）",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.5,
    )
    export_fmt = st.selectbox("書き出しフォーマット", ["mp3", "wav (PCM16)"], index=0)
    target_bitrate = st.selectbox(
        "（MP3時のみ）ビットレート",
        ["原則そのまま/自動", "128k", "160k", "192k", "256k", "320k"],
        index=2,
        help="WAV出力では無効です。",
    )
    fade_ms = st.number_input(
        "フェード（クリックノイズ低減, ms）",
        min_value=0,
        max_value=2000,
        value=0,
        step=100,
    )
    min_tail_keep = st.checkbox(
        "最後の“短すぎる尻尾”は前チャンクに吸収（重複を増やさない）",
        value=True,
    )

# ============================================================
# 入力
# ============================================================
st.subheader("入力")

uploaded = st.file_uploader(
    "音声ファイルをアップロード（MP3/WAV/MP4）",
    type=["mp3", "wav", "mp4"],
)

if uploaded is None:
    st.info("音声ファイルをアップロードしてください。")
    st.stop()

# ============================================================
# アップロードが変わったらジョブ固定を解除（“2つできる”防止）
# ============================================================
upload_sig = f"{uploaded.name}:{uploaded.size}"
_reset_job_lock_on_new_upload(upload_sig)

# ============================================================
# プレビュー（保存しない）
# ============================================================
uploaded_bytes = uploaded.getvalue()
suffix = Path(uploaded.name).suffix.lower()

if suffix not in {".mp3", ".wav", ".mp4"}:
    st.error("対応していない拡張子です（.mp3 / .wav / .mp4）。")
    st.stop()

try:
    load_fmt = _guess_format_from_suffix(suffix)
    audio = AudioSegment.from_file(io.BytesIO(uploaded_bytes), format=load_fmt)
except Exception as e:
    st.error(f"音声の読み込みに失敗しました: {e}")
    st.stop()

chunk_ms = int(chunk_min * 60_000)
overlap_ms = int(float(overlap_min) * 60_000)

if overlap_ms >= chunk_ms:
    st.error("オーバーラップはチャンク長未満にしてください。")
    st.stop()

parts = split_with_overlap(
    audio=audio,
    chunk_ms=chunk_ms,
    overlap_ms=overlap_ms,
    fade_ms=int(fade_ms),
    absorb_tiny_tail=bool(min_tail_keep),
)

rows = []
for i, p in enumerate(parts):
    rows.append(
        {
            "Part": i,
            "Start": hhmmss(p["start_ms"]),
            "End": hhmmss(p["end_ms"]),
            "Duration": hhmmss(p["end_ms"] - p["start_ms"]),
        }
    )

# プレビュー結果を保持（結果表示用）
st.session_state[K_AUDIO_LEN_MS] = int(len(audio))
st.session_state[K_PARTS] = parts
st.session_state[K_ROWS] = rows

# ============================================================
# 実行（非AI）— 保存トリガー
# ============================================================
st.subheader("実行")

go = st.button(
    "📦 分割済み音声を生成（ZIP作成＋ストレージ保存）",
    type="primary",
    disabled=bool(st.session_state.get(K_JOB_LOCKED, False)),
)

if go:
    st.session_state[K_JOB_LOCKED] = True

    try:
        with st.spinner("分割済み音声をエクスポートして ZIP を作成しています…"):
            # ============================================================
            # 1) job 固定：最初の1回だけ作る
            # ============================================================
            if st.session_state[K_JOB_ID] is None:
                today_dir = datetime.now().strftime("%Y-%m-%d")
                job_id = now_job_id()
                job_root = STORAGE_ROOT / username / "minutes_app" / today_dir / job_id

                st.session_state[K_JOB_ID] = job_id
                st.session_state[K_JOB_ROOT] = str(job_root)

            job_id = st.session_state[K_JOB_ID]
            job_root = Path(st.session_state[K_JOB_ROOT])

            # ============================================================
            # 2) 保存先ディレクトリ作成
            # ============================================================
            original_dir = job_root / "original"
            split_dir = job_root / "split"
            transcript_dir = job_root / "transcript"
            logs_dir = job_root / "logs"

            for d in (original_dir, split_dir, transcript_dir, logs_dir):
                safe_mkdir(d)

            log_path = logs_dir / "process.log"

            # ============================================================
            # 3) ログ開始
            # ============================================================
            append_log(log_path, "AUDIO SPLIT START")
            append_log(log_path, f"job_id={job_id}")
            append_log(log_path, f"user_display={current_user}")
            append_log(log_path, f"user_dir={username}")
            append_log(log_path, f"uploaded_name={uploaded.name}")
            append_log(
                log_path,
                f"params chunk_min={chunk_min} overlap_min={overlap_min} export_fmt={export_fmt} "
                f"bitrate={target_bitrate} fade_ms={fade_ms} tail_absorb={min_tail_keep}",
            )

            # ============================================================
            # 4) original 保存（元ファイル）
            # ============================================================
            original_path = original_dir / uploaded.name
            original_path.write_bytes(uploaded_bytes)
            append_log(log_path, f"saved original -> {original_path.name}")

            # ============================================================
            # 5) 出力設定（WAV/MP3）
            # ============================================================
            base_name = (uploaded.name.rsplit(".", 1)[0] or "audio").replace(" ", "_")

            if export_fmt.startswith("wav"):
                out_ext = "wav"
                export_kwargs = {"format": "wav"}  # 既定でPCM16
            else:
                out_ext = "mp3"
                bitrate_arg = None if "自動" in target_bitrate else target_bitrate
                export_kwargs = {"format": "mp3"}
                if bitrate_arg:
                    export_kwargs["bitrate"] = bitrate_arg

            # ============================================================
            # 6) ZIP 作成（メモリ）＋ split/ に個別ファイル保存
            # ============================================================
            mem_zip = io.BytesIO()
            zip_name = f"{base_name}_split_overlap.zip"
            zip_path = split_dir / zip_name

            index_rows = []

            with zipfile.ZipFile(mem_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for i, p in enumerate(parts):
                    start_tag = hhmmss(p["start_ms"]).replace(":", "")
                    end_tag = hhmmss(p["end_ms"]).replace(":", "")
                    filename = f"{base_name}_part{i:03d}_{start_tag}-{end_tag}.{out_ext}"

                    buf = io.BytesIO()
                    p["segment"].export(buf, **export_kwargs)
                    chunk_bytes = buf.getvalue()

                    # split/ に保存（個別ファイルも残す）
                    (split_dir / filename).write_bytes(chunk_bytes)

                    # ZIP にも入れる
                    zf.writestr(filename, chunk_bytes)

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

                # インデックス（json）も ZIP に同梱
                zf.writestr(
                    f"{base_name}_index.json",
                    json.dumps(index_rows, ensure_ascii=False, indent=2).encode("utf-8"),
                )

            mem_zip.seek(0)

            # ZIP 本体を split/ に保存
            zip_path.write_bytes(mem_zip.getvalue())

            # ============================================================
            # 7) job.json 作成
            # ============================================================
            job_json = {
                "job_id": job_id,
                "user": str(current_user),
                "user_dir": username,
                "date": job_root.parent.name,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "paths": {
                    "job_root": str(job_root),
                    "original_dir": str(original_dir),
                    "original": str(original_path),
                    "split_dir": str(split_dir),
                    "zip": str(zip_path),
                    "transcript_dir": str(transcript_dir),
                    "logs_dir": str(logs_dir),
                },
                "config": {
                    "chunk_min": int(chunk_min),
                    "overlap_min": float(overlap_min),
                    "export_fmt": export_fmt,
                    "target_bitrate": target_bitrate,
                    "fade_ms": int(fade_ms),
                    "min_tail_keep": bool(min_tail_keep),
                },
                "status": {
                    "split": "done",
                    "transcribe": "not_started",
                    "merge": "not_started",
                    "dedup": "not_started",
                },
                "split_index": index_rows,
                "audio_total_ms": int(len(audio)),
            }
            write_json(job_root / "job.json", job_json)

            # ============================================================
            # 8) ログ終了
            # ============================================================
            append_log(log_path, f"saved zip -> {zip_name}")
            append_log(log_path, "AUDIO SPLIT DONE")

            # ============================================================
            # 9) ダウンロード用にセッションへ保持
            # ============================================================
            st.session_state[K_ZIP_BYTES] = mem_zip.getvalue()
            st.session_state[K_ZIP_NAME] = zip_name

        st.success(
            f"分割ファイルをストレージに保存しました（チャンク数: {len(parts)}・総再生時間: {hhmmss(len(audio))}）。"
        )
        st.caption(f"保存先（今回のジョブ）: {job_root}")

    except Exception as e:
        st.error(f"処理中にエラーが発生しました: {e}")

    finally:
        st.session_state[K_JOB_LOCKED] = False

# ============================================================
# 結果表示（テンプレ準拠）
# ============================================================
st.divider()
st.subheader("結果")

# 1) プレビュー表
if st.session_state.get(K_ROWS) is None:
    st.info("まだ結果がありません。")
else:
    st.subheader("分割プレビュー")
    st.dataframe(st.session_state[K_ROWS], hide_index=True)
    st.caption("※ ここでは Storages に保存しません。保存は上のボタンを押したときだけです。")

# 2) ZIPダウンロード（ZIP作成後に表示）
zip_bytes = st.session_state.get(K_ZIP_BYTES)
zip_name = st.session_state.get(K_ZIP_NAME)

if zip_bytes and zip_name:
    st.download_button(
        "⬇️ 分割済み音声をZIPで（パソコンに）ダウンロード",
        data=zip_bytes,
        file_name=zip_name,
        mime="application/zip",
    )
