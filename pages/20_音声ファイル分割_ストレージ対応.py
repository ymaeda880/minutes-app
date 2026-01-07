# pages/20_音声ファイル分割_storage対応.py
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
import re
import sys

import streamlit as st
from pydub import AudioSegment

from lib.explanation import render_audio_split_expander

# ============================================================
# sys.path 調整（ボット頁に倣う）
# ============================================================
_THIS = Path(__file__).resolve()
PROJECTS_ROOT = _THIS.parents[3]

if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

#from common_lib.auth.auth_helpers import get_current_user_from_session_or_cookie  # noqa: E402
from common_lib.storage.external_ssd_root import resolve_storage_subdir_root
from common_lib.auth.auth_helpers import require_login


# ============================================================
# Storage config（PROJECTS_ROOT 基準）
# ============================================================
STORAGE_ROOT = resolve_storage_subdir_root(
    PROJECTS_ROOT,
    subdir="Storages",
)

# ============================================================
# 関数群
# ============================================================
def _sanitize_username_for_path(username: str) -> str:
    u = (username or "").strip()
    if not u:
        return "anonymous"
    u = re.sub(r"[^0-9A-Za-z_-]+", "_", u).strip("_")
    return u or "anonymous"


def now_job_id() -> str:
    return "job_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def hhmmss(ms: int) -> str:
    return str(timedelta(milliseconds=ms)).split(".")[0]


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_log(log_path: Path, msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_mkdir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def write_json(p: Path, obj: dict) -> None:
    safe_mkdir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def split_with_overlap(
    audio: AudioSegment,
    chunk_ms: int,
    overlap_ms: int,
    fade_ms: int,
    absorb_tiny_tail: bool,
):
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

        if absorb_tiny_tail and start > 0 and end == n:
            tail_len = end - start
            if tail_len < overlap_ms:
                prev = results[-1]
                prev["end_ms"] = n
                prev["segment"] = audio[prev["start_ms"]:n]
                break

        if fade_ms > 0 and len(seg) > fade_ms * 2:
            seg = seg.fade_in(fade_ms).fade_out(fade_ms)

        results.append({"start_ms": start, "end_ms": end, "segment": seg})

        if end == n:
            break
        start += step

    return results


def _guess_format_from_suffix(suffix: str) -> str:
    suf = (suffix or "").lower()
    if suf == ".mp3":
        return "mp3"
    if suf == ".wav":
        return "wav"
    if suf in {".mp4", ".m4a"}:
        return "mp4"
    return suf.lstrip(".")


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="音声分割ツール（storage）", page_icon="🎧", layout="centered")

sub = require_login(st)
if not sub:
    st.stop()
left, right = st.columns([2, 1])
with left:
    st.title("🎧 音声分割ツール（ストレージ対応）")
with right:
    st.success(f"✅ ログイン中: **{sub}**")

#st.title("🎧 音声分割ツール（storage 保存対応）")

current_user = sub

#user_dir = _sanitize_username_for_path(str(current_user))
username = _sanitize_username_for_path(str(current_user))


st.write(
    "アップロードした音声（MP3/WAV/MP4）を一定長さで分割し、隣接チャンクに重なり（オーバーラップ）をつけます。"
    "文字起こし（transcription）前の前処理に行ってください。"
)
render_audio_split_expander()

# === DEBUG: base_dir（保存先ルート）===
st.caption(f"[DEBUG] storages_root = {STORAGE_ROOT}")

# ============================================================
# ログイン判定
# ============================================================
#

# ============================================================
# session_state：ジョブ固定（“2つできる”防止）
# ============================================================
st.session_state.setdefault("split_last_upload_sig", None)
st.session_state.setdefault("split_job_id", None)
st.session_state.setdefault("split_job_root", None)
st.session_state.setdefault("split_job_locked", False)

# ============================================================
# Sidebar（設定だけ置く：アップロードは置かない）
# ============================================================
with st.sidebar:
    st.header("設定")
    chunk_min = st.selectbox("チャンク長（分）", [3, 5, 10, 15, 20, 30], index=4)
    overlap_min = st.number_input("オーバーラップ（分）", min_value=0.0, max_value=10.0, value=1.0, step=0.5)
    export_fmt = st.selectbox("書き出しフォーマット", ["mp3", "wav (PCM16)"], index=0)
    target_bitrate = st.selectbox(
        "（MP3時のみ）ビットレート",
        ["原則そのまま/自動", "128k", "160k", "192k", "256k", "320k"],
        index=2,
        help="WAV出力では無効です。",
    )
    fade_ms = st.number_input("フェード（クリックノイズ低減, ms）", min_value=0, max_value=2000, value=0, step=100)
    min_tail_keep = st.checkbox("最後の“短すぎる尻尾”は前チャンクに吸収（重複を増やさない）", value=True)

# ============================================================
# メイン：アップロード（ここが要望）
# ============================================================
uploaded = st.file_uploader(
    "音声ファイルをアップロード（MP3/WAV/MP4）",
    type=["mp3", "wav", "mp4"],
)

if uploaded is None:
    st.info("音声ファイルをアップロードしてください。")
    st.stop()

# アップロードが変わったらジョブ固定を解除
upload_sig = f"{uploaded.name}:{uploaded.size}"
if st.session_state["split_last_upload_sig"] != upload_sig:
    st.session_state["split_last_upload_sig"] = upload_sig
    st.session_state["split_job_id"] = None
    st.session_state["split_job_root"] = None
    st.session_state["split_job_locked"] = False

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
overlap_ms = int(overlap_min * 60_000)
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

st.subheader("分割プレビュー")
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
st.dataframe(rows, hide_index=True)

st.caption("※ ここでは Storages に保存しません。保存は下のボタンを押したときだけです。")

# ============================================================
# 保存ボタン（ここで job 作成＆固定）
# ============================================================
go = st.button(
    "📦 分割済み音声を生成（保存＋ZIP）",
    type="primary",
    disabled=bool(st.session_state.get("split_job_locked", False)),
)

if go:
    st.session_state["split_job_locked"] = True

    try:
        with st.spinner("分割済み音声をエクスポートして ZIP を作成しています…"):
            # job 固定：最初の1回だけ作る
            if st.session_state["split_job_id"] is None:
                today_dir = datetime.now().strftime("%Y-%m-%d")
                job_id = now_job_id()
                job_root = STORAGE_ROOT / username / "minutes_app" / today_dir / job_id
                st.session_state["split_job_id"] = job_id
                st.session_state["split_job_root"] = str(job_root)

            job_id = st.session_state["split_job_id"]
            job_root = Path(st.session_state["split_job_root"])

            original_dir = job_root / "original"
            split_dir = job_root / "split"
            transcript_dir = job_root / "transcript"
            logs_dir = job_root / "logs"
            for d in (original_dir, split_dir, transcript_dir, logs_dir):
                safe_mkdir(d)

            log_path = logs_dir / "process.log"
            append_log(log_path, "AUDIO SPLIT START")
            append_log(log_path, f"job_id={job_id}")
            append_log(log_path, f"user_display={current_user}")
            append_log(log_path, f"user_dir={username}")
            append_log(log_path, f"uploaded_name={uploaded.name}")
            append_log(
                log_path,
                f"params chunk_min={chunk_min} overlap_min={overlap_min} export_fmt={export_fmt} bitrate={target_bitrate} fade_ms={fade_ms} tail_absorb={min_tail_keep}",
            )

            # original（元ファイル名）
            original_path = original_dir / uploaded.name
            original_path.write_bytes(uploaded_bytes)
            append_log(log_path, f"saved original -> {original_path.name}")

            base_name = (uploaded.name.rsplit(".", 1)[0] or "audio").replace(" ", "_")

            if export_fmt.startswith("wav"):
                out_ext = "wav"
                export_kwargs = {"format": "wav"}
            else:
                out_ext = "mp3"
                bitrate_arg = None if "自動" in target_bitrate else target_bitrate
                export_kwargs = {"format": "mp3"}
                if bitrate_arg:
                    export_kwargs["bitrate"] = bitrate_arg

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

                    (split_dir / filename).write_bytes(chunk_bytes)
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

                zf.writestr(
                    f"{base_name}_index.json",
                    json.dumps(index_rows, ensure_ascii=False, indent=2).encode("utf-8"),
                )

            mem_zip.seek(0)
            zip_path.write_bytes(mem_zip.getvalue())

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

            append_log(log_path, f"saved zip -> {zip_name}")
            append_log(log_path, "AUDIO SPLIT DONE")

        st.success(f"分割ファイルをストレージに保存しました（チャンク数: {len(parts)}・総再生時間: {hhmmss(len(audio))}）。")
        st.caption(f"保存先（今回のジョブ）: {job_root}")

        st.download_button(
            "⬇️ 分割済み音声をZIPで（パソコンに）ダウンロード",
            data=mem_zip,
            file_name=zip_name,
            mime="application/zip",
        )

    except Exception as e:
        st.error(f"処理中にエラーが発生しました: {e}")
    finally:
        st.session_state["split_job_locked"] = False
