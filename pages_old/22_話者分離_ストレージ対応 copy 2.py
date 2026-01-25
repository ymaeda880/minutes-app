# -*- coding: utf-8 -*-
# pages/22_話者分離_storage対応（新）.py
# ------------------------------------------------------------
# 🎙️ 話者分離・整形（議事録の前処理）storage対応（ログイン必須）
# - 最初に job_xxxx フォルダーを radio で選択（存在するものを全列挙）
# - 選択した job の transcript/*.txt を順番にすべて話者分離（単発なし）
# - 話者分離結果は transcript_speaker_separated/ に個別保存
# - その後、指定の「つなぎ」行を挟んで全連結し
#   transcript_speaker_separated_combined/ に保存
# - 巨大テキストを AI に直接投入しない（分割→話者分離→連結）
#
# ※ common_lib は改変しない
# ※ use_container_width は使わない（方針）
# ------------------------------------------------------------

from __future__ import annotations

import json
import time
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, List

import streamlit as st
from openai import OpenAI
import google.generativeai as genai

# ==== 共通ユーティリティ ====
from lib.costs import estimate_chat_cost_usd
from lib.tokens import extract_tokens_from_response
from lib.prompts import SPEAKER_PREP, get_group, build_prompt
from lib.prompts import (
    SPEAKER_MANDATORY,
    SPEAKER_MANDATORY_LIGHT,
    SPEAKER_MANDATORY_LIGHTER,
)
from config.config import (
    DEFAULT_USDJPY,
    get_gemini_api_key,
    has_gemini_api_key,
    estimate_tokens_from_text,
    estimate_gemini_cost_usd,
)
from ui.style import disable_heading_anchors
from lib.explanation import render_speaker_prep_expander

# ============================================================
# sys.path 調整（common_lib を import できるように）
# ============================================================
import sys

_THIS = Path(__file__).resolve()
PROJECTS_ROOT = _THIS.parents[3]

if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

# ============================================================
# session_state keys：ページ単位で名前空間化（他ページ汚染防止）
# ============================================================
PAGE_KEY_PREFIX = _THIS.stem  # e.g. "22_話者分離_storage対応（新）"

def k(name: str) -> str:
    return f"{PAGE_KEY_PREFIX}::{name}"


from common_lib.storage.external_ssd_root import resolve_storage_subdir_root
from common_lib.auth.auth_helpers import require_login

STORAGE_ROOT = resolve_storage_subdir_root(
    PROJECTS_ROOT,
    subdir="Storages",
)

# ============================================================
# utils
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


def _sanitize_username_for_path(username: str) -> str:
    u = (username or "").strip()
    if not u:
        return "anonymous"
    u = re.sub(r"[^0-9A-Za-z_-]+", "_", u).strip("_")
    return u or "anonymous"


def _human_dt(s: str | None) -> str:
    if not s:
        return "—"
    try:
        return s.replace("T", " ").replace("+00:00", "Z")
    except Exception:
        return s


def _read_job_json(job_dir: Path) -> dict:
    p = job_dir / "job.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ============================================================
# ここを追加：original から base / original ファイル名を拾う
# （無ければ "none"）
# ============================================================
def get_base_from_original(job_dir: Path) -> tuple[str, str]:
    """
    job_xxxx/original/* の先頭ファイル名を表示用に拾う
    - 無ければ ("none", "none")
    """
    original_dir = job_dir / "original"
    if not original_dir.exists():
        return "none", "none"

    files = list(original_dir.iterdir())
    if not files:
        return "none", "none"

    p = files[0]
    return p.stem, p.name


# ============================================================
# job listing（存在するものをすべて列挙）
# ============================================================
@dataclass
class JobItem:
    label: str
    job_dir: Path
    date: str
    job_id: str
    created_at: Optional[str]

# ============================================================
# ここを置き換え：list_all_jobs() を丸ごと差し替え
# （label に original 名 + base 名を含める）
# ============================================================
def list_all_jobs(user_dir: str) -> list[JobItem]:
    base = STORAGE_ROOT / user_dir / "minutes_app"
    if not base.exists():
        return []

    items: list[JobItem] = []

    for day_dir in sorted(base.glob("*"), reverse=True):
        if not day_dir.is_dir():
            continue

        for job_dir in sorted(day_dir.glob("job_*"), reverse=True):
            if not job_dir.is_dir():
                continue

            # transcript がある job のみ
            if not (job_dir / "transcript").exists():
                continue

            meta = _read_job_json(job_dir)
            job_id = str(meta.get("job_id") or job_dir.name)
            date = str(meta.get("date") or day_dir.name)
            created_at = meta.get("created_at")

            base_stem, original_name = get_base_from_original(job_dir)

            # radio は改行表示されるので、見やすく複数行にする
            label = (
                f"{date} / {job_dir.name}\n"
                f"  └ original: {original_name}\n"
                f"  └ base: {base_stem} / created={_human_dt(created_at)}"
            )

            items.append(
                JobItem(
                    label=label,
                    job_dir=job_dir,
                    date=date,
                    job_id=job_id,
                    created_at=created_at,
                )
            )

    return items


# ============================================================
# transcript/*.txt を順番に処理
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

# ============================================================
# AI 実行（単ファイル）
# ============================================================
def run_speaker_prep_one(prompt: str, model: str, client, usd_jpy: float):
    t0 = time.perf_counter()

    if model.startswith("gemini"):
        gem = genai.GenerativeModel(model)
        resp = gem.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        elapsed = time.perf_counter() - t0

        out_tok = estimate_tokens_from_text(text)
        in_tok = estimate_tokens_from_text(prompt)
        usd = estimate_gemini_cost_usd(
            model=model, input_tokens=in_tok, output_tokens=out_tok
        )
        jpy = (usd * usd_jpy) if usd is not None else None
        return text, elapsed, in_tok, out_tok, usd, jpy

    # OpenAI
    if client is None:
        raise RuntimeError("OPENAI_API_KEY が未設定です。")

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=100000,
    )
    elapsed = time.perf_counter() - t0
    text = resp.choices[0].message.content or ""

    in_tok, out_tok, _ = extract_tokens_from_response(resp)
    usd = estimate_chat_cost_usd(model, in_tok, out_tok)
    jpy = (usd * usd_jpy) if usd is not None else None
    return text, elapsed, in_tok, out_tok, usd, jpy

# ============================================================
# UI 設定
# ============================================================
st.set_page_config(
    page_title="③ 話者分離・整形（storage対応）",
    page_icon="🎙️",
    layout="wide",
)
disable_heading_anchors()

sub = require_login(st)
if not sub:
    st.stop()
left, right = st.columns([2, 1])
with left:
    st.title("🎙️ 話者分離・整形（ストレージ対応）")
with right:
    st.success(f"✅ ログイン中: **{sub}**")
current_user=sub

user_dir = _sanitize_username_for_path(str(current_user))

#st.title("🎙️ 話者分離・整形（storage対応）")

render_speaker_prep_expander()

st.markdown(
    """
- **ログイン必須**
- 最初に **job_xxxx フォルダーを radio で選択**
- 選択した job の **transcript/*.txt を順番にすべて話者分離**
- 話者分離後に **つなぎ行を挟んで連結**
"""
)

# ============================================================
# ログイン
# ============================================================


# ============================================================
# OpenAI / Gemini init
# ============================================================
OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key") or st.secrets.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

GEMINI_ENABLED = has_gemini_api_key()
if GEMINI_ENABLED:
    genai.configure(api_key=get_gemini_api_key())

# ============================================================
# Sidebar（モデル / プロンプト / 通貨）
# - UIは「旧・正常動作版」と同じ
# - ただし key は衝突回避のため最小限だけ名前空間化（k()）
# ============================================================
with st.sidebar:
    st.subheader("モデル設定")

    MODEL_OPTIONS = [
        "gpt-5-mini",
        "gpt-5-nano",
        "gemini-2.0-flash",
    ]

    # ---- model ----
    st.session_state.setdefault(k("speaker_model"), "gpt-5-mini")
    model = st.radio(
        "モデル",
        MODEL_OPTIONS,
        key=k("speaker_model"),
    )

    if model.startswith("gemini") and not GEMINI_ENABLED:
        st.warning("Gemini API Key が未設定です。")
        st.stop()

    # ---- prompt level ----
    st.subheader("プロンプト設定")

    PROMPT_LEVEL_OPTIONS = [
        "標準（精度優先）",
        "軽量（タイムアウト低減）",
        "超軽量（最小負荷）",
    ]

    st.session_state.setdefault(k("speaker_prompt_level"), "標準（精度優先）")
    prompt_level = st.radio(
        "話者分離プロンプト",
        PROMPT_LEVEL_OPTIONS,
        key=k("speaker_prompt_level"),
    )

    # ---- currency ----
    st.subheader("通貨換算")
    usd_jpy = st.number_input(
        "USD/JPY",
        min_value=50.0,
        max_value=500.0,
        value=float(DEFAULT_USDJPY),
        step=0.5,
        key=k("speaker_usd_jpy"),
    )


# ============================================================
# メイン UI
# ============================================================
left, right = st.columns([1, 1], gap="large")

# ---- 右：入力方式 + job 確定 ----
with right:
    st.subheader("① 入力方式")

    input_mode = st.radio(
        "どこから transcript を読み込むか",
        options=[
            "既存ジョブ（storage より読み込み）",
            "新規アップロード（drop → 新規job作成 → transcript/ に保存）",
        ],
        index=0,
        key=k("input_mode"),
        help="新規アップロードは、アップロードした複数 .txt を transcript/ に保存し、その順番で連続話者分離します。",
    )

    # --- 新規アップロード用：21ページと同じ考え方（rerun耐性） ---
    K_UP_LAST_SIG = k("upload_last_sig")
    K_UP_JOB_ID = k("upload_job_id")
    K_UP_JOB_ROOT = k("upload_job_root")
    K_UP_LOCKED = k("upload_job_locked")

    st.session_state.setdefault(K_UP_LAST_SIG, None)
    st.session_state.setdefault(K_UP_JOB_ID, None)
    st.session_state.setdefault(K_UP_JOB_ROOT, None)
    st.session_state.setdefault(K_UP_LOCKED, False)

    def now_job_id() -> str:
        return "job_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    def safe_filename(name: str) -> str:
        s = (name or "text").strip()
        s = s.replace("\\", "_").replace("/", "_").replace(":", "_")
        s = re.sub(r"\s+", " ", s)
        return s or "text"

    # --- 直近 N ジョブだけ残す（21と同様） ---
    MAX_JOBS = 5

    def cleanup_old_jobs(user_root: Path, max_jobs: int) -> None:
        # 日付降順、job降順で数える（scanは list_all_jobs と同じ基準）
        jobs_all = list_all_jobs(user_dir)
        if len(jobs_all) <= max_jobs:
            return
        for j in jobs_all[max_jobs:]:
            try:
                import shutil
                shutil.rmtree(j.job_dir)
                day_dir = j.job_dir.parent
                if day_dir.exists() and not any(day_dir.iterdir()):
                    day_dir.rmdir()
            except Exception:
                pass

    picked_job: Optional[JobItem] = None
    transcript_files: List[Path] = []

    if input_mode.startswith("既存ジョブ"):
        st.subheader("ジョブ選択（storage）")
        st.caption("直近5件より古いジョブは自動削除されます。")

        cleanup_old_jobs(STORAGE_ROOT / user_dir / "minutes_app", MAX_JOBS)

      

        jobs = list_all_jobs(user_dir)
        if not jobs:
            st.info("minutes_app の job が見つかりません。先に文字起こしで job を作成してください。")
            st.stop()




        labels = [j.label for j in jobs]
        picked = st.radio(
            "対象ジョブ",
            options=labels,
            index=0,
            key=k("job_picker"),
            help="transcript/ を参照できる job を選びます。",
        )
        picked_job = jobs[labels.index(picked)]

        job_dir = picked_job.job_dir
        st.caption(f"job_dir: {job_dir}")

        transcript_files = list_transcript_txts(job_dir)
        if not transcript_files:
            st.error("transcript/*.txt が見つかりません。")
            st.stop()

        st.markdown("### 対象ファイル（処理順）")
        st.write([p.name for p in transcript_files])

    else:
        st.subheader("新規アップロード（drop）")

        uploaded_files = st.file_uploader(
            "文字起こしテキスト（.txt）をドロップ/選択（複数可・アップロード順＝連続順）",
            type=["txt"],
            accept_multiple_files=True,
            key=k("uploader_txt"),
        )

        # ✅ 0件なら job作成ボタン自体を出さない（ここで終了）
        if not uploaded_files:
            st.info("テキストファイル（.txt）を1つ以上アップロードしてください。")
            st.stop()
            

        sig_parts = [
            f"{getattr(f, 'name', 'file')}:{getattr(f, 'size', 0)}"
            for f in uploaded_files
        ]
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
            st.success(f"✅ 作成済みジョブを復元しました: {job_root}")

        else:
            create_job_clicked = st.button(
                "📦 アップロードを transcript/ として保存して新規ジョブを作成",
                type="primary",
                disabled=bool(st.session_state.get(K_UP_LOCKED, False)),
                key=k("create_job_btn"),
                help="押した時点で Storages に job_YYYYMMDD_HHMMSS を作り、transcript/ に保存します（確定制）。",
            )


            if not create_job_clicked:
                uploaded_ready = False


            st.session_state[K_UP_LOCKED] = True

            try:
                with st.spinner("新規ジョブを作成して transcript/ に保存しています…"):
                    today_dir = datetime.now().strftime("%Y-%m-%d")
                    job_id = now_job_id()
                    job_root = STORAGE_ROOT / user_dir / "minutes_app" / today_dir / job_id

                    transcript_dir = job_root / "transcript"
                    logs_dir = job_root / "logs"
                    safe_mkdir(transcript_dir)
                    safe_mkdir(logs_dir)

                    log_path = logs_dir / "process.log"
                    append_log(log_path, "UPLOAD->JOB START")
                    append_log(log_path, f"job_dir={job_root}")

                    transcript_index: List[dict] = []

                    for i, uf in enumerate(uploaded_files, start=1):
                        name = safe_filename(getattr(uf, "name", f"text_{i}.txt"))
                        out_name = f"{i:03d}_{name}"
                        out_path = transcript_dir / out_name
                        b = uf.getvalue()
                        try:
                            text = b.decode("utf-8")
                        except UnicodeDecodeError:
                            text = b.decode("cp932", errors="ignore")
                        write_text(out_path, text)

                        transcript_index.append(
                            {
                                "order": i,
                                "original_name": name,
                                "saved_name": out_name,
                                "saved_path": str(out_path),
                                "size_bytes": int(len(b)),
                            }
                        )
                        append_log(log_path, f"saved transcript -> {out_name}")

                    job_json = {
                        "job_id": job_id,
                        "user": str(current_user),
                        "user_dir": user_dir,
                        "date": today_dir,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "source": "upload_drop_txt",
                        "paths": {
                            "job_root": str(job_root),
                            "transcript_dir": str(transcript_dir),
                            "logs_dir": str(logs_dir),
                        },
                        "status": {
                            "speaker_separate": "not_started",
                        },
                        "transcript_index": transcript_index,
                    }
                    write_text(job_root / "job.json", json.dumps(job_json, ensure_ascii=False, indent=2))
                    append_log(log_path, "UPLOAD->JOB DONE")

                st.session_state[K_UP_JOB_ID] = job_id
                st.session_state[K_UP_JOB_ROOT] = str(job_root)

                cleanup_old_jobs(STORAGE_ROOT / user_dir / "minutes_app", MAX_JOBS)

                st.success(f"✅ 新規ジョブを作成しました: {job_root}")
                st.caption(f"保存先: {job_root}")

            finally:
                st.session_state[K_UP_LOCKED] = False

        # --- job確定（復元後/作成後 共通） ---
        job_root = Path(st.session_state[K_UP_JOB_ROOT])
        st.caption(f"job_dir: {job_root}")

        transcript_files = list_transcript_txts(job_root)
        if not transcript_files:
            st.error("transcript/*.txt が見つかりません。保存を確認してください。")
            st.stop()

        st.markdown("### 対象ファイル（処理順）")
        st.write([p.name for p in transcript_files])

    # ---- 以降の実行用に session に確定保存（キーは名前空間化） ----
    # st.session_state[k("speaker_job_dir")] = str(job_dir)
    # st.session_state[k("speaker_transcript_files")] = [str(p) for p in transcript_files]

# ---- 以降の実行用に session に確定保存（job が確定している時だけ） ----
job_ready = bool(transcript_files)

if job_ready:
    # 新規アップロード側では job_root を job_dir として扱う
    if input_mode.startswith("新規アップロード"):
        job_dir = job_root

    st.session_state[k("speaker_job_dir")] = str(job_dir)
    st.session_state[k("speaker_transcript_files")] = [str(p) for p in transcript_files]
else:
    st.session_state[k("speaker_job_dir")] = None
    st.session_state[k("speaker_transcript_files")] = []





# ---- 左：プロンプト & 実行 ----
with left:
    st.subheader("② プロンプト")

    group = get_group(SPEAKER_PREP)

    #_level = st.session_state.get("speaker_prompt_level")
    _level = st.session_state.get(k("speaker_prompt_level"))

    if _level == "標準（精度優先）":
        mandatory_default = SPEAKER_MANDATORY
    elif _level == "軽量（タイムアウト低減）":
        mandatory_default = SPEAKER_MANDATORY_LIGHT
    else:
        mandatory_default = SPEAKER_MANDATORY_LIGHTER

    # ============================================================
    # ★ 追加：prompt level の変更を検知して mandatory_prompt を自動更新
    #   - level が変わったときだけ mandatory_prompt を差し替える
    #   - 初回は setdefault で入れる
    # ============================================================
    prev_level = st.session_state.get(k("_speaker_prompt_level_prev"))

    if prev_level is None:
        st.session_state.setdefault(k("mandatory_prompt"), mandatory_default)
        st.session_state[k("_speaker_prompt_level_prev")] = _level
    else:
        if prev_level != _level:
            st.session_state[k("mandatory_prompt")] = mandatory_default
            st.session_state[k("_speaker_prompt_level_prev")] = _level
        else:
            st.session_state.setdefault(k("mandatory_prompt"), mandatory_default)

    st.text_area("必須プロンプト", height=220, key=k("mandatory_prompt"))

    st.session_state.setdefault(
        k("preset_label"), group.label_for_key(group.default_preset_key)
    )
    st.session_state.setdefault(
        k("preset_text"), group.body_for_label(st.session_state[k("preset_label")])
    )
    st.session_state.setdefault(k("extra_text"), "")

    def _on_change_preset():
        st.session_state[k("preset_text")] = group.body_for_label(
            st.session_state[k("preset_label")]
        )

    st.selectbox(
        "追記プリセット",
        options=group.preset_labels(),
        key=k("preset_label"),
        on_change=_on_change_preset,
    )

    st.text_area("プリセット本文", height=120, key=k("preset_text"))
    st.text_area("追加指示（任意）", height=80, key=k("extra_text"))



    job_ready = bool(st.session_state.get(k("speaker_job_dir"))) and bool(st.session_state.get(k("speaker_transcript_files")))

    batch_btn = st.button(
        "③ transcript を順番に話者分離（ストレージ保存＋連結）",
        type="primary",
        key=k("batch_btn"),
        disabled=not job_ready,
    )


# ============================================================
# 実行（バッチのみ）
# ============================================================
if batch_btn:
    job_dir = Path(st.session_state[k("speaker_job_dir")])
    transcript_files = [Path(p) for p in st.session_state[k("speaker_transcript_files")]]

    speaker_sep_dir = job_dir / "transcript_speaker_separated"
    speaker_comb_dir = job_dir / "transcript_speaker_separated_combined"
    logs_dir = job_dir / "logs"
    safe_mkdir(speaker_sep_dir)
    safe_mkdir(speaker_comb_dir)
    safe_mkdir(logs_dir)

    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_path = speaker_comb_dir / f"transcript_speaker_separated_combined_{ts_tag}.txt"

    combined_chunks: List[str] = []
    total_in = total_out = 0
    total_usd = 0.0

    log_path = logs_dir / "process.log"
    append_log(log_path, "SPEAKER PREP BATCH START")
    append_log(log_path, f"job_dir={job_dir}")
    append_log(log_path, f"count={len(transcript_files)} model={model}")

    prog = st.progress(0)
    status = st.empty()

    for i, src_path in enumerate(transcript_files, start=1):
        status.write(f"{i}/{len(transcript_files)} 処理中: {src_path.name}")
        src = src_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not src:
            continue

        prompt = build_prompt(
            st.session_state[k("mandatory_prompt")],
            st.session_state[k("preset_text")],
            st.session_state[k("extra_text")],
            src,
        )

        text, elapsed, in_tok, out_tok, usd, jpy = run_speaker_prep_one(
            prompt, model, client, usd_jpy
        )

        out_path = speaker_sep_dir / f"{i:03d}_{src_path.stem}_speaker.txt"
        write_text(out_path, text or "")

        combined_chunks.append(text or "")
        if i < len(transcript_files):
            combined_chunks.append(make_connector_line(src_path.name))

        total_in += int(in_tok)
        total_out += int(out_tok)
        if usd is not None:
            total_usd += float(usd)

        append_log(
            log_path,
            f"DONE {src_path.name} -> {out_path.name} in={in_tok} out={out_tok} usd={usd}",
        )

        prog.progress(int(i / len(transcript_files) * 100))

    combined_text = "\n\n".join(combined_chunks)
    write_text(combined_path, combined_text)

    st.success(f"✅ ストレージに保存しました: {combined_path.name}")
    st.caption(f"保存先: {combined_path}")


    append_log(log_path, f"COMBINED -> {combined_path.name}")
    append_log(log_path, "SPEAKER PREP BATCH DONE")

    st.success("完了しました。")
    st.write(
        {
            "speaker_separated_dir": str(speaker_sep_dir),
            "speaker_combined_path": str(combined_path),
            "total_tokens": {"input": total_in, "output": total_out},
            "total_usd_est": total_usd,
            "total_jpy_est": total_usd * usd_jpy,
        }
    )

    st.download_button(
        "連結結果をダウンロード（.txt）",
        data=combined_text.encode("utf-8"),
        file_name=combined_path.name,
        mime="text/plain",
    )
