# -*- coding: utf-8 -*-
# pages/22_話者分離_storage対応（新）.py
# ------------------------------------------------------------
# 🎙️ 話者分離・整形（議事録の前処理）storage対応（ログイン必須）
# - 既存 job 選択 or 新規アップロード（.txt）→ 新規 job 作成
# - job/transcript/*.txt を順番に話者分離（単発なし）
# - transcript_speaker_separated/ に個別保存
# - transcript_speaker_separated_combined/ に連結保存
#
# テンプレ準拠（AI実行 + busy 記録）：
# - page_session_heartbeat をログイン判定の正本とする（require_login は使わない）
# - busy（ai_runs.db）を with busy_run で必ず記録（バッチ=1run）
# - 実行時間・費用・tokens は「run summary（共通UI）」で 1ブロック表示
#
# 製本（AI寄せ + cost/usage寄せ）：
# - pages は providers 直叩き禁止（OpenAI / google.generativeai / requests など）
# - pages は推計禁止（tokens推定・単価換算・USD→JPY換算禁止）
# - pages は common_lib.ai.routing の task API のみを呼ぶ（入口の正本）
# - cost/usage は Result が返せる範囲で busy/UI に反映するのみ
# - busy への反映は「最後に1回だけ（合計）」
#
# UI方針：
# - use_container_width は使わない
# - st.form は使わない
# - st.button()/st.download_button() に width 引数は使わない
# ------------------------------------------------------------

# ============================================================
# 標準ライブラリ
# ============================================================
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Any

# ============================================================
# サードパーティ
# ============================================================
import streamlit as st

# ============================================================
# パス解決（common_lib を import できるように）
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
# page key（session_state key 統一）
# ============================================================
PAGE_KEY_PREFIX = PAGE_NAME


def k(name: str) -> str:
    return f"{PAGE_KEY_PREFIX}::{name}"

# ============================================================
# Gemini 利用可否チェック（google-genai）
# ============================================================
def _gemini_available() -> bool:
    """
    Gemini が利用可能かを判定する。
    - google-genai が import できるかのみ確認
    """
    try:
        from google import genai  # google-genai
        _ = genai
        return True
    except Exception:
        return False

# ============================================================
# common_lib（正本：sessions / ui / busy / storage）
# ============================================================
from common_lib.ui.page_header import render_standard_page_header
from common_lib.busy import busy_run
from common_lib.storage.external_ssd_root import resolve_storage_subdir_root

# ============================================================
# common_lib（正本UI：run summary）
# - cost/tokens/time を 1ブロックで表示（テンプレ準拠）
# ============================================================
from common_lib.ui.run_summary import render_run_summary_compact

# ============================================================
# AI 正本（入口：routing）
# ============================================================
from common_lib.ai.routing import call_text

# ============================================================
# common_lib.ai.models（モデル定義の正本）
# ============================================================
from common_lib.ai.models import TEXT_MODEL_CATALOG, DEFAULT_TEXT_MODEL_KEY

# ============================================================
# common_lib.ui（正本UI：Textモデル選択）
# ============================================================
from common_lib.ui.model_picker import render_text_model_picker

# ============================================================
# 外部ユーティリティ（既存資産：プロンプト生成などはページ責務）
# ============================================================
from ui.style import disable_heading_anchors
from lib.prompts import SPEAKER_PREP, get_group, build_prompt
from lib.prompts import (
    SPEAKER_MANDATORY,
    SPEAKER_MANDATORY_LIGHT,
    SPEAKER_MANDATORY_LIGHTER,
)

from lib.speaker_prep.explanation import (
    render_speaker_prep_page_intro,
    render_speaker_prep_help_expander,
)

# ============================================================
# Storage root（minutes_app 配下）
# ============================================================
STORAGE_ROOT = resolve_storage_subdir_root(
    PROJECTS_ROOT,
    subdir="Storages",
)

# ============================================================
# utils（ファイルI/O）
# ============================================================
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_text(p: Path, s: str) -> None:
    safe_mkdir(p.parent)
    p.write_text(s, encoding="utf-8")


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


def get_base_from_original(job_dir: Path) -> tuple[str, str]:
    """job_xxxx/original/* の先頭ファイル名を表示用に拾う（無ければ none）"""
    original_dir = job_dir / "original"
    if not original_dir.exists():
        return "none", "none"
    files = list(original_dir.iterdir())
    if not files:
        return "none", "none"
    p = files[0]
    return p.stem, p.name


# ============================================================
# job listing
# ============================================================
@dataclass
class JobItem:
    label: str
    job_dir: Path
    date: str
    job_id: str
    created_at: Optional[str]


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

            # ------------------------------------------------------------
            # transcript がある job のみ
            # ------------------------------------------------------------
            if not (job_dir / "transcript").exists():
                continue

            meta = _read_job_json(job_dir)
            job_id = str(meta.get("job_id") or job_dir.name)
            date = str(meta.get("date") or day_dir.name)
            created_at = meta.get("created_at")

            base_stem, original_name = get_base_from_original(job_dir)
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
# transcript listing
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
# - pages は provider 推測をしない（sidebarで明示選択）
# ============================================================
def run_speaker_prep_one(*, prompt: str, provider: str, model: str) -> Any:
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
# ページ設定（必須・統一）
# ============================================================
st.set_page_config(
    page_title="③ 話者分離・整形（storage対応）",
    page_icon="🎙️",
    layout="wide",
)


# ------------------------------------------------------------
# UI小物（見出しアンカー無効化）
# ------------------------------------------------------------
#disable_heading_anchors()

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
    title="🎙️ 話者分離",
    subtitle_text="文字起こしテキストを話者ごとに分離",
    default_banner_key="light_green",
)

# ============================================================
# ページ説明
# ============================================================
render_speaker_prep_page_intro()

# ============================================================
# 詳細説明
# ============================================================
render_speaker_prep_help_expander(
    theme=theme,
    banner_key=BANNER_KEY,
)

# ============================================================
# セッションキー（正本：run + 合計）
# ============================================================
K_LAST_RUN_ID = f"{PAGE_NAME}__last_run_id"
K_LAST_RUN_ACTION = f"{PAGE_NAME}__last_run_action"
K_LAST_RESULT = f"{PAGE_NAME}__last_result"

# ------------------------------------------------------------
# 合計（表示用）
# ------------------------------------------------------------
K_LAST_TOTAL_IN = f"{PAGE_NAME}__total_in"
K_LAST_TOTAL_OUT = f"{PAGE_NAME}__total_out"
K_LAST_TOTAL_COST = f"{PAGE_NAME}__total_cost"  # CostResult or None
K_LAST_MODEL = f"{PAGE_NAME}__model"
K_LAST_PROVIDER = f"{PAGE_NAME}__provider"

# ------------------------------------------------------------
# 初期化
# ------------------------------------------------------------
st.session_state.setdefault(K_LAST_RUN_ID, "")
st.session_state.setdefault(K_LAST_RUN_ACTION, "")
st.session_state.setdefault(K_LAST_RESULT, None)

st.session_state.setdefault(K_LAST_TOTAL_IN, None)
st.session_state.setdefault(K_LAST_TOTAL_OUT, None)
st.session_state.setdefault(K_LAST_TOTAL_COST, None)
st.session_state.setdefault(K_LAST_MODEL, "")
st.session_state.setdefault(K_LAST_PROVIDER, "")

# ============================================================
# Sidebar（モデル・プロンプト設定：テンプレ準拠）
# - provider/model は common_lib の正本UIに寄せる
# - pages 側で provider 推測・分岐ロジックを書かない
# ============================================================
with st.sidebar:
    # ------------------------------------------------------------
    # モデル設定（Text 正本UI）
    # - 戻り値は provider:model（例: openai:gpt-5-mini）
    # ------------------------------------------------------------
    #st.subheader("モデル設定")

    st.session_state.setdefault(k("speaker_model_key"), DEFAULT_TEXT_MODEL_KEY)

    model_key = render_text_model_picker(
        title="モデル選択",
        catalog=TEXT_MODEL_CATALOG,
        session_key=k("speaker_model_key"),
        default_key=DEFAULT_TEXT_MODEL_KEY,
        page_name=PAGE_NAME,
        gemini_available=_gemini_available(),
    )

    # ------------------------------------------------------------
    # provider / model の確定（文字列から分解：推測ではない）
    # ------------------------------------------------------------
    provider = str(model_key).split(":", 1)[0]
    model = str(model_key).split(":", 1)[1] if ":" in str(model_key) else str(model_key)

    # ------------------------------------------------------------
    # プロンプト設定
    # ------------------------------------------------------------
    st.subheader("プロンプト設定")

    PROMPT_LEVEL_OPTIONS = ["標準（精度優先）", "軽量（タイムアウト低減）", "超軽量（最小負荷）"]
    st.session_state.setdefault(k("speaker_prompt_level"), "標準（精度優先）")
    prompt_level = st.radio(
        "話者分離プロンプト",
        PROMPT_LEVEL_OPTIONS,
        key=k("speaker_prompt_level"),
    )

# ============================================================
# ① プロンプト（expander）
# ============================================================
with st.expander("プロンプト（クリックで開く）", expanded=False):
    st.subheader("プロンプト")

    group = get_group(SPEAKER_PREP)

    _level = st.session_state.get(k("speaker_prompt_level"))
    if _level == "標準（精度優先）":
        mandatory_default = SPEAKER_MANDATORY
    elif _level == "軽量（タイムアウト低減）":
        mandatory_default = SPEAKER_MANDATORY_LIGHT
    else:
        mandatory_default = SPEAKER_MANDATORY_LIGHTER

    prev_level = st.session_state.get(k("_speaker_prompt_level_prev"))
    if prev_level is None:
        st.session_state.setdefault(k("mandatory_prompt"), mandatory_default)
        st.session_state[k("_speaker_prompt_level_prev")] = _level
    elif prev_level != _level:
        st.session_state[k("mandatory_prompt")] = mandatory_default
        st.session_state[k("_speaker_prompt_level_prev")] = _level
    else:
        st.session_state.setdefault(k("mandatory_prompt"), mandatory_default)

    st.text_area("必須プロンプト", height=220, key=k("mandatory_prompt"))

    st.session_state.setdefault(
        k("preset_label"),
        group.label_for_key(group.default_preset_key),
    )
    st.session_state.setdefault(
        k("preset_text"),
        group.body_for_label(st.session_state[k("preset_label")]),
    )
    st.session_state.setdefault(k("extra_text"), "")

    # ------------------------------------------------------------
    # プリセット連動（label -> text）
    # ------------------------------------------------------------
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

# ============================================================
# ② 入力（ここで job_dir / transcript_files を確定）
# ============================================================
#st.subheader("入力方式")
#st.caption("文字起こしテキスト（.txt）を複数アップロードできます（順番＝連続処理順）。")

st.subheader("① テキストファイルの設定")
st.caption("文字起こししたテキストファイルを読み込む先をここで指定します．"
    "「文字起こし」を行った時にサーバー内部に自動保存されたテキストファイルを"
    "使用するときは「既存ジョブ」を選択してください．")



current_user = sub
user_dir = _sanitize_username_for_path(str(current_user))

input_mode = st.radio(
    "どこから transcript を読み込むか",
    options=[
        "既存ジョブ（サーバー内部のストレージより読み込み）",
        "新規アップロード",
    ],
    index=0,
    label_visibility="collapsed",
    key=k("input_mode"),
)

# ------------------------------------------------------------
# state（新規アップロード復元用）
# ------------------------------------------------------------
K_UP_LAST_SIG = k("upload_last_sig")
K_UP_JOB_ROOT = k("upload_job_root")
st.session_state.setdefault(K_UP_LAST_SIG, None)
st.session_state.setdefault(K_UP_JOB_ROOT, None)

job_dir: Optional[Path] = None
transcript_files: List[Path] = []

# ------------------------------------------------------------
# 既存ジョブ
# ------------------------------------------------------------
if input_mode.startswith("既存ジョブ"):
    jobs = list_all_jobs(user_dir)
    #st.subheader("ジョブ選択（サーバー内部の保存ファイル）")
    st.markdown("##### ジョブ選択（サーバー内部の保存ファイル）")
    st.write("直近の5つのジョブより古いものは自動的に消去されます")

    if not jobs:
        st.info("minutes_app の job が見つかりません。先に文字起こしで job を作成してください。")
    else:
        labels = [j.label for j in jobs]
        picked = st.radio("対象ジョブ", options=labels, index=0, key=k("job_picker"))
        job = jobs[labels.index(picked)]
        job_dir = job.job_dir

        transcript_files = list_transcript_txts(job_dir)

        # if transcript_files:
        #     st.caption(f"job_dir: {job_dir}")
        #     st.markdown("##### 対象ファイル（処理順）")
        #     st.write([p.name for p in transcript_files])
        # else:
        #     st.error("transcript/*.txt が見つかりません。")


        if transcript_files:

            with st.expander(
                f"対象ファイル（処理順）：{len(transcript_files)}件",
                expanded=False,
            ):
                st.caption(f"job_dir: {job_dir}")

                for p in transcript_files:
                    st.markdown(f"- `{p.name}`")

        else:
            st.error(
                "transcript/*.txt が見つかりません。"
            )

# ------------------------------------------------------------
# 新規アップロード
# ------------------------------------------------------------
else:
    uploaded_files = st.file_uploader(
        "文字起こしテキスト（.txt）をドロップ/選択（複数可・アップロード順＝連続順）",
        type=["txt"],
        accept_multiple_files=True,
        key=k("uploader_txt"),
    )

    if not uploaded_files:
        st.info("テキストファイル（.txt）を1つ以上アップロードしてください。")
    else:
        sig_parts = [f"{getattr(f, 'name', 'file')}:{getattr(f, 'size', 0)}" for f in uploaded_files]
        upload_sig = "|".join(sig_parts)

        if st.session_state.get(K_UP_LAST_SIG) != upload_sig:
            st.session_state[K_UP_LAST_SIG] = upload_sig
            st.session_state[K_UP_JOB_ROOT] = None

        st.caption("※ アップロード順をそのまま連続処理の順番として使います。")
        st.write({"files": [getattr(f, "name", "(no name)") for f in uploaded_files]})

        existing_job_root = st.session_state.get(K_UP_JOB_ROOT)

        # ------------------------------------------------------------
        # 内部ヘルパ（新規job作成）
        # ------------------------------------------------------------
        def _safe_filename(name: str) -> str:
            s = (name or "text").strip()
            s = s.replace("\\", "_").replace("/", "_").replace(":", "_")
            s = re.sub(r"\s+", " ", s)
            return s or "text"

        def _now_job_id() -> str:
            return "job_" + datetime.now().strftime("%Y%m%d_%H%M%S")

        if existing_job_root:
            job_dir = Path(existing_job_root)
            st.success(f"✅ 作成済みジョブを復元しました: {job_dir}")
        else:
            create_job_clicked = st.button(
                "📦 アップロードを transcript/ として保存して新規ジョブを作成",
                type="primary",
                key=k("create_job_btn"),
            )

            if create_job_clicked:
                with st.spinner("新規ジョブを作成して transcript/ に保存しています…"):
                    today_dir = datetime.now().strftime("%Y-%m-%d")
                    job_id = _now_job_id()
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
                        name = _safe_filename(getattr(uf, "name", f"text_{i}.txt"))
                        out_name = f"{i:03d}_{name}"
                        out_path = transcript_dir / out_name

                        b = uf.getvalue()
                        try:
                            text0 = b.decode("utf-8")
                        except UnicodeDecodeError:
                            text0 = b.decode("cp932", errors="ignore")

                        write_text(out_path, text0)
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
                        "status": {"speaker_separate": "not_started"},
                        "transcript_index": transcript_index,
                    }
                    write_text(job_root / "job.json", json.dumps(job_json, ensure_ascii=False, indent=2))
                    append_log(log_path, "UPLOAD->JOB DONE")

                st.session_state[K_UP_JOB_ROOT] = str(job_root)
                job_dir = job_root

                st.success(f"✅ 新規ジョブを作成しました: {job_root}")
                st.caption(f"保存先: {job_root}")

        if job_dir:
            transcript_files = list_transcript_txts(job_dir)
            # if transcript_files:
            #     st.caption(f"job_dir: {job_dir}")
            #     st.markdown("### 対象ファイル（処理順）")
            #     st.write([p.name for p in transcript_files])
            # else:
            #     st.error("transcript/*.txt が見つかりません。保存を確認してください。")

            if transcript_files:
                with st.expander(
                    f"対象ファイル（処理順）：{len(transcript_files)}件",
                    expanded=False,
                ):
                    st.caption(f"job_dir: {job_dir}")

                    for p in transcript_files:
                        st.markdown(f"- `{p.name}`")

            else:
                st.error(
                    "transcript/*.txt が見つかりません。保存を確認してください。"
                )
# ============================================================
# job_ready / 実行用 state 確定
# ============================================================
job_ready = bool(job_dir) and bool(transcript_files)
st.session_state[k("job_ready")] = job_ready
st.session_state[k("speaker_job_dir")] = str(job_dir) if job_dir else ""
st.session_state[k("speaker_transcript_files")] = [str(p) for p in transcript_files] if transcript_files else []

if not job_ready:
    st.warning("まだ実行できません。上で job を確定してください。（既存ジョブ選択 or 新規アップロード→作成ボタン）")

# ============================================================
# ③ 実行（AI実行 + busy 記録）
# ============================================================
st.divider()
st.subheader("② 話者分離の実行")

batch_btn = st.button(
    "▶️ 文字起こしテキストを順番に話者分離（ストレージ保存＋連結）",
    type="primary",
    key=k("batch_btn"),
    disabled=not job_ready,
    help="入力（既存ジョブ選択 or 新規アップロード→ジョブ作成）が完了すると押せます。",
)

if batch_btn:
    # ------------------------------------------------------------
    # ガード：job_ready
    # ------------------------------------------------------------
    if not st.session_state.get(k("job_ready"), False):
        st.error("job が未確定です。先に入力を確定してください。")
        st.stop()

    # ------------------------------------------------------------
    # 実行対象の確定（session_state 正本）
    # ------------------------------------------------------------
    job_dir = Path(st.session_state[k("speaker_job_dir")])
    transcript_files = [Path(p) for p in st.session_state[k("speaker_transcript_files")]]

    # ------------------------------------------------------------
    # 出力先（storage）
    # ------------------------------------------------------------
    speaker_sep_dir = job_dir / "transcript_speaker_separated"
    speaker_comb_dir = job_dir / "transcript_speaker_separated_combined"
    logs_dir = job_dir / "logs"
    safe_mkdir(speaker_sep_dir)
    safe_mkdir(speaker_comb_dir)
    safe_mkdir(logs_dir)

    log_path = logs_dir / "process.log"

    action_name = "speaker_prep_batch"

    meta_start = {
        "feature": "minutes_speaker_separate_storage",
        "action": action_name,
        "input_mode": ("job" if input_mode.startswith("既存") else "upload_txt"),
        "job_dir": str(job_dir),
        "file_count": len(transcript_files),
        "provider": str(provider),
        "model": str(model),
        "prompt_level": str(prompt_level),
    }

    # ------------------------------------------------------------
    # busy（ai_runs.db）：1バッチ=1run
    # ------------------------------------------------------------
    try:
        with busy_run(
            projects_root=PROJECTS_ROOT,
            user_sub=str(sub),
            app_name=str(APP_NAME),
            page_name=str(PAGE_NAME),
            task_type="text",
            provider=str(provider),
            model=str(model),
            meta=meta_start,
        ) as br:
            with st.spinner("話者分離を実行中です。しばらくお待ちください…"):
                ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
                combined_path = speaker_comb_dir / f"transcript_speaker_separated_combined_{ts_tag}.txt"

                combined_chunks: List[str] = []
                total_in = 0
                total_out = 0

                # 合計cost（「返ってきたもの」を足すだけ。推計しない）
                total_usd: Optional[float] = 0.0
                total_jpy: Optional[float] = 0.0

                # 表示用（CostResultをそのまま受ける）：最後に1つだけ保持
                last_cost_obj = None  # CostResult or None

                total_elapsed = 0.0
                processed_count = 0

                append_log(log_path, "SPEAKER PREP BATCH START")
                append_log(log_path, f"job_dir={job_dir}")
                append_log(log_path, f"count={len(transcript_files)} model={model} provider={provider}")
                append_log(log_path, f"prompt_level={prompt_level}")

                prog = st.progress(0)
                status = st.empty()

                # ------------------------------------------------------------
                # 連続処理（単発なし）
                # ------------------------------------------------------------
                for i, src_path in enumerate(transcript_files, start=1):
                    status.write(f"{i}/{len(transcript_files)} 処理中: {src_path.name}")

                    src = src_path.read_text(encoding="utf-8", errors="ignore").strip()
                    if not src:
                        prog.progress(int(i / len(transcript_files) * 100))
                        continue

                    prompt = build_prompt(
                        st.session_state[k("mandatory_prompt")],
                        st.session_state[k("preset_text")],
                        st.session_state[k("extra_text")],
                        src,
                    )

                    t0 = time.perf_counter()
                    res = run_speaker_prep_one(prompt=prompt, provider=str(provider), model=str(model))
                    elapsed = time.perf_counter() - t0

                    text = getattr(res, "text", "") or ""

                    out_path = speaker_sep_dir / f"{i:03d}_{src_path.stem}_speaker.txt"
                    write_text(out_path, text or "")

                    combined_chunks.append(text or "")
                    if i < len(transcript_files):
                        combined_chunks.append(make_connector_line(src_path.name))

                    total_elapsed += float(elapsed)
                    processed_count += 1

                    # ------------------------------------------------------------
                    # usage（取れたら合計に足す：推計しない）
                    # ------------------------------------------------------------
                    usage = getattr(res, "usage", None)
                    in_tok = getattr(usage, "input_tokens", None) if usage is not None else None
                    out_tok = getattr(usage, "output_tokens", None) if usage is not None else None

                    if isinstance(in_tok, int):
                        total_in += int(in_tok)
                    if isinstance(out_tok, int):
                        total_out += int(out_tok)

                    # ------------------------------------------------------------
                    # cost（取れたら合計に足す：推計しない）
                    # ------------------------------------------------------------
                    cost = getattr(res, "cost", None)
                    usd = getattr(cost, "usd", None) if cost is not None else None
                    jpy = getattr(cost, "jpy", None) if cost is not None else None

                    if cost is not None:
                        last_cost_obj = cost  # 表示用に最後のCostResultを保持（計算しない）

                    if (usd is not None) and (total_usd is not None):
                        total_usd += float(usd)
                    elif usd is None:
                        total_usd = None

                    if (jpy is not None) and (total_jpy is not None):
                        total_jpy += float(jpy)
                    elif jpy is None:
                        total_jpy = None

                    append_log(
                        log_path,
                        f"DONE {src_path.name} -> {out_path.name} "
                        f"in={in_tok} out={out_tok} usd={usd} jpy={jpy} elapsed={elapsed:.2f}s",
                    )

                    prog.progress(int(i / len(transcript_files) * 100))

                # ------------------------------------------------------------
                # 連結保存
                # ------------------------------------------------------------
                combined_text = "\n\n".join(combined_chunks)
                write_text(combined_path, combined_text)

                append_log(log_path, f"COMBINED -> {combined_path.name}")
                append_log(log_path, "SPEAKER PREP BATCH DONE")

                # ============================================================
                # ✅ busy への反映は「最後に1回だけ」（合計）
                # - 推計しない（res が返した usage/cost を足した結果のみ）
                # ============================================================
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

                # ------------------------------------------------------------
                # ページ表示用 state（正本）
                # ------------------------------------------------------------
                st.session_state[K_LAST_RUN_ID] = br.run_id
                st.session_state[K_LAST_RUN_ACTION] = action_name
                st.session_state[K_LAST_RESULT] = {
                    "speaker_sep_dir": str(speaker_sep_dir),
                    "combined_path": str(combined_path),
                    "total_in": int(total_in),
                    "total_out": int(total_out),
                    "total_usd": (float(total_usd) if total_usd is not None else None),
                    "total_jpy": (float(total_jpy) if total_jpy is not None else None),
                    "processed_count": int(processed_count),
                }

                st.session_state[K_LAST_TOTAL_IN] = int(total_in)
                st.session_state[K_LAST_TOTAL_OUT] = int(total_out)

                # ------------------------------------------------------------
                # 合計CostResult（計算しない：返ってきた合計を詰め直すだけ）
                # ------------------------------------------------------------
                if (total_usd is not None) and (total_jpy is not None) and (last_cost_obj is not None):
                    try:
                        from common_lib.ai.types import CostResult  # 正本型

                        st.session_state[K_LAST_TOTAL_COST] = CostResult(
                            usd=float(total_usd),
                            jpy=float(total_jpy),
                            usd_jpy=float(getattr(last_cost_obj, "usd_jpy")),
                            fx_source=str(getattr(last_cost_obj, "fx_source")),
                        )
                    except Exception:
                        st.session_state[K_LAST_TOTAL_COST] = None
                else:
                    st.session_state[K_LAST_TOTAL_COST] = None

                st.session_state[K_LAST_MODEL] = str(model)
                st.session_state[K_LAST_PROVIDER] = str(provider)

        # ------------------------------------------------------------
        # 完了UI
        # ------------------------------------------------------------
        st.success("完了しました。ストレージに保存しました。")

        with st.expander("📊 処理結果（保存先・トークン・料金）", expanded=False):
            st.write(st.session_state.get(K_LAST_RESULT))

        st.download_button(
            "連結結果をダウンロード（.txt）",
            data=(combined_text or "").encode("utf-8"),
            file_name=Path(st.session_state[K_LAST_RESULT]["combined_path"]).name
            if st.session_state.get(K_LAST_RESULT)
            else f"speaker_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            key=k("download_combined"),
        )

    except Exception as e:
        st.error(f"実行に失敗しました: {e}")
        st.stop()

# ============================================================
# 📊 実行サマリ（テンプレ準拠：1ブロック表示）
# - run_id / tokens / cost / time を 1つに統合
# ============================================================
#st.divider()
st.markdown("#### 📊 実行サマリ")

render_run_summary_compact(
    projects_root=PROJECTS_ROOT,
    run_id=str(st.session_state.get(K_LAST_RUN_ID) or "").strip(),
    model=str(st.session_state.get(K_LAST_MODEL) or "").strip() or str(model),
    in_tokens=st.session_state.get(K_LAST_TOTAL_IN),
    out_tokens=st.session_state.get(K_LAST_TOTAL_OUT),
    cost=st.session_state.get(K_LAST_TOTAL_COST),
    note="",
    show_divider=False,
)
