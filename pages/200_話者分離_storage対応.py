# -*- coding: utf-8 -*-
# pages/200_話者分離_storage対応.py
# ------------------------------------------------------------
# 🎙️ 話者分離・整形（議事録の前処理）storage対応（ログイン必須）
# - ログイン確認（pages/13 と同じ）
# - Storages/<user>/minutes_app/ 配下の「文字起こしテキスト」を列挙し、radio で引き継ぎ
#   既定は transcript/transcripts_combined_*.txt（無ければ transcript/*.txt から候補）
# - OpenAI / Gemini 両対応
# - sidebar はモデル radio
# - default: gpt-5-mini（gpt-5 は除外）
# - 生成結果は選択ジョブの transcript/ に保存
# - job の logs/process.log にも軽く書く
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
from typing import Any, Optional, List, Dict

import streamlit as st
from openai import OpenAI

# ===== Gemini =====
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

#from common_lib.auth.auth_helpers import get_current_user_from_session_or_cookie


# ============================================================
# paths（PROJECTS_ROOT 基準）
# ============================================================
# _THIS = Path(__file__).resolve()
# PROJECTS_ROOT = _THIS.parents[3]
# STORAGES_ROOT = PROJECTS_ROOT / "Storages"


# ============================================================
# sys.path 調整（common_lib を import できるように）
# - 暗黙の推測を避けるため、存在確認して無ければエラーで停止
# ============================================================
import sys
from pathlib import Path
import streamlit as st  # ← st.stop したいのでここで import

_THIS = Path(__file__).resolve()

# 期待する構造：
#   .../projects/minutes_project/minutes_app/pages/22_*.py
# よって projects ルートは parents[3]
PROJECTS_ROOT = _THIS.parents[3]
COMMON_LIB_DIR = PROJECTS_ROOT / "common_lib"
STORAGES_ROOT = PROJECTS_ROOT / "Storages"

if not COMMON_LIB_DIR.exists():
    st.error(
        "common_lib が見つかりません。\n"
        f"- expected: {COMMON_LIB_DIR}\n"
        f"- this file: {_THIS}\n"
        f"- projects_root: {PROJECTS_ROOT}\n"
        "配置（common_lib の場所）か parents[] の段数を確認してください。"
    )
    st.stop()

if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))


from common_lib.auth.auth_helpers import get_current_user_from_session_or_cookie

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
# storage listing（文字起こし → 話者分離の引き継ぎ）
#
# 優先:
#   transcript/transcripts_combined_*.txt
# 次点:
#   transcript/*.txt（ただし transcript_marked 配下は除外）
# ============================================================
@dataclass
class SourceItem:
    label: str
    path: Path
    job_dir: Path
    transcript_dir: Path
    job_id: str
    date: str
    created_at: Optional[str]


def list_transcript_sources(user_dir: str) -> list[SourceItem]:
    base = STORAGES_ROOT / user_dir / "minutes_app"
    if not base.exists():
        return []

    items: list[SourceItem] = []

    for day_dir in sorted(base.glob("*"), reverse=True):
        if not day_dir.is_dir():
            continue

        for job_dir in sorted(day_dir.glob("job_*"), reverse=True):
            if not job_dir.is_dir():
                continue

            meta = _read_job_json(job_dir)
            job_id = str(meta.get("job_id") or job_dir.name)
            date = str(meta.get("date") or day_dir.name)
            created_at = meta.get("created_at")

            transcript_dir = job_dir / "transcript"
            if not transcript_dir.exists():
                continue

            # 1) combined 優先
            combined = sorted(
                transcript_dir.glob("transcripts_combined_*.txt"),
                key=lambda p: p.name.lower(),
                reverse=True,
            )

            # 2) 無ければ transcript 直下の .txt を候補（transcript_marked 等は除外）
            fallback: list[Path] = []
            if not combined:
                for p in sorted(transcript_dir.glob("*.txt"), reverse=True):
                    if "marked" in p.name.lower():
                        continue
                    fallback.append(p)

            candidates = combined if combined else fallback
            for p in candidates:
                label = f"{date} / {job_id} / {p.name} / created={_human_dt(created_at)}"
                items.append(
                    SourceItem(
                        label=label,
                        path=p,
                        job_dir=job_dir,
                        transcript_dir=transcript_dir,
                        job_id=job_id,
                        date=date,
                        created_at=created_at,
                    )
                )

    return items


def read_text_guess_encoding(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="cp932", errors="replace")


# ============================================================
# UI 共通設定
# ============================================================
st.set_page_config(page_title="③ 話者分離・整形（storage対応）", page_icon="🎙️", layout="wide")
disable_heading_anchors()
st.title("🎙️ 話者分離・整形（storage対応）")

render_speaker_prep_expander()

st.markdown(
    """
- **ログイン必須**（Cookie/JWT）
- 「文字起こし（storage対応）」で作られた **transcript/** のテキストを **radio で選んで引き継ぎ** できます  
- 話者分離結果は、同じジョブの **transcript/** に保存します（次の重複検出へ渡しやすい命名）
"""
)

# ============================================================
# ログイン
# ============================================================
current_user, _payload = get_current_user_from_session_or_cookie(st)

col_a, col_b = st.columns([2, 1], vertical_alignment="center")
with col_a:
    if current_user:
        st.success(f"ログイン中: **{current_user}**")
    else:
        st.error("未ログイン（ポータルでログイン後に再読み込みしてください）")
# with col_b:
#     show_debug = st.toggle("🔍 デバッグ", value=False)

# if show_debug:
#     with st.expander("🔍 デバッグ（最小）", expanded=True):
#         st.write(
#             {
#                 "THIS": str(_THIS),
#                 "PROJECTS_ROOT": str(PROJECTS_ROOT),
#                 "STORAGES_ROOT": str(STORAGES_ROOT),
#                 "current_user": current_user,
#             }
#         )

if not current_user:
    st.stop()

user_dir = _sanitize_username_for_path(str(current_user))


# ============================================================
# OpenAI / Gemini init
# ============================================================
OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key") or st.secrets.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

GEMINI_ENABLED = has_gemini_api_key()
if GEMINI_ENABLED:
    genai.configure(api_key=get_gemini_api_key())


# ============================================================
# Sidebar（モデル/通貨）
# ============================================================
with st.sidebar:
    st.subheader("モデル設定")

    MODEL_OPTIONS = [
        "gpt-5-mini",
        "gpt-5-nano",
        "gemini-2.0-flash",
    ]

    st.session_state.setdefault("speaker_model", "gpt-5-mini")

    model = st.radio(
        "モデル",
        MODEL_OPTIONS,
        key="speaker_model",
    )

    if model.startswith("gemini") and not GEMINI_ENABLED:
        st.warning("Gemini API Key が未設定のため使用できません")
        st.stop()

    max_completion_tokens = 100000

    st.subheader("プロンプト設定")

    PROMPT_LEVEL_OPTIONS = [
        "標準（精度優先）",
        "軽量（タイムアウト低減）",
        "超軽量（最小負荷）",
    ]

    # デフォルトは「標準」
    st.session_state.setdefault("speaker_prompt_level", PROMPT_LEVEL_OPTIONS[2])

    prompt_level = st.radio(
        "話者分離プロンプト",
        PROMPT_LEVEL_OPTIONS,
        key="speaker_prompt_level",
    )

    st.subheader("通貨換算")
    usd_jpy = st.number_input(
        "USD/JPY",
        min_value=50.0,
        max_value=500.0,
        value=float(DEFAULT_USDJPY),
        step=0.5,
    )


# ============================================================
# メインUI（左=プロンプト/実行、右=入力）
#  - 右上に「storage引き継ぎ」のradio
#  - 実行ボタンはメイン（左）に配置
# ============================================================
left, right = st.columns([1, 1], gap="large")

# ---- 入力（右）----
with right:
    st.subheader("入力テキスト")

    source_mode = st.radio(
        "入力元",
        ["storage から引き継ぐ（推奨）", "ファイルアップロード", "貼り付け（直接入力）"],
        index=0,
    )

    if source_mode.startswith("storage"):
        items = list_transcript_sources(user_dir)
        if not items:
            st.info(
                "Storages に文字起こしテキストが見つかりません。\n\n"
                "先に「文字起こし（storage対応）」で transcript を作成してください。"
            )
            st.stop()

        labels = [it.label for it in items]
        picked = st.radio("処理対象（文字起こし txt）", options=labels, index=0)
        it = items[labels.index(picked)]

        st.caption(f"選択ファイル: {it.path}")

        with st.expander("📌 選択ジョブ情報", expanded=False):
            st.write(
                {
                    "job_dir": str(it.job_dir),
                    "transcript_dir": str(it.transcript_dir),
                    "job_id": it.job_id,
                    "date": it.date,
                    "created_at": it.created_at,
                }
            )

        # 読み込み（選択時点でプレビューできるように）
        st.session_state["prep_source_text"] = read_text_guess_encoding(it.path)
        st.session_state["prep_input_filename"] = it.path.name
        st.session_state["prep_job_dir"] = str(it.job_dir)

    elif source_mode.startswith("ファイル"):
        up = st.file_uploader("文字起こしテキスト（.txt）", type=["txt"])
        if up:
            raw = up.read()
            try:
                st.session_state["prep_source_text"] = raw.decode("utf-8")
            except UnicodeDecodeError:
                st.session_state["prep_source_text"] = raw.decode("cp932", errors="ignore")
            st.session_state["prep_input_filename"] = up.name
            st.session_state.pop("prep_job_dir", None)

    else:
        # 直接入力
        st.session_state.setdefault("prep_source_text", "")
        st.session_state.setdefault("prep_input_filename", "pasted.txt")
        st.session_state.pop("prep_job_dir", None)

    st.text_area(
        "文字起こしテキスト（編集可）",
        height=420,
        key="prep_source_text",
    )

# ---- プロンプト（左）----
with left:
    st.subheader("プロンプト")

    group = get_group(SPEAKER_PREP)

    # ---- prompt_level（sidebar）に応じた mandatory の候補 ----
    _level = st.session_state.get("speaker_prompt_level", "標準（精度優先）")
    if _level == "標準（精度優先）":
        _mandatory_default = SPEAKER_MANDATORY
    elif _level == "軽量（タイムアウト低減）":
        _mandatory_default = SPEAKER_MANDATORY_LIGHT
    else:
        _mandatory_default = SPEAKER_MANDATORY_LIGHTER

    # ★ 初回だけセット（未設定なら入れる）
    st.session_state.setdefault("mandatory_prompt", _mandatory_default)

    # ---- radio変更時だけ mandatory_prompt を自動で切り替える ----
    prev_level = st.session_state.get("_speaker_prompt_level_prev")
    if prev_level != _level:
        st.session_state["mandatory_prompt"] = _mandatory_default
        st.session_state["_speaker_prompt_level_prev"] = _level

    # 以降、プリセット周りは従来通り
    st.session_state.setdefault("preset_label", group.label_for_key(group.default_preset_key))
    st.session_state.setdefault("preset_text", group.body_for_label(st.session_state["preset_label"]))
    st.session_state.setdefault("extra_text", "")

    with st.expander("必ず入る部分（常に先頭）", expanded=False):
        st.text_area(
            "必須プロンプト",
            height=220,
            key="mandatory_prompt",
            label_visibility="collapsed",
        )

    def _on_change_preset():
        st.session_state["preset_text"] = group.body_for_label(st.session_state["preset_label"])

    st.selectbox(
        "追記プリセット",
        options=group.preset_labels(),
        key="preset_label",
        on_change=_on_change_preset,
    )

    st.text_area("プリセット本文（編集可）", height=120, key="preset_text")
    st.text_area("追加指示（任意）", height=88, key="extra_text")

    run_btn = st.button("話者分離して整形（保存も行う）", type="primary")

# ============================================================
# 実行
# ============================================================
if run_btn:
    src = st.session_state.get("prep_source_text", "").strip()
    if not src:
        st.warning("文字起こしテキストを入力してください。")
        st.stop()

    prompt = build_prompt(
        st.session_state["mandatory_prompt"],
        st.session_state["preset_text"],
        st.session_state["extra_text"],
        src,
    )

    t0 = time.perf_counter()

    # どのジョブに保存するか（storage引継ぎ以外は保存先が無いのでダウンロードのみ）
    job_dir_str = st.session_state.get("prep_job_dir")
    job_dir = Path(job_dir_str) if job_dir_str else None

    # 保存名のための入力ファイル名
    input_name = st.session_state.get("prep_input_filename") or "transcript.txt"
    input_stem = Path(input_name).stem

    with st.spinner("話者分離・整形を実行中…"):
        # -------- Gemini --------
        if model.startswith("gemini"):
            gem = genai.GenerativeModel(model)
            resp = gem.generate_content(prompt)
            text = getattr(resp, "text", "") or ""
            elapsed = time.perf_counter() - t0

            out_tok = estimate_tokens_from_text(text)
            in_tok = estimate_tokens_from_text(prompt)
            usd = estimate_gemini_cost_usd(
                model=model,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
            jpy = (usd * usd_jpy) if usd is not None else None

        # -------- OpenAI --------
        else:
            if client is None:
                st.error("OPENAI_API_KEY が未設定のため OpenAI モデルを使用できません。")
                st.stop()

            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=max_completion_tokens,
            )
            elapsed = time.perf_counter() - t0
            text = resp.choices[0].message.content or ""

            in_tok, out_tok, _ = extract_tokens_from_response(resp)
            usd = estimate_chat_cost_usd(model, in_tok, out_tok)
            jpy = (usd * usd_jpy) if usd is not None else None

    # ================= 出力表示 =================
    if text.strip():
        st.markdown("### ✅ 整形結果")
        st.markdown(text)
    else:
        st.warning("⚠️ 空の応答が返りました。")
        try:
            st.json(resp)
        except Exception:
            pass

    # ================= 保存（storage引継ぎ時のみ）=================
    saved_txt: Optional[Path] = None
    saved_log: Optional[Path] = None

    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")


    if job_dir is not None and job_dir.exists():
        transcript_dir = job_dir / "transcript"              # 入力（生文字起こし）
        speaker_dir = job_dir / "transcript_speaker"          # ★ 話者分離の出力
        logs_dir = job_dir / "logs"

        safe_mkdir(transcript_dir)
        safe_mkdir(speaker_dir)
        safe_mkdir(logs_dir)

        base_prefix = "transcripts_combined_"
        if input_name.startswith(base_prefix):
            out_stem = f"{input_stem}_speaker_{ts_tag}"
            out_name = f"{out_stem}.txt"
        else:
            out_name = f"{base_prefix}{input_stem}_speaker_{ts_tag}.txt"

        saved_txt = speaker_dir / out_name
        write_text(saved_txt, text or "")

        saved_log = speaker_dir / f"{saved_txt.stem}_speaker_log.json"
        write_json(
            saved_log,
            {
                "input": input_name,
                "output_text": str(saved_txt),
                "output_log": str(saved_log),
                "model": model,
                "elapsed_sec": float(elapsed),
                "tokens": {"input": int(in_tok), "output": int(out_tok)},
                "cost": {
                    "usd": float(usd) if usd is not None else None,
                    "jpy": float(jpy) if jpy is not None else None,
                },
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "user": str(current_user),
                "job_dir": str(job_dir),
                "prompt_meta": {
                    "mandatory_len": len(st.session_state.get("mandatory_prompt", "")),
                    "preset_len": len(st.session_state.get("preset_text", "")),
                    "extra_len": len(st.session_state.get("extra_text", "")),
                    "src_len": len(src),
                },
            },
        )

        log_path = logs_dir / "process.log"
        append_log(log_path, "SPEAKER PREP START")
        append_log(log_path, f"input={input_name}")
        append_log(log_path, f"output={saved_txt.name}")
        append_log(log_path, f"log={saved_log.name}")
        append_log(log_path, f"model={model} in_tok={in_tok} out_tok={out_tok}")
        append_log(log_path, "SPEAKER PREP DONE")


        st.success("処理が完了しました（storage に保存しました）。")
        st.markdown("### 💾 保存先（storage）")
        st.write({"speaker_txt": str(saved_txt), "log_json": str(saved_log)})

    else:
        st.info("※ 今回は storage 引き継ぎではないため、storage への保存は行いません（ダウンロードのみ）。")

    # ================= ダウンロード =================
    dl_name = f"speaker_prep_{ts_tag}.txt"
    st.download_button(
        "📝 整形結果を（パソコンに）ダウンロード（.txt）",
        data=(text or "").encode("utf-8"),
        file_name=dl_name,
        mime="text/plain",
    )

    # ================= 料金/トークン =================
    st.subheader("📊 処理・料金")
    st.table(
        {
            "処理時間": [f"{elapsed:.2f} 秒"],
            "入力tokens": [in_tok],
            "出力tokens": [out_tok],
            "概算料金": [f"${usd:,.6f} / ¥{jpy:,.2f}" if usd is not None else "—"],
            "モデル": [model],
            "保存": [str(saved_txt) if saved_txt else "—"],
        }
    )

    # 次工程に渡す（既存の流れを維持）
    st.session_state["minutes_source_text"] = text
