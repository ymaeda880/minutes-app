# pages/04_話者分離（Gemini）.py
# ------------------------------------------------------------
# 🎙️ 話者分離・整形（議事録の前処理）
# - OpenAI / Gemini 両対応
# - sidebar は radio ボタン
# - default: gpt-5-mini
# - gpt-5 は除外
# ------------------------------------------------------------
from __future__ import annotations

import time
from typing import Dict, Any
import datetime as dt
import json

import streamlit as st
from openai import OpenAI

# ===== Gemini =====
import google.generativeai as genai

# ==== 共通ユーティリティ ====
from lib.costs import estimate_chat_cost_usd
from lib.tokens import extract_tokens_from_response
from lib.prompts import SPEAKER_PREP, get_group, build_prompt
from config.config import (
    DEFAULT_USDJPY,
    get_gemini_api_key,
    has_gemini_api_key,
    estimate_tokens_from_text,
    estimate_gemini_cost_usd,
)
from ui.style import disable_heading_anchors
from lib.explanation import render_speaker_prep_expander

# ========================== 共通設定 ==========================
st.set_page_config(page_title="③ 話者分離・整形（Gemini対応）", page_icon="🎙️", layout="wide")
disable_heading_anchors()
st.title("話者分離（Gemini対応）")

render_speaker_prep_expander()

# ===== OpenAI =====
OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key") or st.secrets.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ===== Gemini =====
GEMINI_ENABLED = has_gemini_api_key()
if GEMINI_ENABLED:
    genai.configure(api_key=get_gemini_api_key())

# ========================== UI ==========================
left, right = st.columns([1, 1], gap="large")

# ---- プロンプト ----
with left:
    st.subheader("プロンプト")

    group = get_group(SPEAKER_PREP)

    st.session_state.setdefault("mandatory_prompt", group.mandatory_default)
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

    run_btn = st.button("話者分離して整形", type="primary", use_container_width=True)

# ---- 入力 ----
with right:
    st.subheader("入力テキスト")

    up = st.file_uploader("文字起こしテキスト（.txt）", type=["txt"])
    if up:
        raw = up.read()
        try:
            st.session_state["prep_source_text"] = raw.decode("utf-8")
        except UnicodeDecodeError:
            st.session_state["prep_source_text"] = raw.decode("cp932", errors="ignore")
        st.session_state["prep_input_filename"] = up.name

    st.text_area(
        "文字起こしテキスト（貼り付け可）",
        height=420,
        key="prep_source_text",
    )

# ========================== Sidebar ==========================
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

    st.subheader("通貨換算")
    usd_jpy = st.number_input(
        "USD/JPY",
        min_value=50.0,
        max_value=500.0,
        value=float(DEFAULT_USDJPY),
        step=0.5,
    )

# ========================== 実行 ==========================
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

    # ================= 出力 =================
    if text.strip():
        st.markdown("### ✅ 整形結果")
        st.markdown(text)
    else:
        st.warning("⚠️ 空の応答が返りました。")
        try:
            st.json(resp)
        except Exception:
            pass

    JST = dt.timezone(dt.timedelta(hours=9))
    now = dt.datetime.now(JST).strftime("%Y%m%d_%H%M%S")
    fname = f"speaker_prep_{now}.txt"

    st.download_button(
        "📝 ダウンロード",
        data=(text or "").encode("utf-8"),
        file_name=fname,
        mime="text/plain",
        use_container_width=True,
    )

    st.subheader("📊 処理・料金")
    st.table({
        "処理時間": [f"{elapsed:.2f} 秒"],
        "入力tokens": [in_tok],
        "出力tokens": [out_tok],
        "概算料金": [f"${usd:,.6f} / ¥{jpy:,.2f}" if usd is not None else "—"],
        "モデル": [model],
    })

    st.session_state["minutes_source_text"] = text

