# pages/01_前処理（文分割＋句点改行）.py
import streamlit as st
from datetime import datetime
from lib.utils_text import (
    sentence_split_with_inferred_periods, sentence_split_by_period,
    add_line_numbers, post_process
)

st.set_page_config(page_title="前処理（文分割＋句点改行）", page_icon="📝", layout="wide")
st.title("📝 前処理（文分割＋句点改行）")

with st.sidebar:
    st.header("前処理オプション")
    do_sentence_split = st.checkbox("文ごとに改行（句点推測＋補完）", value=True)
    do_period_split = st.checkbox("句点ごとに必ず改行", value=True)
    trim_spaces = st.checkbox("前後の空白を詰める", value=True)
    remove_empty_lines = st.checkbox("空行を詰める（3連→2連）", value=True)

st.subheader("1. 入力")
col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("テキストファイル（.txt）をアップロード", type=["txt"])
    raw_text = uploaded.read().decode("utf-8", errors="replace") if uploaded else ""
with col2:
    pasted = st.text_area("ここにテキストを貼り付け", height=200)

source = raw_text if uploaded else pasted

if not source:
    st.info("テキストを入力してください。")
else:
    # 前処理パイプライン
    text = sentence_split_with_inferred_periods(source) if do_sentence_split else source
    text = sentence_split_by_period(text) if do_period_split else text
    final = post_process(text, trim_spaces, remove_empty_lines)

    st.subheader("2. Before / After（前処理）")
    c1, c2 = st.columns(2)
    with c1:
        st.text_area("Before（原文）", add_line_numbers(source), height=300, label_visibility="collapsed")
    with c2:
        st.text_area("After（文分割＋句点改行）", add_line_numbers(final), height=300, label_visibility="collapsed")

    # 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "💾 前処理テキストを保存",
        data=final.encode("utf-8"),
        file_name=f"preprocessed_{ts}.txt",
        use_container_width=True
    )
