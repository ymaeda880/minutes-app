# pages/02_後処理（【…】削除のみ）.py
import streamlit as st
from datetime import datetime
from lib.utils_text import (
    add_line_numbers, post_process,
    strip_bracketed, build_line_diff, build_sentence_diff,
    LINE_DIFF_STYLE, SENT_DIFF_STYLE
)

st.set_page_config(page_title="後処理（【…】削除のみ）", page_icon="✂️", layout="wide")
st.title("✂️ 後処理（【…】削除のみ）")

with st.sidebar:
    st.header("削除の設定")
    enabled_pairs = []
    if st.checkbox("【…】 を削除", value=True):
        enabled_pairs.append(("【", "】"))
    non_greedy = st.checkbox("最短一致で削除", value=True)
    multiline_spanning = st.checkbox("改行をまたぐ括弧も削除", value=True)

    st.header("後処理オプション")
    trim_spaces = st.checkbox("前後の空白を詰める", value=True)
    remove_empty_lines = st.checkbox("空行を詰める（3連→2連）", value=True)

    st.divider()
    show_sentence_diff = st.checkbox("文単位（変更文のみ）で差分表示", value=True)
    show_only_changed_line = st.checkbox("（行単位）変更行のみ表示", value=False)

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
    before = source

    # 【…】削除
    stripped, counts = strip_bracketed(
        before,
        pairs=enabled_pairs if enabled_pairs else [("【", "】")],
        non_greedy=non_greedy,
        multiline_spanning=multiline_spanning
    )

    after = post_process(stripped, trim_spaces, remove_empty_lines)

    st.subheader("2. Before / After（削除処理）")
    c1, c2 = st.columns(2)
    with c1:
        st.text_area("Before", add_line_numbers(before), height=300, label_visibility="collapsed")
    with c2:
        st.text_area("After", add_line_numbers(after), height=300, label_visibility="collapsed")

    # 差分：文単位
    if show_sentence_diff:
        st.subheader("3. 差分ビュー（文単位：変更文のみ）")
        rows2 = build_sentence_diff(before, after)
        st.markdown(SENT_DIFF_STYLE, unsafe_allow_html=True)
        st.write("凡例: ~変更 -削除 +追加（文単位）")
        for status, b_html, a_html in rows2:
            st.markdown(
                f"<div class='sdiffrow'><div>{status}</div><div>{b_html}</div><div>{a_html}</div></div>",
                unsafe_allow_html=True
            )

    # 差分：行単位
    st.subheader("4. 差分ビュー（行単位）")
    rows = build_line_diff(before, after)
    st.markdown(LINE_DIFF_STYLE, unsafe_allow_html=True)
    st.write("凡例: =同一 ~変更 -削除 +追加")
    if show_only_changed_line:
        rows = [r for r in rows if r[0] != "="]
    for status, b_no, a_no, b_html, a_html in rows:
        st.markdown(
            f"<div class='diffrow'>"
            f"<div>{status}</div><div>{b_no or ''}</div><div>{b_html}</div>"
            f"<div>{a_no or ''}</div><div>{a_html}</div></div>",
            unsafe_allow_html=True
        )

    # 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "💾 削除後テキストを保存",
        data=after.encode("utf-8"),
        file_name=f"postprocessed_{ts}.txt",
        use_container_width=True
    )
