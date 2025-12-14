# ui/sidebar.py
import streamlit as st

def init_metrics_state():
    if "metrics" not in st.session_state:
        st.session_state["metrics"] = dict(
            whisper_time=None, whisper_cost_usd=None,
            chat_time=None, chat_cost_usd=None,
            ptok=0, ctok=0
        )

def render_sidebar():
    with st.sidebar:
        st.header("⏱️ 処理 & 概算料金")

        # 為替（全ページ共通で編集可）
        st.subheader("通貨設定")
        st.session_state["usd_jpy"] = st.number_input(
            "USD/JPY（手動設定）",
            min_value=50.0, max_value=500.0,
            value=float(st.session_state.get("usd_jpy", 150.0)),
            step=0.5
        )
        fx = float(st.session_state["usd_jpy"])
        m = st.session_state["metrics"]

        # Whisper
        st.metric("Whisper 処理時間", f"{m['whisper_time']:.2f} 秒" if m["whisper_time"] is not None else "—")
        if m["whisper_cost_usd"] is not None:
            st.metric("Whisper 概算料金 (USD)", f"${m['whisper_cost_usd']:,.6f}")
            st.metric("Whisper 概算料金 (JPY)", f"¥{m['whisper_cost_usd']*fx:,.2f}")
        else:
            st.metric("Whisper 概算料金 (USD)", "—")
            st.metric("Whisper 概算料金 (JPY)", "—")

        st.divider()

        # Chat
        st.metric("Chat 処理時間", f"{m['chat_time']:.2f} 秒" if m["chat_time"] is not None else "—")
        if m["chat_cost_usd"] is not None:
            st.metric("Chat 概算料金 (USD)", f"${m['chat_cost_usd']:,.6f}")
            st.metric("Chat 概算料金 (JPY)", f"¥{m['chat_cost_usd']*fx:,.2f}")
        else:
            st.metric("Chat 概算料金 (USD)", "—")
            st.metric("Chat 概算料金 (JPY)", "—")

        st.caption(f"入力: {m['ptok']:,} / 出力: {m['ctok']:,} tokens")
