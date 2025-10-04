# app.py
# =========================================================
# 🎙️ Whisper Transcribe + 📝 Minutes Maker (All Python)
# - ① 音声→テキスト（Whisper API）
# - ② テキスト→議事録（Chat Completions）
# - 処理時間 & 概算料金（USD/JPY）を本文＋サイドバーに表示
# - JPYは小数点2桁まで正確表示／為替はサイドバーで可変
# - サイドバーはファイル末尾で描画（更新値が反映されやすい）
# - 「議事録タブへ引き継ぐ」ボタン付き（繋ぎ）
# ---------------------------------------------------------
# 事前準備：
# 1) .streamlit/secrets.toml に OPENAI_API_KEY を設定
# 2) pip install streamlit requests mutagen audioread
# 3) （任意）secrets に USDJPY=150.0 を入れると初期レートを変更可能
# =========================================================

import io
import os
import time
import tempfile
import requests
import streamlit as st

# ====== UI 先に設定 ======
st.set_page_config(page_title="Whisper + Minutes (All Python)", layout="wide")
st.title("🎙️ Whisper Transcribe + 📝 Minutes Maker (All Python)")

# ====== API キー ======
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY が .streamlit/secrets.toml に設定されていません。")
    st.stop()

OPENAI_TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

# ====== Pricing / 為替 ======
# 単位：USD / 100万トークン（input, output）
MODEL_PRICES_USD = {
    "gpt-4o-mini":  {"in": 0.60, "out": 2.40},
    "gpt-4o":       {"in": None, "out": None},
    "gpt-4.1-mini": {"in": None, "out": None},
    "gpt-4.1":      {"in": None, "out": None},
    "gpt-3.5-turbo":{"in": None, "out": None},
}
# Whisper（音声→テキスト）参考価格：USD / 分
WHISPER_PRICE_PER_MIN = 0.006

DEFAULT_USDJPY = float(st.secrets.get("USDJPY", 150.0))
if "usd_jpy" not in st.session_state:
    st.session_state["usd_jpy"] = DEFAULT_USDJPY

def estimate_chat_cost_usd(model: str, prompt_tokens: int, completion_tokens: int):
    p = MODEL_PRICES_USD.get(model)
    if not p or p["in"] is None or p["out"] is None:
        return None
    return round((prompt_tokens * p["in"] + completion_tokens * p["out"]) / 1_000_000, 6)

def get_audio_duration_seconds(uploaded_file) -> float | None:
    """mutagen → wave（WAV） → audioread の順で音声長（秒）を推定。ダメなら None。"""
    # 1) mutagen（多形式OK）
    try:
        from mutagen import File as MutagenFile  # pip install mutagen
        f = MutagenFile(io.BytesIO(uploaded_file.getbuffer()))
        if getattr(f, "info", None) and getattr(f.info, "length", None):
            return float(f.info.length)
    except Exception:
        pass
    # 2) wave（WAVのみ）
    try:
        import wave, contextlib
        if uploaded_file.name.lower().endswith(".wav"):
            with contextlib.closing(wave.open(io.BytesIO(uploaded_file.getbuffer()))) as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return frames / float(rate)
    except Exception:
        pass
    # 3) audioread（多形式OK）
    try:
        import audioread  # pip install audioread
        suffix = os.path.splitext(uploaded_file.name)[1] or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        try:
            with audioread.audio_open(tmp_path) as af:
                if getattr(af, "duration", None):
                    return float(af.duration)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    except Exception:
        pass
    return None

# ====== session_state 初期化 ======
if "sidebar_metrics" not in st.session_state:
    st.session_state["sidebar_metrics"] = dict(
        whisper_time=None, whisper_cost_usd=None,
        chat_time=None, chat_cost_usd=None,
        ptok=0, ctok=0
    )

# ====== タブ ======
tab1, tab2 = st.tabs(["① 文字起こし", "② 議事録作成（Markdown）"])

# -----------------------------------------
# ① 文字起こしタブ
# -----------------------------------------
with tab1:
    st.subheader("音声 → テキスト（Whisper API）")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        uploaded = st.file_uploader(
            "音声ファイル（.wav / .mp3 / .m4a / .webm / .ogg 等）",
            type=["wav", "mp3", "m4a", "webm", "ogg"],
            accept_multiple_files=False,
        )
        fmt = st.selectbox("返却形式（response_format）", ["json", "text", "srt", "vtt"], index=0)
        language = st.text_input("言語コード（未指定なら自動判定）", value="ja")
        go = st.button("文字起こしを実行", type="primary", use_container_width=True)

    with col_right:
        st.caption("結果")
        out_area = st.empty()

    if go:
        if not uploaded:
            st.warning("先に音声ファイルをアップロードしてください。")
        else:
            audio_sec = get_audio_duration_seconds(uploaded)
            audio_min = audio_sec / 60.0 if audio_sec else None

            t0 = time.perf_counter()
            with st.spinner("Whisper API に送信中…"):
                file_bytes = uploaded.read()
                files = {"file": (uploaded.name, file_bytes, uploaded.type or "application/octet-stream")}
                data = {"model": "whisper-1", "response_format": fmt}
                if language.strip():
                    data["language"] = language.strip()
                headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
                resp = requests.post(OPENAI_TRANSCRIBE_URL, headers=headers, files=files, data=data, timeout=600)
            elapsed = time.perf_counter() - t0

            if not resp.ok:
                st.error(f"APIエラー: {resp.status_code}\n{resp.text}")
            else:
                if fmt == "json":
                    j = resp.json()
                    text = j.get("text", "")
                    out_area.text_area("テキスト", value=text, height=350)
                    st.session_state["transcribed_text"] = text
                    st.download_button("テキストをダウンロード", data=text, file_name=f"{uploaded.name}.txt",
                                       mime="text/plain", use_container_width=True)
                else:
                    text = resp.text
                    out_area.text_area("テキスト", value=text, height=350)
                    st.session_state["transcribed_text"] = text
                    ext = "txt" if fmt == "text" else fmt
                    st.download_button(f"{fmt.upper()} をダウンロード", data=text, file_name=f"{uploaded.name}.{ext}",
                                       mime="text/plain", use_container_width=True)

                # 本文メトリクス
                c1, c2, c3 = st.columns(3)
                c1.metric("処理時間", f"{elapsed:.2f} 秒")
                if audio_min is not None:
                    usd = audio_min * WHISPER_PRICE_PER_MIN
                    jpy = usd * float(st.session_state["usd_jpy"])
                    c2.metric("概算料金 (USD)", f"${usd:,.6f}")
                    c3.metric("概算料金 (JPY)", f"¥{jpy:,.2f}")
                    st.session_state["sidebar_metrics"]["whisper_cost_usd"] = usd
                else:
                    c2.metric("概算料金 (USD)", "—")
                    c3.metric("概算料金 (JPY)", "—")
                    st.info("音声長の推定に失敗しました。`pip install mutagen audioread` を推奨します。")

                # サイドバー用
                st.session_state["sidebar_metrics"]["whisper_time"] = elapsed

    # —— 繋ぎ：②タブへ引き継ぐ（復活）——
    if st.session_state.get("transcribed_text"):
        st.info("👇 下のボタンで議事録タブへテキストを引き継げます。")
        with st.expander("直近の文字起こし（確認用）", expanded=False):
            st.text_area(
                "文字起こしテキスト（抜粋）",
                value=st.session_state["transcribed_text"][:2000],
                height=160,
            )
        if st.button("② 議事録タブへ引き継ぐ", type="primary", use_container_width=True):
            st.session_state["minutes_source_text"] = st.session_state["transcribed_text"]
            st.success("引き継ぎました。上部タブ『② 議事録作成（Markdown）』を開いてください。")
            # st.rerun()  # 即サイドバー反映させたい場合のみ

# -----------------------------------------
# ② 議事録作成タブ
# -----------------------------------------
with tab2:
    st.subheader("議事録（Markdown）を作成")

    default_prompt = """あなたは20年以上のキャリアがある議事録作成の専門家です。
以下の文字起こしテキストをもとに正式な議事録を作成し、その後、発言をそのままの形でまとめた逐語記録も出力してください。
"""

    colA, colB = st.columns([1, 1], gap="large")

    with colA:
        base_prompt = st.text_area("プロンプト（編集可）", value=default_prompt, height=260)
        extra = st.text_area("追加指示（任意）", value="", height=120)
        model = st.selectbox("モデル", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-3.5-turbo"], index=0)
        temperature = st.slider("温度（0=厳格 / 2=自由）", 0.0, 2.0, value=0.2, step=0.1)
        go_minutes = st.button("議事録を生成", type="primary", use_container_width=True)

    with colB:
        src = st.text_area(
            "文字起こしテキスト",
            value=st.session_state.get("minutes_source_text", ""),
            height=440,
            placeholder="①タブで文字起こししたテキストを引き継ぐか、ここに貼り付けてください。",
        )

    if go_minutes:
        if not src.strip():
            st.warning("文字起こしテキストを入力してください。")
        else:
            # 入力準備
            combined = base_prompt.strip()
            if extra.strip():
                combined += "\n\n【追加指示】\n" + extra.strip() + "\n"
            combined += "\n" + src

            t0 = time.perf_counter()
            with st.spinner("議事録を生成中…"):
                headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
                body = {
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": 1200,
                    "messages": [{"role": "user", "content": combined}],
                }
                r = requests.post(OPENAI_CHAT_URL, headers=headers, json=body, timeout=600)
            elapsed = time.perf_counter() - t0

            if not r.ok:
                st.error(f"APIエラー: {r.status_code}\n{r.text}")
            else:
                data = r.json()
                text = data["choices"][0]["message"]["content"]

                st.markdown("### 📝 生成結果（Markdown 表示）")
                st.markdown(text)
                st.download_button("Markdown をダウンロード", data=text, file_name="minutes.md",
                                   mime="text/markdown", use_container_width=True)
                st.session_state["minutes_markdown"] = text

                # usage と料金
                usage = data.get("usage", {}) or {}
                ptok = int(usage.get("prompt_tokens", 0))
                ctok = int(usage.get("completion_tokens", 0))
                usd = estimate_chat_cost_usd(model, ptok, ctok)
                jpy = usd * float(st.session_state["usd_jpy"]) if usd is not None else None

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("処理時間", f"{elapsed:.2f} 秒")
                c2.metric("入力トークン", f"{ptok:,}")
                c3.metric("出力トークン", f"{ctok:,}")
                c4.metric("概算料金 (USD)", f"${usd:,.6f}" if usd is not None else "—")
                c5.metric("概算料金 (JPY)", f"¥{jpy:,.2f}" if jpy is not None else "—")

                st.session_state["sidebar_metrics"].update(
                    chat_time=elapsed, chat_cost_usd=usd, ptok=ptok, ctok=ctok
                )
                # st.rerun()  # 即サイドバー反映させたい場合のみ

    # コピー用
    if st.session_state.get("minutes_markdown"):
        with st.expander("生成結果（プレーンテキスト）を表示してコピー", expanded=False):
            st.text_area("コピー用テキスト", value=st.session_state["minutes_markdown"], height=200)

# -----------------------------------------
# サイドバー（末尾で描画：更新後の値で表示）
# -----------------------------------------
with st.sidebar:
    st.header("⏱️ 処理 & 概算料金")
    st.subheader("通貨設定")
    st.session_state["usd_jpy"] = st.number_input(
        "USD/JPY（手動設定）", min_value=50.0, max_value=500.0,
        value=float(st.session_state["usd_jpy"]), step=0.5
    )
    fx = float(st.session_state["usd_jpy"])
    m = st.session_state["sidebar_metrics"]

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
