import io
import time
import requests
import streamlit as st

# ====== 設定 ======
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
OPENAI_TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY が .streamlit/secrets.toml に設定されていません。")
    st.stop()

# ====== UI ヘッダ ======
st.set_page_config(page_title="Whisper + Minutes (All Python)", layout="wide")
st.title("🎙️ Whisper Transcribe + 📝 Minutes Maker (All Python)")

tab1, tab2 = st.tabs(["① 文字起こし", "② 議事録作成（Markdown）"])

# 共通: 小ユーティリティ
def copy_button_js(id_of_textarea: str):
    """Streamlitの要素をコピーする簡易JSを返す（見た目の演出用）。"""
    # ここではシンプルにダウンロードボタンを使うので未使用。必要ならst.componentsで実装可。
    pass

# =========================================
# ① 文字起こしタブ
# =========================================
with tab1:
    st.subheader("音声 → テキスト（Whisper API）")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        uploaded = st.file_uploader(
            "音声ファイルをドラッグ＆ドロップ（.wav / .mp3 / .m4a / .webm など）",
            type=["wav", "mp3", "m4a", "webm", "ogg"],
            accept_multiple_files=False,
        )
        fmt = st.selectbox("返却形式（response_format）", ["json", "text", "srt", "vtt"], index=0)
        language = st.text_input("言語コード（未指定なら自動判定）", value="ja")
        go = st.button("文字起こしを実行", type="primary", use_container_width=True)

    with col_right:
        st.caption("結果")
        out_area = st.empty()
        dl_col = st.container()

    if go:
        if not uploaded:
            st.warning("先に音声ファイルをアップロードしてください。")
        else:
            with st.spinner("Whisper API に送信中…"):
                file_bytes = uploaded.read()
                files = {
                    "file": (uploaded.name, file_bytes, uploaded.type or "application/octet-stream"),
                }
                data = {"model": "whisper-1", "response_format": fmt}
                if language.strip():
                    data["language"] = language.strip()

                headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
                resp = requests.post(
                    OPENAI_TRANSCRIBE_URL, headers=headers, files=files, data=data, timeout=600
                )

            if not resp.ok:
                st.error(f"APIエラー: {resp.status_code}\n{resp.text}")
            else:
                if fmt == "json":
                    j = resp.json()
                    text = j.get("text", "")
                    out_area.text_area("テキスト（JSONから抽出）", value=text, height=350)
                    st.session_state["transcribed_text"] = text
                    # ダウンロード（.txt）
                    st.download_button(
                        "テキストをダウンロード",
                        data=text,
                        file_name=f"{uploaded.name}.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                else:
                    # text / srt / vtt は生テキスト
                    text = resp.text
                    label = {"text": "テキスト", "srt": "SRT字幕", "vtt": "VTT字幕"}[fmt]
                    out_area.text_area(label, value=text, height=350)
                    st.session_state["transcribed_text"] = text
                    # そのままダウンロード
                    ext = "txt" if fmt == "text" else fmt
                    st.download_button(
                        f"{label}をダウンロード",
                        data=text,
                        file_name=f"{uploaded.name}.{ext}",
                        mime="text/plain",
                        use_container_width=True,
                    )

    # 直近の結果をワンタップでエディタに展開（次タブで使う）
    if "transcribed_text" in st.session_state and st.session_state["transcribed_text"]:
        st.info("👇 下のボタンで議事録タブへテキストを引き継げます。")
        if st.button("議事録タブへ引き継ぐ", use_container_width=True):
            st.session_state["minutes_source_text"] = st.session_state["transcribed_text"]
            st.success("引き継ぎました。②タブを開いてください。")

# =========================================
# ② 議事録作成タブ
# =========================================
with tab2:
    st.subheader("議事録（Markdown）を作成")

    default_prompt = """あなたは20年以上のキャリアがある議事録作成の専門家です。
（１）以下に与えられた文字起こしテキストをもとに、正式な議事録を作成してください。
（２）議事録を作成した後に，会話部分も含めて逐語的な記録（発言をそのまま書き起こした形）にまとめ直してください．
つまり，逐語記録も作成してください．

【議事録の要件】
- 会議の「目的」「出席者」「議題」「議論の内容」「結論」「今後のアクション」を整理してください。
- 発言の細かい口語表現は削除し、要点だけを簡潔にまとめてください。
- 出力は日本語で、見出し付きの箇条書き形式にしてください。
- Markdown形式で，見栄えよく，専門家が書いた議事録に見えるように出力してください

【逐語記録の要件】
- 発言者ごとの会話を発言者がわかるように見やすくまとめてください．
- 文字起こしされたテキストをそのまま使用してください．
- Markdown形式で，見栄えよく，専門家が書いた逐語記録に見えるように出力してください
"""

    colA, colB = st.columns([1, 1])

    with colA:
        base_prompt = st.text_area("プロンプト（編集可）", value=default_prompt, height=260)
        extra_prompt = st.text_area("追加指示（任意）", value="", height=120)
        model = st.selectbox(
            "モデル",
            ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-3.5-turbo"],
            index=0,
        )
        temperature = st.slider("温度（0=厳格 / 2=自由）", min_value=0.0, max_value=2.0, value=0.2, step=0.1)
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
            with st.spinner("議事録を生成中…"):
                combined = base_prompt.strip()
                if extra_prompt.strip():
                    combined += "\n\n【追加指示】\n" + extra_prompt.strip() + "\n"
                combined += "\n" + src

                headers = {
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                }
                body = {
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": 1200,
                    "messages": [
                        {"role": "user", "content": combined}
                    ],
                }
                r = requests.post(OPENAI_CHAT_URL, headers=headers, json=body, timeout=600)
            if not r.ok:
                st.error(f"APIエラー: {r.status_code}\n{r.text}")
            else:
                data = r.json()
                text = data["choices"][0]["message"]["content"]
                st.markdown("### 📝 生成結果（Markdown 表示）")
                st.markdown(text)
                st.download_button(
                    "Markdown をダウンロード",
                    data=text,
                    file_name="minutes.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
                st.session_state["minutes_markdown"] = text

    # 生成済みをコピー用に表示（任意）
    if st.session_state.get("minutes_markdown"):
        with st.expander("生成結果（プレーンテキスト）を表示してコピー", expanded=False):
            st.text_area(
                "コピー用テキスト",
                value=st.session_state["minutes_markdown"],
                height=200,
            )
