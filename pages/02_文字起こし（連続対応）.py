# pages/02_文字起こし（連続対応）.py
# ============================================================
# 📄 このファイルで最初にやっていること / 変更点（サマリ）
# ------------------------------------------------------------
# ■ 目的：
#   GPT-4o系 Transcribe / Whisper API で音声ファイル（複数可）を文字起こし。
#   各ファイルの結果を表示・保存し、最後に1つに連結（つなぎ目マーカー入り）します。
#
# ■ 主な流れ：
#   1) ページ構成（タイトル・レイアウト）の設定
#   2) 共有メトリクスの初期化（init_metrics_state）
#   3) APIキーの取得（未設定なら停止）
#   4) UI（左：ファイル/パラメータ, 右：結果表示）
#   5) 文字起こしモデルをラジオボタンで選択
#   6) 選択された複数ファイルを上から順番に Transcribe API 呼び出し（リトライ付き）
#   7) 各結果を表示・ダウンロード、合算メトリクス表を表示
#   8) 複数テキストを連結し、”ここがつなぎ目です” マーカーを挿入して提示・DL
#   9) 次タブ引き継ぎ用に session_state["transcribed_text"] を連結済みで保存
# ============================================================

from __future__ import annotations

import io
import re
import time
import json
import requests
from requests.adapters import HTTPAdapter, Retry
import pandas as pd
import streamlit as st

from config.config import (
    get_openai_api_key,
    OPENAI_TRANSCRIBE_URL,
    WHISPER_PRICE_PER_MIN,
    TRANSCRIBE_PRICES_USD_PER_MIN,
    DEFAULT_USDJPY,
)
from lib.audio import get_audio_duration_seconds
from ui.sidebar import init_metrics_state  # render_sidebar は使わない

from lib.explanation import render_transcribe_continuous_expander

# ================= ページ設定 =================
st.set_page_config(page_title="01 文字起こし — Transcribe", layout="wide")
st.title("文字起こし（連続対応）")

# CSS を貼って赤くする（key="dl_combined"）
# st.markdown(
#     """
#     <style>
#     /* 🔴 連結テキスト用のダウンロードボタンだけ赤くする */
#     #combined-download-wrapper button {
#         background-color: #e02424 !important;  /* 赤 */
#         color: white !important;
#         border: none !important;
#         font-weight: bold;
#     }
#     #combined-download-wrapper button:hover {
#         background-color: #c81e1e !important;  /* 濃い赤 */
#     }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )



render_transcribe_continuous_expander()

# ================= 初期化 =================
init_metrics_state()
OPENAI_API_KEY = get_openai_api_key()
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY が .streamlit/secrets.toml に設定されていません。")
    st.stop()

# session_state に為替レートのデフォルトをセット（無ければ）
st.session_state.setdefault("usd_jpy", float(DEFAULT_USDJPY))

# ================= ユーティリティ =================
BRACKET_TAG_PATTERN = re.compile(r"【[^】]*】")

def strip_bracket_tags(text: str) -> str:
    """全角の角括弧【…】で囲まれた短いタグを丸ごと削除。"""
    if not text:
        return text
    return BRACKET_TAG_PATTERN.sub("", text)

PROMPT_OPTIONS = [
    "",  # デフォルト: 空（未指定）
    "出力に話者名や【】などのラベルを入れない。音声に無い単語は書かない。",
    "人名やプロジェクト名は正確に出力してください。専門用語はカタカナで。",
    "句読点を正しく付与し、自然な文章にしてください。",
]

# ================= UI（左／右カラム） =================
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    # ---- モデル選択（ラジオボタン）----
    model = st.radio(
        "モデル",
        options=[
            "whisper-1",
            "gpt-4o-mini-transcribe",
            "gpt-4o-transcribe",
        ],
        index=0,
        help=(
            "OpenAI: 互換/精度重視。"
        ),
)

    uploaded_files = st.file_uploader(
        "音声ファイル（複数可：.wav / .mp3 / .m4a / .webm / .ogg）",
        type=["wav", "mp3", "m4a", "webm", "ogg"],
        accept_multiple_files=True,  # ★ 複数対応
    )

    fmt = st.selectbox("返却形式（response_format）", ["json", "text", "srt", "vtt"], index=0)
    language = st.text_input("言語コード（未指定なら自動判定）", value="ja")

    prompt_hint = st.selectbox(
        "Transcribeプロンプト（省略可）",
        options=PROMPT_OPTIONS,
        index=0,
        help="誤変換しやすい固有名詞や抑止指示などを短く入れると精度が安定します。空でもOK。",
    )

    do_strip_brackets = st.checkbox("書き起こし後に【…】を除去する", value=True)

    st.subheader("通貨換算（任意）")
    usd_jpy = st.number_input(
        "USD/JPY",
        min_value=50.0,
        max_value=500.0,
        value=float(st.session_state.get("usd_jpy", DEFAULT_USDJPY)),
        step=0.5,
    )
    st.session_state["usd_jpy"] = float(usd_jpy)

    go = st.button("文字起こしを実行（選択された順に処理）", type="primary", use_container_width=True)

with col_right:
    st.caption("結果")
    out_area = st.container()

# ================= 実行ハンドラ =================
if go:
    if not uploaded_files:
        st.warning("先に音声ファイルをアップロードしてください。")
        st.stop()

    # 並び順はアップローダに表示された順（Streamlit はその順を保持）
    # 進捗バー
    progress = st.progress(0, text="準備中…")

    # セッションとリトライ設定（POST のみ）
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    sess = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset({"POST"}),
    )
    sess.mount("https://", HTTPAdapter(max_retries=retries))

    per_file_results = []  # [{name, text, sec, min, usd, jpy, elapsed, req_id}]
    combined_parts = []    # 連結用（テキストのみ）
    total_elapsed = 0.0

    for idx, uploaded in enumerate(uploaded_files, start=1):
        progress.progress((idx - 1) / len(uploaded_files), text=f"{idx}/{len(uploaded_files)} 処理中: {uploaded.name}")

        file_bytes = uploaded.read()
        if not file_bytes:
            st.error(f"{uploaded.name}: アップロードファイルが空です。スキップします。")
            continue

        # 長さ推定
        try:
            audio_sec = get_audio_duration_seconds(io.BytesIO(file_bytes))
            audio_min = audio_sec / 60.0 if audio_sec else None
        except Exception:
            audio_sec = None
            audio_min = None
            st.info(f"{uploaded.name}: 音声長の推定に失敗しました。`pip install mutagen audioread` を推奨。")

        mime = uploaded.type or "application/octet-stream"
        files = {"file": (uploaded.name, file_bytes, mime)}

        # 空文字は送らない（prompt/language を条件付きで付与）
        data: dict = {
            "model": model,
            "response_format": fmt,
        }
        if prompt_hint and prompt_hint.strip():
            data["prompt"] = prompt_hint.strip()
        if language and language.strip():
            data["language"] = language.strip()

        t0 = time.perf_counter()
        with st.spinner(f"Transcribe API に送信中…（{uploaded.name}）"):
            resp = sess.post(
                OPENAI_TRANSCRIBE_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=600,
            )
        elapsed = time.perf_counter() - t0
        total_elapsed += elapsed

        req_id = resp.headers.get("x-request-id")
        if not resp.ok:
            st.error(f"{uploaded.name}: APIエラー: {resp.status_code}\n{resp.text}\nrequest-id: {req_id}")
            continue

        # フォーマット別の取り出し
        if fmt == "json":
            try:
                text = resp.json().get("text", "")
            except Exception:
                text = resp.text
        else:
            text = resp.text

        if do_strip_brackets and text:
            text = strip_bracket_tags(text)

        # コスト見積
        usd = jpy = None
        if audio_min is not None:
            price_per_min = TRANSCRIBE_PRICES_USD_PER_MIN.get(model, WHISPER_PRICE_PER_MIN)
            usd = float(audio_min) * float(price_per_min)
            jpy = usd * float(st.session_state["usd_jpy"])

        # 個別表示（右ペイン）
        with out_area:
            st.markdown(f"#### 📁 {idx}. {uploaded.name}")
            st.text_area("テキスト（個別）", value=text, height=220, key=f"ta_{idx}")

            # 個別DL
            base_filename = (uploaded.name.rsplit(".", 1)[0] if uploaded else f"transcript_{idx}").replace(" ", "_")
            # st.download_button(
            #     "📝 このテキスト（.txt）をダウンロード",
            #     data=(text or "").encode("utf-8"),
            #     file_name=f"{base_filename}.txt",
            #     mime="text/plain",
            #     use_container_width=True,
            #     key=f"dl_{idx}",
            # )

            # 個別メトリクス表
            metrics_data = {
                "処理時間": [f"{elapsed:.2f} 秒"],
                "音声長": [f"{audio_sec:.1f} 秒 / {audio_min:.2f} 分" if audio_sec else "—"],
                "概算 (USD/JPY)": [f"${usd:,.6f} / ¥{jpy:,.2f}" if usd is not None else "—"],
                "request-id": [req_id or "—"],
                "モデル": [model],
            }
            st.table(pd.DataFrame(metrics_data))

        # 連結用に保存
        per_file_results.append(
            dict(
                name=uploaded.name,
                text=text,
                sec=audio_sec,
                min=audio_min,
                usd=usd,
                jpy=jpy,
                elapsed=elapsed,
                req_id=req_id,
            )
        )

        combined_parts.append(text or "")

        # つなぎ目マーカー（最後の要素以外の後ろに入れる）
        if idx < len(uploaded_files):
            combined_parts.append(
                f"\n\n----- ここがつなぎ目です（{uploaded.name} と次のファイルの間）-----\n\n"
            )

    progress.progress(1.0, text="完了")

    # ====== まとめ（連結テキスト & 合算メトリクス）======
    combined_text = "".join(combined_parts)

    # 右ペインに連結結果をまとめて表示
    with out_area:
        st.subheader("🔗 連結テキスト（全ファイル）")
        st.text_area("テキスト（連結済み）", value=combined_text, height=350)

        # 連結テキストのDL & クリップボード
        comb_fname = "transcripts_combined"
  
        # 🔴 ラッパー div を追加
        #st.markdown('<div id="combined-download-wrapper">', unsafe_allow_html=True)

        st.download_button(
            "🧩 連結テキスト（.txt）をダウンロード",
            data=(combined_text or "").encode("utf-8"),
            file_name=f"{comb_fname}.txt",
            mime="text/plain",
            use_container_width=True,
            key="dl_combined",
            help="combined download button",
        )

        st.markdown('</div>', unsafe_allow_html=True)

        safe_json = json.dumps(combined_text or "", ensure_ascii=False)
        st.components.v1.html(f"""
        <div style="display:flex;align-items:center;gap:.5rem">
          <button id="copyBtnCombined" style="width:100%;padding:.6rem 1rem;border-radius:.5rem;border:1px solid #e0e0e0;cursor:pointer">
            📋 連結テキストをコピー
          </button>
          <span id="copyMsgCombined" style="font-size:.9rem;color:#888"></span>
        </div>
        <script>
          const content = {safe_json};
          const btn = document.getElementById("copyBtnCombined");
          const msg = document.getElementById("copyMsgCombined");
          btn.addEventListener("click", async () => {{
            try {{
              await navigator.clipboard.writeText(content);
              msg.textContent = "コピーしました";
              setTimeout(() => msg.textContent = "", 1600);
            }} catch (e) {{
              msg.textContent = "コピーに失敗";
              setTimeout(() => msg.textContent = "", 1600);
            }}
          }});
        </script>
        """, height=60)

        # 合算メトリクス表
        total_sec = sum([r["sec"] for r in per_file_results if r["sec"] is not None]) if per_file_results else None
        total_min = sum([r["min"] for r in per_file_results if r["min"] is not None]) if per_file_results else None
        total_usd = sum([r["usd"] for r in per_file_results if r["usd"] is not None]) if per_file_results else None
        total_jpy = sum([r["jpy"] for r in per_file_results if r["jpy"] is not None]) if per_file_results else None

        st.subheader("📊 料金の概要（合算）")
        df_total = pd.DataFrame({
            "ファイル数": [len(per_file_results)],
            "合計処理時間": [f"{total_elapsed:.2f} 秒"],
            "合計音声長": [f"{total_sec:.1f} 秒 / {total_min:.2f} 分" if total_sec else "—"],
            "合計概算 (USD/JPY)": [f"${total_usd:,.6f} / ¥{total_jpy:,.2f}" if total_usd is not None else "—"],
            "モデル": [model],
        })
        st.table(df_total)

        # 参考：ファイル別サマリー表
        if per_file_results:
            st.caption("ファイル別サマリー")
            df_each = pd.DataFrame([{
                "ファイル": r["name"],
                "処理時間(秒)": round(r["elapsed"], 2),
                "音声長(分)": (round(r["min"], 2) if r["min"] is not None else None),
                "概算USD": (round(r["usd"], 6) if r["usd"] is not None else None),
                "概算JPY": (round(r["jpy"], 2) if r["jpy"] is not None else None),
                "request-id": r["req_id"] or "—",
            } for r in per_file_results])
            st.dataframe(df_each, use_container_width=True)

    # 議事録タブへの引き継ぎ（連結版を保存）
    st.session_state["transcribed_texts"] = [r["text"] for r in per_file_results]
    st.session_state["transcribed_text"] = combined_text

# ================= 次タブへの引き継ぎ =================
# if st.session_state.get("transcribed_text"):
#     st.info("👇 下のボタンで話者分離タブへテキストを引き継げます。")
#     with st.expander("直近の文字起こし（確認用：連結テキストの先頭抜粋）", expanded=False):
#         st.text_area(
#             "文字起こしテキスト（連結・抜粋）",
#             value=st.session_state["transcribed_text"][:2000],
#             height=160,
#         )
#     if st.button("② 話者分離タブへ引き継ぐ", type="primary", use_container_width=True):
#         st.session_state["minutes_source_text"] = st.session_state["transcribed_text"]
#         st.success("引き継ぎました。上部タブ『② 話者分離（Markdown）』を開いてください。")
