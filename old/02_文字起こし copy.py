# pages/01_文字起こし.py
# ============================================================
# 📄 このファイルで最初にやっていること / 変更点（サマリ）
# ------------------------------------------------------------
# ■ 目的：
#   GPT-4o系 Transcribe / Whisper API で音声ファイルを文字起こしし、結果と概算料金を表示します。
#
# ■ 主な流れ：
#   1) ページ構成（タイトル・レイアウト）の設定
#   2) 共有メトリクスの初期化（init_metrics_state）
#   3) APIキーの取得（未設定なら停止）
#   4) 正規表現やプロンプト候補などのユーティリティ準備
#   5) UI（左：ファイル/パラメータ, 右：結果表示）
#   6) 文字起こしモデルをラジオボタンで選択
#   7) Transcribe API 呼び出し（リトライ付き）
#   8) 結果表示＋料金サマリー表
#   9) 🔽 追加：整形結果テキストの「.txt ダウンロード」「ワンクリックコピー」機能
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
from ui.sidebarOld import init_metrics_state  # render_sidebar は使わない

# ================= ページ設定 =================
st.set_page_config(page_title="01 文字起こし — Transcribe", layout="wide")
st.title("① 文字起こし（GPT-4o Transcribe / Whisper）")

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
    # ---- モデル選択（ラジオボタン） ----
    model = st.radio(
        "モデル",
        options=["gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"],
        index=0,
        help="コスト/速度重視なら mini、精度重視なら 4o-transcribe、互換重視なら whisper-1。",
    )

    uploaded = st.file_uploader(
        "音声ファイル（.wav / .mp3 / .m4a / .webm / .ogg 等）",
        type=["wav", "mp3", "m4a", "webm", "ogg"],
        accept_multiple_files=False,
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

    go = st.button("文字起こしを実行", type="primary", use_container_width=True)

with col_right:
    st.caption("結果")
    out_area = st.empty()

# ================= 実行ハンドラ =================
if go:
    if not uploaded:
        st.warning("先に音声ファイルをアップロードしてください。")
        st.stop()

    file_bytes = uploaded.read()
    if not file_bytes:
        st.error("アップロードファイルが空です。もう一度アップロードしてください。")
        st.stop()

    try:
        # get_audio_duration_seconds は BytesIO/一時ファイル経由で複数ライブラリで推定
        audio_sec = get_audio_duration_seconds(io.BytesIO(file_bytes))
        audio_min = audio_sec / 60.0 if audio_sec else None
    except Exception:
        audio_sec = None
        audio_min = None
        st.info("音声長の推定に失敗しました。`pip install mutagen audioread` を推奨。")

    mime = uploaded.type or "application/octet-stream"
    files = {"file": (uploaded.name, file_bytes, mime)}
    data = {
        "model": model,  # ← ラジオ選択値をそのまま利用
        "response_format": fmt,
        "prompt": (prompt_hint or "").strip(),
    }
    if language.strip():
        data["language"] = language.strip()

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

    sess = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["POST"],
    )
    sess.mount("https://", HTTPAdapter(max_retries=retries))

    t0 = time.perf_counter()
    with st.spinner("Transcribe API に送信中…"):
        resp = sess.post(
            OPENAI_TRANSCRIBE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=600,
        )
    elapsed = time.perf_counter() - t0

    req_id = resp.headers.get("x-request-id")
    if not resp.ok:
        st.error(f"APIエラー: {resp.status_code}\n{resp.text}\nrequest-id: {req_id}")
        st.stop()

    if fmt == "json":
        try:
            text = resp.json().get("text", "")
        except Exception:
            text = resp.text
    else:
        text = resp.text

    if do_strip_brackets and text:
        text = strip_bracket_tags(text)

    # ====== 結果テキスト表示 ======
    out_area.text_area("テキスト", value=text, height=350)
    st.session_state["transcribed_text"] = text

    # ====== 追加：テキストのダウンロード & クリップボードコピー ======
    base_filename = (uploaded.name.rsplit(".", 1)[0] if uploaded else "transcript").replace(" ", "_")
    txt_bytes = (text or "").encode("utf-8")

    cols_dl, cols_cp = st.columns([1, 1], gap="small")
    with cols_dl:
        st.download_button(
            "📝 テキスト（.txt）をダウンロード",
            data=txt_bytes,
            file_name=f"{base_filename}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with cols_cp:
        # クリップボードコピー（components を使った簡易ボタン）
        safe_json = json.dumps(text or "", ensure_ascii=False)
        st.components.v1.html(f"""
        <div style="display:flex;align-items:center;gap:.5rem">
          <button id="copyBtn" style="width:100%;padding:.6rem 1rem;border-radius:.5rem;border:1px solid #e0e0e0;cursor:pointer">
            📋 テキストをコピー
          </button>
          <span id="copyMsg" style="font-size:.9rem;color:#888"></span>
        </div>
        <script>
          const content = {safe_json};
          const btn = document.getElementById("copyBtn");
          const msg = document.getElementById("copyMsg");
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

    # ====== 料金サマリー表 ======
    # モデル別の分課金に対応。設定が無ければ WHISPER_PRICE_PER_MIN をフォールバック。
    usd = jpy = None
    if audio_min is not None:
        price_per_min = TRANSCRIBE_PRICES_USD_PER_MIN.get(model, WHISPER_PRICE_PER_MIN)
        usd = float(audio_min) * float(price_per_min)
        jpy = usd * float(usd_jpy)

    metrics_data = {
        "処理時間": [f"{elapsed:.2f} 秒"],
        "音声長": [f"{audio_sec:.1f} 秒 / {audio_min:.2f} 分" if audio_sec else "—"],
        "概算 (USD/JPY)": [f"${usd:,.6f} / ¥{jpy:,.2f}" if usd is not None else "—"],
        "request-id": [req_id or "—"],
        "モデル": [model],
    }
    df_metrics = pd.DataFrame(metrics_data)
    st.subheader("料金の概要")
    st.table(df_metrics)

# ================= 次タブへの引き継ぎ =================
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
