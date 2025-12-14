# pages/02_文字起こし（Gemini）（連続対応）.py
# ============================================================
# ■ 目的：
#   OpenAI Transcribe / Whisper API で音声ファイル（複数可）を文字起こし。
#   さらに Gemini（Google AI Studio API Key）でも文字起こし可能に拡張。
#
# ■ Gemini の扱い：
#   - radio options に gemini を表示
#   - GEMINI_API_KEY が無いときは「選べない」ように、選択されたら直前の有効モデルへ戻す
#   - Gemini でも費用（概算）を表示（トークン推定 × USD/100万トークン単価）
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
    # --- keys / endpoints ---
    get_openai_api_key,
    get_gemini_api_key,
    has_gemini_api_key,
    OPENAI_TRANSCRIBE_URL,
    # --- prices ---
    WHISPER_PRICE_PER_MIN,
    TRANSCRIBE_PRICES_USD_PER_MIN,
    DEFAULT_USDJPY,
    # --- gemini cost helpers ---
    estimate_tokens_from_text,
    estimate_gemini_cost_usd,
)
from lib.audio import get_audio_duration_seconds
from ui.sidebar import init_metrics_state  # render_sidebar は使わない
from lib.explanation import render_transcribe_continuous_expander

# ================= ページ設定 =================
st.set_page_config(page_title="01 文字起こし — Transcribe", layout="wide")
st.title("文字起こし（Gemini対応）（連続対応）")

render_transcribe_continuous_expander()

# ================= 初期化 =================
init_metrics_state()

# OpenAI Key（既存挙動を維持：無いと停止）
OPENAI_API_KEY = get_openai_api_key()
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY が .streamlit/secrets.toml に設定されていません。")
    st.stop()

# Gemini Key（任意：無い場合は Gemini を選べないようにする）
GEMINI_ENABLED = has_gemini_api_key()
GEMINI_API_KEY = get_gemini_api_key() if GEMINI_ENABLED else ""

# session_state に為替レートのデフォルトをセット（無ければ）
st.session_state.setdefault("usd_jpy", float(DEFAULT_USDJPY))

# 「モデル選択を戻す」ための状態
st.session_state.setdefault("model_last_valid", "whisper-1")
st.session_state.setdefault("model_picker", "whisper-1")
st.session_state.setdefault("gemini_disabled_notice", False)

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

MODEL_OPTIONS = [
    "whisper-1",
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe",
    "gemini-2.0-flash",
    # 必要なら追加：
    # "gemini-2.0-pro",
]


def model_label(x: str) -> str:
    if x.startswith("gemini") and not GEMINI_ENABLED:
        return f"{x}（GEMINI_API_KEY 未設定）"
    return x


def on_change_model_picker():
    picked = st.session_state.get("model_picker", "whisper-1")
    if picked.startswith("gemini") and not GEMINI_ENABLED:
        # Gemini は選べない：直前の有効モデルに戻す
        st.session_state["gemini_disabled_notice"] = True
        st.session_state["model_picker"] = st.session_state.get("model_last_valid", "whisper-1")
    else:
        st.session_state["model_last_valid"] = picked
        st.session_state["gemini_disabled_notice"] = False


# ================= UI（左／右カラム） =================
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    # ---- モデル選択（ラジオボタン）----
    st.radio(
        "モデル",
        options=MODEL_OPTIONS,
        # index=MODEL_OPTIONS.index(st.session_state.get("model_picker", "whisper-1"))
        # if st.session_state.get("model_picker", "whisper-1") in MODEL_OPTIONS
        # else 0,
        key="model_picker",
        format_func=model_label,
        on_change=on_change_model_picker,
        help="OpenAI: 互換/精度重視。Gemini: 高速・長音声・要約向き（要 GEMINI_API_KEY）。",
    )

    if st.session_state.get("gemini_disabled_notice", False) and not GEMINI_ENABLED:
        st.warning(
            "GEMINI_API_KEY が未設定のため、Gemini は選択できません。"
            "（.streamlit/secrets.toml に GEMINI_API_KEY を設定してください）"
        )

    model = st.session_state["model_picker"]

    uploaded_files = st.file_uploader(
        "音声ファイル（複数可：.wav / .mp3 / .m4a / .webm / .ogg）",
        type=["wav", "mp3", "m4a", "webm", "ogg"],
        accept_multiple_files=True,
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

    # Gemini が選ばれているのに key が無い、という状態は UI で防いでいるが念のため
    if model.startswith("gemini") and not GEMINI_ENABLED:
        st.error("GEMINI_API_KEY が未設定のため、Gemini は利用できません。")
        st.stop()

    # 進捗バー
    progress = st.progress(0, text="準備中…")

    # OpenAI 用セッションとリトライ設定（POST のみ）
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    sess = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset({"POST"}),
    )
    sess.mount("https://", HTTPAdapter(max_retries=retries))

    per_file_results = []  # [{name, text, sec, min, usd, jpy, elapsed, req_id, in_tok, out_tok}]
    combined_parts = []  # 連結用（テキストのみ）
    total_elapsed = 0.0

    USE_GEMINI = model.startswith("gemini")

    # Gemini クライアント（必要なときだけ import）
    if USE_GEMINI:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(model)

    for idx, uploaded in enumerate(uploaded_files, start=1):
        progress.progress(
            (idx - 1) / len(uploaded_files),
            text=f"{idx}/{len(uploaded_files)} 処理中: {uploaded.name}",
        )

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
        t0 = time.perf_counter()

        # -------------------
        # Gemini 分岐
        # -------------------
        if USE_GEMINI:
            # fmt/language/prompt_hint を「指示」として渡す（API パラメータではない）
            # instr_parts = ["この音声を日本語で正確に文字起こししてください。"]
            instr_parts = [
                "この音声を日本語で正確に文字起こししてください。",
                "日本語は分かち書きにしないでください（単語の間に不要な半角スペースを入れない）。",
                "句読点（、。）を適切に補い、自然な文章として出力してください。",
]
            if language and language.strip():
                instr_parts.append(f"言語コードは {language.strip()} を優先（不明なら自動判定）。")
            if prompt_hint and prompt_hint.strip():
                instr_parts.append(prompt_hint.strip())
            instruction = " ".join(instr_parts)

            with st.spinner(f"Gemini 文字起こし中…（{uploaded.name}）"):
                response = gemini_model.generate_content(
                    [
                        instruction,
                        {"mime_type": mime, "data": file_bytes},
                    ]
                )

            elapsed = time.perf_counter() - t0
            total_elapsed += elapsed

            text = getattr(response, "text", "") or ""
            req_id = "gemini"

        # -------------------
        # OpenAI 分岐（既存）
        # -------------------
        else:
            files = {"file": (uploaded.name, file_bytes, mime)}

            data: dict = {
                "model": model,
                "response_format": fmt,
            }
            if prompt_hint and prompt_hint.strip():
                data["prompt"] = prompt_hint.strip()
            if language and language.strip():
                data["language"] = language.strip()

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

            if fmt == "json":
                try:
                    text = resp.json().get("text", "")
                except Exception:
                    text = resp.text
            else:
                text = resp.text

        # 後処理
        if do_strip_brackets and text:
            text = strip_bracket_tags(text)

        # -------------------
        # コスト見積
        # -------------------
        usd = jpy = None
        in_tok = out_tok = None

        if USE_GEMINI:
            # Gemini：トークン推定で概算
            out_tok = estimate_tokens_from_text(text)
            # 音声入力の正確な token は直接取れないため近似（必要なら係数を調整）
            in_tok = out_tok

            usd_est = estimate_gemini_cost_usd(
                model=model,
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
            if usd_est is not None:
                usd = float(usd_est)
                jpy = usd * float(st.session_state["usd_jpy"])
        else:
            # OpenAI：分単価で概算
            if audio_min is not None:
                price_per_min = TRANSCRIBE_PRICES_USD_PER_MIN.get(model, WHISPER_PRICE_PER_MIN)
                usd = float(audio_min) * float(price_per_min)
                jpy = usd * float(st.session_state["usd_jpy"])

        # 個別表示（右ペイン）
        with out_area:
            st.markdown(f"#### 📁 {idx}. {uploaded.name}")
            st.text_area("テキスト（個別）", value=text, height=220, key=f"ta_{idx}")

            cost_str = "—"
            if usd is not None and jpy is not None:
                cost_str = f"${usd:,.6f} / ¥{jpy:,.2f}"
            elif USE_GEMINI:
                cost_str = "—（Gemini：モデル単価未設定 or 推定不能）"

            metrics_data = {
                "処理時間": [f"{elapsed:.2f} 秒"],
                "音声長": [f"{audio_sec:.1f} 秒 / {audio_min:.2f} 分" if audio_sec else "—"],
                "概算 (USD/JPY)": [cost_str],
                "推定tokens(in/out)": [f"{in_tok}/{out_tok}" if USE_GEMINI and in_tok is not None else "—"],
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
                in_tok=in_tok,
                out_tok=out_tok,
            )
        )

        combined_parts.append(text or "")

        # つなぎ目マーカー
        if idx < len(uploaded_files):
            combined_parts.append(
                f"\n\n----- ここがつなぎ目です（{uploaded.name} と次のファイルの間）-----\n\n"
            )

    progress.progress(1.0, text="完了")

    # ====== まとめ（連結テキスト & 合算メトリクス）======
    combined_text = "".join(combined_parts)

    with out_area:
        st.subheader("🔗 連結テキスト（全ファイル）")
        st.text_area("テキスト（連結済み）", value=combined_text, height=350)

        comb_fname = "transcripts_combined"
        st.download_button(
            "🧩 連結テキスト（.txt）をダウンロード",
            data=(combined_text or "").encode("utf-8"),
            file_name=f"{comb_fname}.txt",
            mime="text/plain",
            use_container_width=True,
            key="dl_combined",
            help="combined download button",
        )

        safe_json = json.dumps(combined_text or "", ensure_ascii=False)
        st.components.v1.html(
            f"""
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
        """,
            height=60,
        )

        total_sec = sum([r["sec"] for r in per_file_results if r["sec"] is not None]) if per_file_results else None
        total_min = sum([r["min"] for r in per_file_results if r["min"] is not None]) if per_file_results else None
        total_usd = sum([r["usd"] for r in per_file_results if r["usd"] is not None]) if per_file_results else None
        total_jpy = sum([r["jpy"] for r in per_file_results if r["jpy"] is not None]) if per_file_results else None

        st.subheader("📊 料金の概要（合算）")
        df_total = pd.DataFrame(
            {
                "ファイル数": [len(per_file_results)],
                "合計処理時間": [f"{total_elapsed:.2f} 秒"],
                "合計音声長": [f"{total_sec:.1f} 秒 / {total_min:.2f} 分" if total_sec else "—"],
                "合計概算 (USD/JPY)": [
                    f"${total_usd:,.6f} / ¥{total_jpy:,.2f}" if total_usd is not None else "—"
                ],
                "モデル": [model],
                "備考": ["Gemini は tokens 推定による概算" if USE_GEMINI else "OpenAI は分単価による概算"],
            }
        )
        st.table(df_total)

        if per_file_results:
            st.caption("ファイル別サマリー")
            df_each = pd.DataFrame(
                [
                    {
                        "ファイル": r["name"],
                        "処理時間(秒)": round(r["elapsed"], 2),
                        "音声長(分)": (round(r["min"], 2) if r["min"] is not None else None),
                        "推定tokens(in/out)": (f"{r['in_tok']}/{r['out_tok']}" if r["in_tok"] is not None else None),
                        "概算USD": (round(r["usd"], 6) if r["usd"] is not None else None),
                        "概算JPY": (round(r["jpy"], 2) if r["jpy"] is not None else None),
                        "request-id": r["req_id"] or "—",
                    }
                    for r in per_file_results
                ]
            )
            st.dataframe(df_each, use_container_width=True)

    # 次タブ引き継ぎ
    st.session_state["transcribed_texts"] = [r["text"] for r in per_file_results]
    st.session_state["transcribed_text"] = combined_text
