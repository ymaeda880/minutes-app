# ------------------------------------------------------------
# 🎙️ 話者分離・整形（議事録の前処理）— modern専用・リトライなし版
# - .txt をドラッグ＆ドロップ or 貼り付け
# - LLMで話者推定（S1/S2/...）＋発話ごとに改行・整形
# - GPT-5 系列は temperature を変更不可（=1固定）→ UI 無効化＆API未送信
# - 長文（~2万文字）対応：max_completion_tokens は大きめに設定して一発実行
# - 空応答時は resp 全体を st.json で出してデバッグ
# - ✅ 料金計算: lib.costs.estimate_chat_cost_usd（config.MODEL_PRICES_USD 参照）
# - ✅ トークン取得: lib.tokens.extract_tokens_from_response（modern専用）
# - ✅ プロンプト管理: lib/prompts.py のレジストリに統一
# - ✅ 文字起こし.py → 本ページへの自動引き継ぎ＋再読み込みボタン（本ファイルで追加）
# ------------------------------------------------------------
from __future__ import annotations

import time
from typing import Dict, Any

import streamlit as st
from openai import OpenAI

# ==== 共通ユーティリティ ====
from lib.costs import estimate_chat_cost_usd
from lib.tokens import extract_tokens_from_response, debug_usage_snapshot
from lib.prompts import SPEAKER_PREP, get_group, build_prompt
from config.config import DEFAULT_USDJPY
from ui.style import disable_heading_anchors

# ========================== 共通設定 ==========================
st.set_page_config(page_title="③ 話者分離・整形（新）", page_icon="🎙️", layout="wide")
disable_heading_anchors()
st.title("③ 話者分離・整形（新）— 議事録の前処理")

OPENAI_API_KEY = st.secrets.get("openai", {}).get("api_key") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OpenAI API Key が見つかりません。.streamlit/secrets.toml を確認してください。")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ===== 文字起こしタブからの自動引き継ぎ（初回のみ） =====
# 「文字起こし.py」で st.session_state["minutes_source_text"] に入れた内容を、
# 本ページの入力欄（prep_source_text）へ最初の1回だけ自動反映します。
if "prep_source_text_autofilled" not in st.session_state:
    st.session_state["prep_source_text_autofilled"] = False
if (not st.session_state["prep_source_text_autofilled"]) and st.session_state.get("minutes_source_text"):
    st.session_state["prep_source_text"] = st.session_state["minutes_source_text"]
    st.session_state["prep_source_text_autofilled"] = True
    st.session_state["from_transcribe_notice"] = True  # 次のUIで一度だけ通知

# ========================== モデル設定補助 ==========================
def supports_temperature(model_name: str) -> bool:
    """GPT-5系は temperature 変更不可（=1固定）。"""
    return not model_name.startswith("gpt-5")

# ========================== UI ==========================
left, right = st.columns([1, 1], gap="large")

# ---- プロンプト・UI（lib/prompts レジストリ経由） ----
with left:
    st.subheader("プロンプト")

    group = get_group(SPEAKER_PREP)

    # --- セッション初期化 ---
    if "mandatory_prompt" not in st.session_state:
        st.session_state["mandatory_prompt"] = group.mandatory_default
    if "preset_label" not in st.session_state:
        st.session_state["preset_label"] = group.label_for_key(group.default_preset_key)
    if "preset_text" not in st.session_state:
        st.session_state["preset_text"] = group.body_for_label(st.session_state["preset_label"])
    if "extra_text" not in st.session_state:
        st.session_state["extra_text"] = ""

    mandatory = st.text_area(
        "必ず入る部分（常にプロンプトの先頭に含まれます）",
        height=220,
        key="mandatory_prompt",
    )

    def _on_change_preset():
        st.session_state["preset_text"] = group.body_for_label(st.session_state["preset_label"])

    st.selectbox(
        "追記プリセット",
        options=group.preset_labels(),
        index=group.preset_labels().index(st.session_state["preset_label"]),
        key="preset_label",
        help="選んだ内容が上の必須文の下に自動的に連結されます。",
        on_change=_on_change_preset,
    )

    preset_text = st.text_area("（編集可）プリセット本文", height=120, key="preset_text")
    extra = st.text_area("追加指示（任意）", height=88, key="extra_text")

    st.subheader("モデル設定")
    model = st.selectbox(
        "モデル",
        [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-4.1-mini",
            "gpt-4.1",
        ],
        index=1,
    )

    temp_supported = supports_temperature(model)
    temperature = st.slider(
        "温度（0=厳格 / 2=自由）",
        0.0, 2.0, value=1.0, step=0.1,
        disabled=not temp_supported,
        help="GPT-5 系列は temperature=1 固定です",
    )
    if not temp_supported:
        st.caption("ℹ️ GPT-5 系列は temperature を変更できません（=1固定）")

    # 出力上限（modern専用）
    max_completion_tokens = st.slider(
        "最大出力トークン（目安）",
        min_value=1000, max_value=40000, value=12000, step=500,
        help="2万文字級の整形なら 8,000〜12,000 程度を推奨（本版はリトライなし）。",
    )

    st.subheader("通貨換算（任意）")
    usd_jpy = st.number_input("USD/JPY", min_value=50.0, max_value=500.0, value=float(DEFAULT_USDJPY), step=0.5)

    c1, c2 = st.columns(2)
    run_btn = c1.button("話者分離して整形", type="primary", use_container_width=True)
    push_btn = c2.button("➕ この結果を『② 議事録作成』へ渡す", use_container_width=True)

with right:
    st.subheader("入力テキスト")

    # 自動取り込み通知（1回だけ）
    if st.session_state.pop("from_transcribe_notice", False):
        st.success("✅ 『文字起こし』ページからテキストを受け取りました。")

    # 明示的に「文字起こしから再読み込み」するボタン
    reload_col, _ = st.columns([1, 3])
    if reload_col.button("↩ 文字起こしから再読み込み"):
        if st.session_state.get("minutes_source_text"):
            st.session_state["prep_source_text"] = st.session_state["minutes_source_text"]
            st.toast("文字起こしテキストを再読み込みしました", icon="✅")
        else:
            st.toast("読み込める文字起こしテキストがありません", icon="⚠️")

    # .txt ドロップ → prep_source_text へ格納
    up = st.file_uploader("文字起こしテキスト（.txt）をドロップ", type=["txt"], accept_multiple_files=False)
    if up is not None:
        raw = up.read()
        try:
            text_from_file = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text_from_file = raw.decode("cp932")
            except Exception:
                text_from_file = raw.decode(errors="ignore")
        st.session_state["prep_source_text"] = text_from_file

    # テキストエリア（優先: prep_source_text → 次点: minutes_source_text → 空）
    src = st.text_area(
        "文字起こしテキスト（貼り付け可）",
        value=st.session_state.get("prep_source_text", st.session_state.get("minutes_source_text", "")),
        height=420,
        placeholder="①ページの結果を引き継ぐか、ここに貼り付けるか、.txt をドロップしてください。",
        key="prep_source_text_area",
    )

# ========================== 実行（リトライなし一発実行） ==========================
if run_btn:
    # 直近のテキストエリア内容を最優先に使用
    src = st.session_state.get("prep_source_text", st.session_state.get("minutes_source_text", ""))

    if not src.strip():
        st.warning("文字起こしテキストを入力してください。")
    else:
        # プロンプト組み立て
        combined = build_prompt(
            st.session_state["mandatory_prompt"],
            st.session_state["preset_text"],
            st.session_state["extra_text"],
            src,
        )

        def call_once(prompt_text: str, out_tokens: int):
            chat_kwargs: Dict[str, Any] = dict(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                max_completion_tokens=int(out_tokens),
            )
            # GPT-5系は温度固定なので送らない。それ以外で1.0と違う時のみ送る。
            if supports_temperature(model) and abs(temperature - 1.0) > 1e-9:
                chat_kwargs["temperature"] = float(temperature)
            return client.chat.completions.create(**chat_kwargs)

        t0 = time.perf_counter()
        with st.spinner("話者分離・整形を実行中…"):
            resp = call_once(combined, max_completion_tokens)

            text = ""
            finish_reason = None
            if resp and getattr(resp, "choices", None):
                # 出力テキストの取り出し（リトライなし）
                try:
                    text = resp.choices[0].message.content or ""
                except Exception:
                    text = getattr(resp.choices[0], "text", "")
                try:
                    finish_reason = resp.choices[0].finish_reason
                except Exception:
                    finish_reason = None

        elapsed = time.perf_counter() - t0

        if text.strip():
            st.markdown("### ✅ 整形結果")
            st.markdown(text)

            # === ダウンロード & コピー ===
            import json
            base_filename = "speaker_prep_result"
            txt_bytes = text.encode("utf-8")

            dl_col, cp_col = st.columns([1, 1], gap="small")
            with dl_col:
                st.download_button(
                    "📝 テキスト（.txt）をダウンロード",
                    data=txt_bytes,
                    file_name=f"{base_filename}.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with cp_col:
                # クリップボードコピー（Clipboard API）
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
            # === 追加ここまで ===

        else:
            st.warning("⚠️ モデルから空の応答が返されました。レスポンス全体を表示します。")
            try:
                st.json(resp.model_dump())
            except Exception:
                st.write(resp)

        # === トークン算出（modern専用） ===
        input_tok, output_tok, total_tok = extract_tokens_from_response(resp)

        # 料金見積り（modern専用: input/output）
        usd = estimate_chat_cost_usd(model, input_tok, output_tok)
        jpy = (usd * usd_jpy) if usd is not None else None

        import pandas as pd
        metrics_data = {
            "処理時間": [f"{elapsed:.2f} 秒"],
            "入力トークン": [f"{input_tok:,}"],
            "出力トークン": [f"{output_tok:,}"],
            "合計トークン": [f"{total_tok:,}"],
            "概算 (USD/JPY)": [f"${usd:,.6f} / ¥{jpy:,.2f}" if usd is not None else "—"],
        }
        df_metrics = pd.DataFrame(metrics_data)
        st.subheader("トークンと料金の概要")
        st.table(df_metrics)

        # === デバッグ用：modern usage スナップショット ===
        with st.expander("🔍 トークン算出の内訳（modern usage スナップショット）"):
            try:
                st.write(debug_usage_snapshot(getattr(resp, "usage", None)))
            except Exception as e:
                st.write({"error": str(e)})

        st.session_state["prep_last_output"] = text
        st.session_state["minutes_source_text"] = text  # ② 議事録作成タブへ渡す用

# ========================== 引き渡し（ボタン） ==========================
if push_btn:
    out = st.session_state.get("prep_last_output") or st.session_state.get("minutes_source_text", "")
    if not out.strip():
        st.warning("先に『話者分離して整形』を実行してください。")
    else:
        st.session_state["minutes_source_text"] = out
        st.success("整形結果を『② 議事録作成』ページへ渡しました。左のナビから移動してください。")

# ========================== ヘルプ ==========================
with st.expander("⚠️ 長文入力（2万文字前後）の注意点"):
    st.markdown(
        """
- 日本語2万文字は **約1万〜1.5万トークン**です。**gpt-4.1 系 / gpt-5 系**（128kコンテキスト）推奨。
- **max_completion_tokens** は 8000〜12000 程度が安全です（本版はリトライなし。必要に応じて最初から十分大きく）。
- 価格表は `config.MODEL_PRICES_USD`（USD/100万トークン）を運用価格に合わせて調整してください。
"""
    )
