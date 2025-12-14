# 07_議事録作成（新）.py
# ------------------------------------------------------------
# 📝 議事録作成（整形済みテキスト → 議事録）— modern専用・リトライなし版
# ------------------------------------------------------------
from __future__ import annotations

import time
from typing import Dict, Any
from io import BytesIO
from pathlib import Path
from datetime import datetime
import re

import streamlit as st
import pandas as pd
from openai import OpenAI

# ★ ここから追加：ページ専用のセッションキー
PAGE_NAME = Path(__file__).stem
SESSION_KEY_SOURCE = f"{PAGE_NAME}_source_text"
# ★ ここまで追加

# 1 回の呼び出しで許可する最大出力トークン数（固定）
MAX_COMPLETION_TOKENS = 100000

# ==== .docx 読み取り／書き出し（python-docx） ====
try:
    from docx import Document
    from lib.docx_minutes_export import build_minutes_docx
    HAS_DOCX = True
except Exception:
    HAS_DOCX = False
    build_minutes_docx = None  # type: ignore

# ==== 共通ユーティリティ ====
from lib.prompts import (
    MINUTES_MAKER,
    MINUTES_MANDATORY_MODES,   # ← キーは「逐語録」「簡易議事録」など
    MINUTES_STYLE,             # ← 見た目スタイル用グループ（UI 用）
    MINUTES_GLOBAL_MANDATORY,  # ← Minutes Maker 全体共通 mandatory（表のルールなど）
    get_group,
    build_prompt,
)
from lib.tokens import extract_tokens_from_response, debug_usage_snapshot  # modern専用
from lib.costs import estimate_chat_cost_usd
from config.config import DEFAULT_USDJPY

from lib.explanation import (
    render_minutes_maker_expander,
    render_minutes_prompt_spec_expander,
)

# ========================== 共通設定 ==========================
st.title("議事録作成 — 逐語録から正式議事録へ")
render_minutes_maker_expander()          # 上：ページの使い方
render_minutes_prompt_spec_expander()    # 下：プロンプト仕様の説明

OPENAI_API_KEY = (
    st.secrets.get("openai", {}).get("api_key")
    or st.secrets.get("OPENAI_API_KEY")
)
if not OPENAI_API_KEY:
    st.error("OpenAI API Key が見つかりません。.streamlit/secrets.toml を確認してください。")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

# ---- セッション初期化（表示が消えない用の保険）----
# GPTの生出力（TXT 用）
st.session_state.setdefault("minutes_raw_output", "")
# 見た目調整後（画面表示・docx 用）
st.session_state.setdefault("minutes_final_output", "")
# ★ 入力テキスト（このページ専用）
st.session_state.setdefault(SESSION_KEY_SOURCE, "")


# ========================== 補助関数（横線の後処理） ==========================

def apply_visual_mode(text: str, mode: str) -> str:
    """
    「見た目1：横線あり」→ 2つ目以降の # 見出しの前に必ず --- を追加
    「見た目2：横線なし」→ 横線を全削除

    ※ 見出しは「# 会議概要」「#会議概要」のどちらでも検出する。
    """
    # st.write("DEBUG apply_visual_mode mode=", repr(mode))  # ★一時的
    lines = text.splitlines()

    # --- 見た目2：横線なし → 全削除 ---
    if mode.startswith("見た目2"):
        return "\n".join(
            [l for l in lines if l.strip() not in ("---", "―――", "ーーー")]
        )

    # --- 見た目1：横線あり ---
    new_lines: list[str] = []
    heading_count = 0

    # 「# 会議概要」「#会議概要」など、先頭が # で始まる行を第1階層見出しとみなす
    heading_re = re.compile(r'^\s*#\s*')

    for line in lines:
        # st.write("DEBUG line=", repr(line))  # ★一時的
        if heading_re.match(line):
            # st.write("DEBUG re.match=", repr(line))  # ★一時的
            heading_count += 1

            # 2つ目以降の見出しは、前に横線を入れる
            if heading_count >= 2:
                # 直近の「非空行」を見る
                last_non_empty = None
                for prev in reversed(new_lines):
                    if prev.strip() != "":
                        last_non_empty = prev
                        break

                # 直近の非空行が横線でなければ横線を追加
                if last_non_empty is None or last_non_empty.strip() != "---":
                    new_lines.append("---")

            new_lines.append(line)
            continue

        # 見出し以外はそのまま入れる
        new_lines.append(line)

    return "\n".join(new_lines)


# ========================== レイアウト（土台） ==========================
left, right = st.columns([1, 1], gap="large")

# ========================== 左カラム：プロンプト設定 ==========================
with left:
    st.subheader("プロンプト")

    # 内容系プロンプト（逐語録 / 簡易 / 詳細）
    group = get_group(MINUTES_MAKER)
    # 見た目スタイルプロンプト（現状は UI 用。GPT には送らない想定）
    style_group = get_group(MINUTES_STYLE)

    # --- mandatory モードの初期化 ---
    mode_options = list(MINUTES_MANDATORY_MODES.keys())  # 例: ["逐語録", "簡易議事録", "詳細議事録"]

    if "minutes_mode" not in st.session_state:
        # デフォルトは「簡易議事録」
        st.session_state["minutes_mode"] = "簡易議事録"

    if "minutes_mandatory" not in st.session_state:
        # UI 上で編集するのは「モード別 mandatory」のみ
        st.session_state["minutes_mandatory"] = MINUTES_MANDATORY_MODES[
            st.session_state["minutes_mode"]
        ]

    def _on_change_minutes_mode() -> None:
        """逐語録 / 簡易 / 詳細 の切り替え時に、モード別 mandatory を差し替える。"""
        mode = st.session_state["minutes_mode"]
        st.session_state["minutes_mandatory"] = MINUTES_MANDATORY_MODES.get(
            mode,
            MINUTES_MANDATORY_MODES["簡易議事録"],
        )

    # --- 議事録の種類（内容） ---
    st.radio(
        "議事録の種類",
        options=mode_options,
        key="minutes_mode",          # index は渡さない
        on_change=_on_change_minutes_mode,
        help="逐語録 / 簡易議事録 / 詳細議事録 を切り替えます。",
    )

    # --- 見た目のスタイル（横線あり／なし を後処理で制御） ---
    if "minutes_visual_mode" not in st.session_state:
        st.session_state["minutes_visual_mode"] = "見た目1：横線あり"

    st.radio(
        "見た目のスタイル（横線）",
        options=["見た目1：横線あり", "見た目2：横線なし"],
        key="minutes_visual_mode",
        help="セクション見出し（# ...）の前に横線を自動で入れるかどうか。",
    )

    # --- プリセットなどの初期化（内容側・複数選択対応） ---
    # 選択されているプリセットキーのリスト
    if "minutes_selected_preset_keys" not in st.session_state:
        st.session_state["minutes_selected_preset_keys"] = []

    # 選択プリセットを結合した本文（ユーザー編集可）
    if "minutes_preset_text" not in st.session_state:
        st.session_state["minutes_preset_text"] = ""

    # 任意の追加指示
    if "minutes_extra_text" not in st.session_state:
        st.session_state["minutes_extra_text"] = ""

    with st.expander("必須パート（編集可：モード別）", expanded=False):
        st.text_area(
            "議事録の種類ごとに異なる必須パートです（Minutes 共通ルールはコード側で自動付与されます）。",
            height=220,
            key="minutes_mandatory",
        )

    # --- 追記プリセット（チェックボックスで複数選択 → 本文を自動結合） ---
    st.markdown("#### 追記プリセット（内容）")

    # 前回選択されていたキー
    prev_selected_keys = st.session_state.get("minutes_selected_preset_keys", [])

    # 今回の選択状態を集める
    current_selected_keys = []
    for preset in group.presets:
        # 以前選ばれていたかどうかで初期値を決める
        default_checked = preset.key in prev_selected_keys
        checked = st.checkbox(
            preset.label,
            value=default_checked,
            key=f"minutes_preset_{preset.key}",
        )
        if checked:
            current_selected_keys.append(preset.key)

    # 選択が変わったときだけ、結合テキストを再生成する
    if set(current_selected_keys) != set(prev_selected_keys):
        st.session_state["minutes_selected_preset_keys"] = current_selected_keys
        combined_body_parts = [
            p.body
            for p in group.presets
            if p.key in current_selected_keys and p.body.strip()
        ]
        st.session_state["minutes_preset_text"] = "\n\n".join(combined_body_parts).strip()
    else:
        # 念のため現在の選択も保存（初回など）
        st.session_state["minutes_selected_preset_keys"] = current_selected_keys

    # 選択されたプリセット本文（ここから自由に編集してOK）
    st.text_area(
        "（編集可）プリセット本文（内容）",
        height=120,
        key="minutes_preset_text",
    )

    # 任意の追加指示
    st.text_area("追加指示（任意）", height=88, key="minutes_extra_text")

# ========================== 右カラム：入力テキスト ==========================
with right:
    st.subheader("整形済みテキスト（入力）")

    up = st.file_uploader(
        "整形済みテキスト（.txt または .docx）をアップロードするか、下の欄に貼り付けてください。",
        type=["txt", "docx"],
        accept_multiple_files=False,
    )

    if up is not None:
        # 入力ファイル名を保持（出力ファイル名に使う）
        st.session_state["minutes_input_filename"] = up.name

        if up.name.lower().endswith(".docx"):
            if not HAS_DOCX:
                st.error("`.docx` を読み込むには python-docx が必要です。`pip install python-docx` を実行してください。")
            else:
                data = up.read()
                try:
                    doc = Document(BytesIO(data))
                    text_from_file = "\n".join(p.text for p in doc.paragraphs)
                except Exception as e:
                    st.error(f"Wordファイルの読み込みに失敗しました: {e}")
                    text_from_file = ""
                st.session_state[SESSION_KEY_SOURCE] = text_from_file
        else:
            raw = up.read()
            try:
                text_from_file = raw.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text_from_file = raw.decode("cp932")
                except Exception:
                    text_from_file = raw.decode(errors="ignore")
            st.session_state[SESSION_KEY_SOURCE] = text_from_file

    src = st.text_area(
        "テキストはここに貼り付けてください。",
        value=st.session_state.get(SESSION_KEY_SOURCE, ""),
        height=460,
        #placeholder="「③ 話者分離・整形（新）」の結果を流し込む想定です。",
    )

# ========================== サイドバー：モデル設定＋通貨 ==========================
with st.sidebar:
    st.subheader("モデル設定")

    model = st.selectbox(
        "モデル",
        [
            "gpt-5-mini",
            "gpt-5-nano",
        ],
        index=0,
    )

    st.caption("ℹ️ GPT-5 系列は temperature=1 固定・最大出力トークンは 100,000 固定です。")

    st.subheader("通貨換算（任意）")
    usd_jpy = st.number_input(
        "USD/JPY",
        min_value=50.0,
        max_value=500.0,
        value=float(DEFAULT_USDJPY),
        step=0.5,
    )

# 実行ボタン（メイン側）
run_btn = st.button("📝 議事録を生成", type="primary", use_container_width=True)

# ========================== 実行（モデル呼び出し：リトライなし） ==========================
if run_btn:
    # テキストエリアの内容を session に反映（貼り付けだけの場合も含めて）
    st.session_state[SESSION_KEY_SOURCE] = src
    if not src.strip():
        st.warning("整形済みテキストを入力してください。")
    else:
        # --- 見た目スタイルの本文を取得 ---
        # 現状、MINUTES_STYLE_PRESETS は GPT には影響しない前提だが、
        # 将来の拡張を見据えて枠だけ残しておく。
        style_body = ""
        if style_group.presets:
            style_body = style_group.presets[0].body or ""

        # --- 内容プリセット + 見た目スタイル を合体 ---
        base_preset = st.session_state.get("minutes_preset_text", "") or ""
        if style_body:
            merged_preset = (
                base_preset.strip()
                + "\n\n【見た目のスタイル指示】\n"
                + style_body.strip()
            )
        else:
            merged_preset = base_preset

        # --- 共通 mandatory + モード別 mandatory を連結 ---
        mode_specific = st.session_state.get("minutes_mandatory", "").strip()
        if mode_specific:
            mandatory_all = MINUTES_GLOBAL_MANDATORY + "\n\n" + mode_specific
        else:
            mandatory_all = MINUTES_GLOBAL_MANDATORY

        # --- プロンプト組み立て ---
        combined = build_prompt(
            mandatory_all,                               # 共通＋モード別 mandatory
            merged_preset,                               # 内容プリセット + 見た目スタイル
            st.session_state["minutes_extra_text"],      # 任意の追加指示
            src,
        )

        def call_once(prompt_text: str):
            chat_kwargs: Dict[str, Any] = dict(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                max_completion_tokens=MAX_COMPLETION_TOKENS,
            )
            # GPT-5 系列は temperature=1 固定なので設定しない
            return client.chat.completions.create(**chat_kwargs)

        t0 = time.perf_counter()
        with st.spinner("議事録を生成中…"):
            resp = call_once(combined)

            text = ""
            finish_reason = None
            if resp and getattr(resp, "choices", None):
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
            # 生のモデル出力を保存（TXT 用）
            st.session_state["minutes_raw_output"] = text

            # 横線スタイルの後処理をここで適用（画面・docx 用）
            visual_mode = st.session_state.get("minutes_visual_mode", "見た目1：横線あり")
            # st.write("DEBUG visual_mode:", visual_mode)  # 一時的に表示
            processed_text = apply_visual_mode(text, visual_mode)

            st.session_state["minutes_final_output"] = processed_text

            if finish_reason == "length":
                st.info(
                    "finish_reason=length: 出力が上限（100,000トークン）で切れています。"
                    " 必要に応じて入力テキストを分割するなどしてください。"
                )
        else:
            st.warning("⚠️ モデルから空の応答が返されました。レスポンス全体を表示します。")
            try:
                st.json(resp.model_dump())
            except Exception:
                st.write(resp)

        if "resp" in locals():
            input_tok, output_tok, total_tok = extract_tokens_from_response(resp)
            usd = estimate_chat_cost_usd(model, input_tok, output_tok)
            jpy = (usd * usd_jpy) if usd is not None else None

            metrics_data = {
                "処理時間": [f"{elapsed:.2f} 秒"],
                "入力トークン": [f"{input_tok:,}"],
                "出力トークン": [f"{output_tok:,}"],
                "合計トークン": [f"{total_tok:,}"],
                "概算 (USD/JPY)": [
                    f"${usd:,.6f} / ¥{jpy:,.2f}" if usd is not None else "—"
                ],
            }
            st.subheader("トークンと料金の概要")
            st.table(pd.DataFrame(metrics_data))

            with st.expander("🔍 トークン算出の内訳（modern usage スナップショット）"):
                try:
                    st.write(debug_usage_snapshot(getattr(resp, "usage", None)))
                except Exception as e:
                    st.write({"error": str(e)})

# ========================== 生成結果の表示 ＆ ダウンロード ==========================
raw_text = (st.session_state.get("minutes_raw_output") or "").strip()
final_text = (st.session_state.get("minutes_final_output") or "").strip()


def safe_filename(s: str) -> str:
    bad = '\\/:*?"<>|'
    for ch in bad:
        s = s.replace(ch, "_")
    return s


if final_text:
    st.markdown("### 📝 生成結果（Markdown 表示）")
    st.markdown(final_text)

    st.subheader("📥 議事録の保存")

    # 入力ファイル名（stem）を取得（なければ "minutes"）
    input_name = st.session_state.get("minutes_input_filename", "")
    input_stem = safe_filename(Path(input_name).stem) if input_name else "minutes"

    # 日時：YYYYMMDD_HHMM
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # --- TXT 保存（GPTの生出力を優先） ---
    # モード（逐語録 / 簡易議事録 / 詳細議事録）をファイル名に反映
    mode_label_for_name = st.session_state.get("minutes_mode", "議事録")
    safe_label = safe_filename(mode_label_for_name)

    base_for_txt = raw_text or final_text
    txt_bytes = base_for_txt.encode("utf-8")
    st.download_button(
        label="💾 テキストで保存 (.txt)",
        data=txt_bytes,
        file_name=f"{input_stem}_{safe_label}_{timestamp}.txt",
        mime="text/plain",
        use_container_width=True,
        key="dl_txt_minutes",
    )

    # --- DOCX 保存（lib のヘルパーに委譲） ---
    if HAS_DOCX and build_minutes_docx is not None:
        try:
            mode_label = st.session_state.get("minutes_mode", "議事録")
            visual_label = st.session_state.get("minutes_visual_mode", "")
            extra_prompt = st.session_state.get("minutes_extra_text", "").strip()
            used_model = model
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

            # メタ情報ブロックを生成
            meta_info_lines = [
                "【生成メタ情報】",
                f"- 作成日時：{now_str}",
                f"- 使用モデル：{used_model}",
                f"- 議事録の種類：{mode_label}",
                f"- 見た目のスタイル：{visual_label}",
            ]

            # 追記プリセット（内容）の選択状況をメタ情報に追加
            minutes_group = get_group(MINUTES_MAKER)  # lib.prompts から
            selected_keys = st.session_state.get("minutes_selected_preset_keys", [])

            # key → label の対応表
            label_by_key = {p.key: p.label for p in minutes_group.presets}
            selected_labels = [label_by_key[k] for k in selected_keys if k in label_by_key]

            if selected_labels:
                meta_info_lines.append("- 追記プリセット（内容）：")
                for lab in selected_labels:
                    meta_info_lines.append(f"    - {lab}")
            else:
                meta_info_lines.append("- 追記プリセット（内容）：なし")


            if extra_prompt:
                meta_info_lines.append("- 追加指示：")
                meta_info_lines.append("    " + extra_prompt.replace("\n", "\n    "))
            else:
                meta_info_lines.append("- 追加指示：なし")

            meta_info = "\n".join(meta_info_lines) + "\n\n"

            # final_text の先頭に挿入
            final_text_with_meta = meta_info + final_text

            # Word 出力生成（★ docx_minutes_export 側で Markdown 表 → Word表 に変換）
            docx_buffer = build_minutes_docx(final_text_with_meta)

            st.download_button(
                label="💾 Wordで保存 (.docx)",
                data=docx_buffer,
                file_name=f"{input_stem}_{safe_label}_{timestamp}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                key="dl_docx_minutes",
            )
        except Exception as e:
            st.error(f"Word 出力でエラーが発生しました: {e}")
else:
    st.info("整形済みテキストを入力して『📝 議事録を生成』を実行してください。")
