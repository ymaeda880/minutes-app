# -*- coding: utf-8 -*-
# pages/07_議事録作成.py
# ------------------------------------------------------------
# 📝 議事録作成（整形済みテキスト → 議事録）
#
# ✅ 新テンプレ準拠（AI実行 + busy 記録）：
# - common_lib.ai（routing）を正本として AI を呼ぶ（providers 直叩き禁止）
# - busy（ai_runs.db）を with busy_run で必ず記録
# - tokens/cost は「返ってきた範囲」で br.set_usage / br.set_cost に反映（推計しない）
# - 実行時間は get_run（ai_runs.db）を正本として UI 表示（perf_counter に依存しない）
# - cost 表示は common_lib.ai.costs.ui を使用（計算しない）
# - sidebar のモデル設定は render_text_model_picker（正本UI）＋ TEXT_MODEL_CATALOG を使用
# - provider:model 形式を正本とする（推定ロジック撤去）
#
# UI方針：
# - use_container_width は使わない
# - st.form は使わない
# - st.button()/st.download_button() に width 引数は使わない
# ------------------------------------------------------------
from __future__ import annotations

import re
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Tuple

import streamlit as st

# ============================================================
# sys.path 調整
# ============================================================
_THIS = Path(__file__).resolve()
PROJECTS_ROOT = _THIS.parents[3]
if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

# ============================================================
# ページ設定（必須・統一）
# ============================================================
st.set_page_config(
    page_title="Minutes Maker / 議事録作成",
    page_icon="🎧",
    layout="wide",
)

# ============================================================
# common_lib（正本）
# ============================================================
from common_lib.ui.page_header import render_standard_page_header
from common_lib.ui.time_format import format_jst_iso_ja
from common_lib.busy import busy_run

# AI 正本（入口：routing）
from common_lib.ai.routing import call_text

# ✅ 3方式入力（貼り付け/ファイル/Inbox）用
from common_lib.io.text import normalize_newlines
from common_lib.inbox.inbox_ui.file_picker import render_inbox_file_picker_no_toggle
from common_lib.inbox.inbox_ui.file_picker import InboxPickedFile

# run summary（共通UI）
from common_lib.ui.run_summary import render_run_summary_compact

# ============================================================
# モデル選択（テンプレ正本）
# ============================================================
from common_lib.ui.model_picker import render_text_model_picker
from common_lib.ai.models import TEXT_MODEL_CATALOG, DEFAULT_TEXT_MODEL_KEY

# ============================================================
# 既存プロンプト資産（ページ責務）
# ============================================================
from lib.prompts import (
    MINUTES_MAKER,
    MINUTES_MANDATORY_MODES,
    MINUTES_STYLE,
    MINUTES_GLOBAL_MANDATORY,
    DEFAULT_MINUTES_MODE,
    get_group,
    build_prompt,
)

from lib.minutes_generation.explanation import (
    render_minutes_page_intro,
    render_minutes_help_expander,
)

# ============================================================
# 要約長設定（話者別要約など）
# ============================================================
from lib.minutes_generation.summary_length import (
    CUSTOM_LABEL,
    build_summary_length_instruction,
    get_summary_length_config,
    get_summary_length_options,
    merge_extra_with_summary_length_instruction,
    resolve_target_chars,
)

# ============================================================
# Gemini 利用可否チェック（テンプレ同一：google-genai）
# ============================================================
def _gemini_available() -> bool:
    try:
        from google import genai  # google-genai
        _ = genai
        return True
    except Exception:
        return False


def _parse_model_key(model_key: str) -> Tuple[str, str]:
    if ":" not in model_key:
        return ("openai", model_key)
    prov, mdl = model_key.split(":", 1)
    return (prov.strip(), mdl.strip())


# ============================================================
# ページ専用キー
# ============================================================
APP_NAME = _THIS.parents[1].name  # minutes_app
PAGE_NAME = _THIS.stem           # 07_議事録作成（新）
SESSION_KEY_SOURCE = f"{PAGE_NAME}_source_text"

# 入力方式
K_INPUT_METHOD = f"{PAGE_NAME}_input_method"
INPUT_PASTE = "📝 貼り付けテキスト"
INPUT_FILE = "📁 ファイルから"
INPUT_INBOX = "📥 Inboxから"

# Inbox 選択保持（rerun 対策）
K_INBOX_BYTES = f"{PAGE_NAME}_inbox_bytes"
K_INBOX_NAME = f"{PAGE_NAME}_inbox_name"
K_INBOX_KIND = f"{PAGE_NAME}_inbox_kind"
K_INBOX_ITEM = f"{PAGE_NAME}_inbox_item_id"
K_INBOX_ADDED = f"{PAGE_NAME}_inbox_added_at"

# モデル（正本UI）
K_MODEL_KEY = f"{PAGE_NAME}__model_key"

# busy 表示用
K_LAST_RUN_ID = f"{PAGE_NAME}__last_run_id"
K_LAST_RUN_ACTION = f"{PAGE_NAME}__last_run_action"

# ============================================================
# 要約長設定（話者別要約など）
# ============================================================
K_SUMMARY_LENGTH_LABEL = f"{PAGE_NAME}__summary_length_label"
K_SUMMARY_LENGTH_CUSTOM_CHARS = f"{PAGE_NAME}__summary_length_custom_chars"

# usage/cost 表示用（推計しない：返ってきた範囲だけ）
K_LAST_IN_TOK = f"{PAGE_NAME}__last_in_tok"
K_LAST_OUT_TOK = f"{PAGE_NAME}__last_out_tok"
K_LAST_COST_OBJ = f"{PAGE_NAME}__last_cost_obj"
K_LAST_MODEL = f"{PAGE_NAME}__last_model"
K_LAST_PROVIDER = f"{PAGE_NAME}__last_provider"
K_LAST_COST_NOTE = f"{PAGE_NAME}__last_cost_note"

# ============================================================
# .docx 読み取り／書き出し（python-docx）
# ============================================================
try:
    from docx import Document  # type: ignore
    from lib.docx_minutes_export import build_minutes_docx  # type: ignore

    HAS_DOCX = True
except Exception:
    HAS_DOCX = False
    build_minutes_docx = None  # type: ignore
    Document = None  # type: ignore

# ============================================================
# 共通ヘッダー
# - settings.toml から BANNER_KEY を取得
# - banner / theme / intro CSS を描画
# - page_session_heartbeat を実行
# - title / subtitle / ログイン状態を描画
# ============================================================
sub, theme, BANNER_KEY, settings = render_standard_page_header(
    st_module=st,
    projects_root=PROJECTS_ROOT,
    app_dir=_THIS.parents[1],
    app_name=APP_NAME,
    page_name=PAGE_NAME,
    title="🎧 議事録作成",
    subtitle_text="逐語録から正式議事録へ",
    default_banner_key="light_green",
)

# ============================================================
# ページ説明
# ============================================================
render_minutes_page_intro()

# ============================================================
# 詳細説明
# ============================================================
render_minutes_help_expander(
    theme=theme,
    banner_key=BANNER_KEY,
)

# ============================================================
# セッション初期化（表示が消えない用の保険）
# ============================================================
st.session_state.setdefault("minutes_raw_output", "")
st.session_state.setdefault("minutes_final_output", "")
st.session_state.setdefault(SESSION_KEY_SOURCE, "")

# 入力方式（３つ）
st.session_state.setdefault(K_INPUT_METHOD, INPUT_FILE)
st.session_state.setdefault(K_INBOX_BYTES, b"")
st.session_state.setdefault(K_INBOX_NAME, "")
st.session_state.setdefault(K_INBOX_KIND, "")
st.session_state.setdefault(K_INBOX_ITEM, "")
st.session_state.setdefault(K_INBOX_ADDED, "")

# モデル（正本UI）
st.session_state.setdefault(K_MODEL_KEY, DEFAULT_TEXT_MODEL_KEY)

# 要約長設定（初期値）
st.session_state.setdefault(K_SUMMARY_LENGTH_LABEL, "標準（500字程度）")
st.session_state.setdefault(K_SUMMARY_LENGTH_CUSTOM_CHARS, 500)

# busy
st.session_state.setdefault(K_LAST_RUN_ID, "")
st.session_state.setdefault(K_LAST_RUN_ACTION, "")

# usage/cost（表示用）
st.session_state.setdefault(K_LAST_IN_TOK, None)
st.session_state.setdefault(K_LAST_OUT_TOK, None)
st.session_state.setdefault(K_LAST_COST_OBJ, None)
st.session_state.setdefault(K_LAST_MODEL, "")
st.session_state.setdefault(K_LAST_PROVIDER, "")
st.session_state.setdefault(K_LAST_COST_NOTE, "")

# ============================================================
# サイドバー：モデル設定（テンプレ準拠）
# ============================================================
with st.sidebar:
    st.subheader("モデル設定")

    model_key = render_text_model_picker(
        title="モデル選択",
        catalog=TEXT_MODEL_CATALOG,
        session_key=K_MODEL_KEY,
        default_key=DEFAULT_TEXT_MODEL_KEY,
        page_name=PAGE_NAME,
        gemini_available=_gemini_available(),
    )

    provider, chosen_model = _parse_model_key(model_key)
    #st.caption("ℹ️ OpenAI / Gemini ともに common_lib.ai（routing）経由で単発生成します。")

# ============================================================
# 補助関数（横線の後処理）
# ============================================================
def apply_visual_mode(text: str, mode: str) -> str:
    """
    「見た目1：横線あり」→ 2つ目以降の # 見出しの前に必ず --- を追加
    「見た目2：横線なし」→ 横線を全削除
    """
    lines = text.splitlines()

    if mode.startswith("見た目2"):
        return "\n".join([l for l in lines if l.strip() not in ("---", "―――", "ーーー")])

    new_lines: list[str] = []
    heading_count = 0
    heading_re = re.compile(r"^\s*#\s*")

    for line in lines:
        if heading_re.match(line):
            heading_count += 1
            if heading_count >= 2:
                last_non_empty = None
                for prev in reversed(new_lines):
                    if prev.strip() != "":
                        last_non_empty = prev
                        break
                if last_non_empty is None or last_non_empty.strip() != "---":
                    new_lines.append("---")
            new_lines.append(line)
            continue
        new_lines.append(line)

    return "\n".join(new_lines)


def safe_filename(s: str) -> str:
    bad = '\\/:*?"<>|'
    for ch in bad:
        s = s.replace(ch, "_")
    return s


def _decode_text_bytes(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return b.decode("utf-8", errors="replace")


def _text_from_docx_bytes(data: bytes) -> str:
    if not HAS_DOCX:
        return ""
    try:
        doc = Document(BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""


def _extract_usage_tokens_from_result(res: Any) -> tuple[Optional[int], Optional[int]]:
    """
    common_lib.ai の result から tokens を安全に抽出。
    - res.usage が dict / object の両方に対応
    - 推計しない（取れなければ None）
    """
    usage = getattr(res, "usage", None)
    if usage is None:
        return (None, None)

    def _get(u: Any, key: str) -> Optional[int]:
        if u is None:
            return None
        if isinstance(u, dict):
            v = u.get(key)
        else:
            v = getattr(u, key, None)
        if v is None:
            return None
        try:
            return int(v)
        except Exception:
            return None

    in_tok = _get(usage, "input_tokens")
    out_tok = _get(usage, "output_tokens")
    if in_tok is None:
        in_tok = _get(usage, "prompt_tokens")
    if out_tok is None:
        out_tok = _get(usage, "completion_tokens")

    return (in_tok, out_tok)

# ============================================================
# 上：入力テキスト（3方式）
# ============================================================
st.divider()
st.subheader("① 逐語録入力")

INBOX_PAGE_SIZE = 6

picked_method = st.radio(
    "入力方法",
    [INPUT_FILE, INPUT_PASTE, INPUT_INBOX],
    key=K_INPUT_METHOD,
    horizontal=True,
)

src_text: str = ""
used_file_name: str = ""

# ------------------------------------------------------------
# ① ファイルから（.txt / .docx）
# ------------------------------------------------------------
if picked_method == INPUT_FILE:
    up = st.file_uploader(
        "整形済みテキスト（.txt / .docx）をアップロード",
        type=["txt", "docx"],
        accept_multiple_files=False,
        key=f"{PAGE_NAME}_uploader",
    )

    do_set_file = st.button(
        "セット（ファイル）",
        type="primary",
        disabled=(up is None),
        key=f"{PAGE_NAME}_btn_set_file",
    )

    if up is not None:
        used_file_name = up.name
        data = up.read()

        if up.name.lower().endswith(".docx"):
            text_from_file = _text_from_docx_bytes(data)
            if (not text_from_file) and (not HAS_DOCX):
                st.error("`.docx` を読むには python-docx が必要です。`pip install python-docx` を確認してください。")
        else:
            try:
                text_from_file = data.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    text_from_file = data.decode("cp932")
                except Exception:
                    text_from_file = data.decode(errors="ignore")

        st.caption(f"選択中: {used_file_name} / length: {len((text_from_file or '').strip()):,} chars")
    else:
        text_from_file = ""

    if do_set_file:
        if not (text_from_file or "").strip():
            st.warning("ファイルからテキストを取得できませんでした。")
            st.stop()

        st.session_state[SESSION_KEY_SOURCE] = (text_from_file or "").strip()
        st.session_state["minutes_input_filename"] = used_file_name or "input.txt"
        st.success("✅ 入力テキストをこのページにセットしました。")

# ------------------------------------------------------------
# ② 貼り付けテキスト
# ------------------------------------------------------------
elif picked_method == INPUT_PASTE:
    pasted = st.text_area(
        "ここに本文を貼り付け",
        height=260,
        key=f"{PAGE_NAME}_pasted_text",
        placeholder="ここに本文を貼り付けてください（改行は保持されます）。",
    )

    do_set_paste = st.button(
        "セット（貼り付け）",
        type="primary",
        key=f"{PAGE_NAME}_btn_set_paste",
    )

    if do_set_paste:
        if not (pasted or "").strip():
            st.warning("テキストを貼り付けてください。")
            st.stop()

        st.session_state[SESSION_KEY_SOURCE] = normalize_newlines(pasted).strip()
        st.session_state["minutes_input_filename"] = "pasted_text.txt"
        st.success("✅ 入力テキストをこのページにセットしました。")

# ------------------------------------------------------------
# ③ Inbox から（kind=text）
# ------------------------------------------------------------
else:
    st.caption("Inbox（kind=text）から読み込みます。last_viewed は更新しません。")

    picked: InboxPickedFile | None = render_inbox_file_picker_no_toggle(
        projects_root=PROJECTS_ROOT,
        user_sub=sub,
        key_prefix=f"{PAGE_NAME}_inbox_picker",
        page_size=INBOX_PAGE_SIZE,
        kinds=["text"],
        show_kind_in_label=True,
        show_added_at_in_label=True,
    )

    if picked is not None:
        st.session_state[K_INBOX_BYTES] = picked.data_bytes or b""
        st.session_state[K_INBOX_NAME] = picked.original_name or "inbox_text.txt"
        st.session_state[K_INBOX_KIND] = picked.kind or "text"
        st.session_state[K_INBOX_ITEM] = str(picked.item_id or "")
        st.session_state[K_INBOX_ADDED] = str(getattr(picked, "added_at", "") or "")
        st.success("✅ Inbox から読み込みました（セットをしてください）")

    kept_bytes: bytes = st.session_state.get(K_INBOX_BYTES, b"") or b""
    kept_name: str = st.session_state.get(K_INBOX_NAME, "") or ""
    kept_item: str = st.session_state.get(K_INBOX_ITEM, "") or ""
    kept_added: str = st.session_state.get(K_INBOX_ADDED, "") or ""

    if kept_bytes:
        st.caption(f"(保持中) item_id={kept_item} / name={kept_name} / added_at={kept_added} / size={len(kept_bytes):,} bytes")
    else:
        st.caption("(保持中) まだ選択されていません。")

    do_set_inbox = st.button(
        "セット（Inbox）",
        type="primary",
        disabled=(not bool(kept_bytes)),
        key=f"{PAGE_NAME}_btn_set_inbox",
    )

    if do_set_inbox:
        if not kept_bytes:
            st.warning("Inbox からテキストを選択してください。")
            st.stop()

        if kept_name.lower().endswith(".docx"):
            txt = _text_from_docx_bytes(kept_bytes)
            if (not txt) and (not HAS_DOCX):
                st.error("`.docx` を読むには python-docx が必要です。`pip install python-docx` を確認してください。")
                st.stop()
        else:
            txt = _decode_text_bytes(kept_bytes)

        src_text = (txt or "").strip()
        if not src_text:
            st.warning("テキストが空でした（0文字）。別のファイルを選択してください。")
            st.stop()

        st.session_state[SESSION_KEY_SOURCE] = src_text
        st.session_state["minutes_input_filename"] = kept_name or "inbox_text.txt"
        st.success("✅ 入力テキストをこのページにセットしました。")

# ------------------------------------------------------------
# 確定済み入力（このページの入力欄）
# ------------------------------------------------------------
src = st.text_area(
    "確定済みテキスト（この内容が『📝 議事録を生成』に使われます）",
    value=st.session_state.get(SESSION_KEY_SOURCE, ""),
    height=200,
)

# ============================================================
# 下：プロンプト設定
# ============================================================
st.divider()
st.subheader("② 議事録形式の設定")

group = get_group(MINUTES_MAKER)
style_group = get_group(MINUTES_STYLE)

mode_options = list(MINUTES_MANDATORY_MODES.keys())

# ============================================================
# 追記プリセット：default=True を初回だけ反映（lib/prompts 正本）
# - 既にユーザーが選択済みなら上書きしない
# ============================================================
K_PRESETS_INIT = f"{PAGE_NAME}__presets_initialized"

if K_PRESETS_INIT not in st.session_state:
    st.session_state[K_PRESETS_INIT] = True

    default_keys = [p.key for p in (group.presets or []) if getattr(p, "default", False)]

    # 初期値を入れる（このページで初回だけ）
    st.session_state["minutes_selected_preset_keys"] = list(default_keys)

    # 選択に対応する本文も同期（後段の merged_preset がこれを使う）
    combined_body_parts = [
        p.body for p in (group.presets or [])
        if (p.key in default_keys) and (getattr(p, "body", "") or "").strip()
    ]
    st.session_state["minutes_preset_text"] = "\n\n".join(combined_body_parts).strip()

# 互換のため：未定義なら空で用意（以降のコードが落ちないように）
st.session_state.setdefault("minutes_selected_preset_keys", [])
st.session_state.setdefault("minutes_preset_text", "")


# ===== デフォルトの議事録 =====
# - 正本（lib/prompts.py）の DEFAULT_MINUTES_MODE に追従する
if "minutes_mode" not in st.session_state:
    st.session_state["minutes_mode"] = DEFAULT_MINUTES_MODE

if "minutes_mandatory" not in st.session_state:
    st.session_state["minutes_mandatory"] = MINUTES_MANDATORY_MODES[DEFAULT_MINUTES_MODE]


def _on_change_minutes_mode() -> None:
    mode = st.session_state.get("minutes_mode")

    if mode in MINUTES_MANDATORY_MODES:
        st.session_state["minutes_mandatory"] = MINUTES_MANDATORY_MODES[mode]
    else:
        # 想定外の値が入ったら正本デフォルトに戻す（KeyError回避）
        st.session_state["minutes_mode"] = DEFAULT_MINUTES_MODE
        st.session_state["minutes_mandatory"] = MINUTES_MANDATORY_MODES[DEFAULT_MINUTES_MODE]


# ============================================================
# 議事録の種類（右の空きに説明文）
# ============================================================
col_left, col_right = st.columns([3, 7])

with col_left:
    st.radio(
        "議事録の種類",
        options=mode_options,
        key="minutes_mode",
        on_change=_on_change_minutes_mode,
        help="逐語録 / 簡易議事録 / 詳細議事録 を切り替えます。",
    )

with col_right:
    # 右側の説明（自由に文章を書ける）
    st.caption("議事録：サマリー，決定事項，作業項目，重要トピック，解決事項・次回アジェンダなどからなる正式議事録")
    st.caption("決定事項・課題まとめ：会議で決定した事項と課題として残った事項をまとめた議事録")
    st.caption("質問・回答まとめ：会議で出た質問とそれに対する回答をまとめた議事録")
    st.caption("主な意見と回答(分類付き)：会議で出た意見や質問とそれに対する回答を表の形にまとめた議事録（意見や質問の分類も行う）")
    st.caption("話者別要約：会議全体の発言を話者ごとに要約した議事録")
    st.caption("要約逐語録：各話者の各発言を話の流れに沿って（時系列を崩さずに）要約した議事録"
               "（注意：要約逐語録を作成するときは，発言数をおおよそ80以内に区切って作成してください．"
               "いくつかに分けて作成した要約議事録を，後で人手で結合して１つの議事録としてください．"
               "時系列的に長い発言の列の順番を正確に記憶することがAIは苦手です．）")
    # st.write("もう少し長い文章もOK。ここは自由枠です。")

# ============================================================
# 要約長設定UI
# - まずは「話者別要約」にだけ接続
# - 将来「議事録」「会議全体要約」にも同じ仕組みを適用する
# ============================================================
current_minutes_mode = str(st.session_state.get("minutes_mode", "") or "")
summary_length_config = get_summary_length_config(current_minutes_mode)

if summary_length_config is not None:
    st.markdown("#### 要約の長さ")

    summary_length_options = get_summary_length_options(current_minutes_mode)

    # ------------------------------------------------------------
    # モード変更時に、そのモードの既定ラベルへ補正する
    # ------------------------------------------------------------
    current_length_label = str(st.session_state.get(K_SUMMARY_LENGTH_LABEL, "") or "")
    if current_length_label not in summary_length_options:
        st.session_state[K_SUMMARY_LENGTH_LABEL] = summary_length_config.default_label
        current_length_label = summary_length_config.default_label

    st.radio(
        summary_length_config.title,
        options=summary_length_options,
        key=K_SUMMARY_LENGTH_LABEL,
        horizontal=True,
    )

    # ------------------------------------------------------------
    # カスタム選択時のみ、文字数入力を表示する
    # ------------------------------------------------------------
    if st.session_state.get(K_SUMMARY_LENGTH_LABEL) == CUSTOM_LABEL:
        st.number_input(
            f"{summary_length_config.custom_label}"
            f"（{summary_length_config.min_chars}字以上）",
            min_value=summary_length_config.min_chars,
            step=summary_length_config.step_chars,
            key=K_SUMMARY_LENGTH_CUSTOM_CHARS,
        )

st.markdown("#### スタイルの調整")
st.radio(
    "横線",
    options=["横線あり", "横線なし"],
    key="minutes_visual_mode",
)

if "minutes_selected_preset_keys" not in st.session_state:
    st.session_state["minutes_selected_preset_keys"] = []
if "minutes_preset_text" not in st.session_state:
    st.session_state["minutes_preset_text"] = ""
if "minutes_extra_text" not in st.session_state:
    st.session_state["minutes_extra_text"] = ""

# with st.expander("必須パート（編集可：モード別）", expanded=False):
#     st.text_area(
#         "議事録の種類ごとに異なる必須パートです（Minutes 共通ルールはコード側で自動付与されます）。",
#         height=220,
#         key="minutes_mandatory",
#     )

st.markdown("#### 追記プリセット")

prev_selected_keys = st.session_state.get("minutes_selected_preset_keys", [])
current_selected_keys = []

for preset in group.presets:
    checked = st.checkbox(
        preset.label,
        value=preset.key in prev_selected_keys,
        key=f"minutes_preset_{preset.key}",
    )
    if checked:
        current_selected_keys.append(preset.key)

if set(current_selected_keys) != set(prev_selected_keys):
    st.session_state["minutes_selected_preset_keys"] = current_selected_keys
    combined_body_parts = [p.body for p in group.presets if p.key in current_selected_keys and p.body.strip()]
    st.session_state["minutes_preset_text"] = "\n\n".join(combined_body_parts).strip()

# st.text_area(
#     "（編集可）プリセット本文（内容）",
#     height=120,
#     key="minutes_preset_text",
# )

st.markdown("#### 追加指示")
st.text_area("追加指示（任意）", height=88, key="minutes_extra_text")

# ============================================================
# 実行
# ============================================================
st.divider()
st.subheader("③ 議事録を生成")
run_btn = st.button("📝 議事録を生成", type="primary")

if run_btn:
    st.session_state[SESSION_KEY_SOURCE] = src

    if not src.strip():
        st.warning("整形済みテキストを入力してください。")
        st.stop()

    # ------------------------------------------------------------
    # provider/model（正本：model_key から分解）
    # ------------------------------------------------------------
    model_key = str(st.session_state.get(K_MODEL_KEY, DEFAULT_TEXT_MODEL_KEY))
    provider, chosen_model = _parse_model_key(model_key)

    # ------------------------------------------------------------
    # 見た目スタイルの本文（枠は保持）
    # ------------------------------------------------------------
    style_body = ""
    if style_group.presets:
        style_body = style_group.presets[0].body or ""

    base_preset = st.session_state.get("minutes_preset_text", "") or ""
    if style_body:
        merged_preset = base_preset.strip() + "\n\n【見た目のスタイル指示】\n" + style_body.strip()
    else:
        merged_preset = base_preset

    # ------------------------------------------------------------
    # 共通 mandatory + モード別 mandatory を連結
    # ------------------------------------------------------------
    mode_specific = (st.session_state.get("minutes_mandatory", "") or "").strip()
    if mode_specific:
        mandatory_all = MINUTES_GLOBAL_MANDATORY + "\n\n" + mode_specific
    else:
        mandatory_all = MINUTES_GLOBAL_MANDATORY


    # ------------------------------------------------------------
    # 要約長設定を追加指示に反映
    # - まずは「話者別要約」に接続
    # - summary_length.py 側の設定があるモードだけ反映される
    # ------------------------------------------------------------
    summary_length_instruction = ""

    current_minutes_mode = str(st.session_state.get("minutes_mode", "") or "")
    summary_length_config = get_summary_length_config(current_minutes_mode)

    if summary_length_config is not None:
        selected_length_label = str(st.session_state.get(K_SUMMARY_LENGTH_LABEL, "") or "")
        custom_chars = st.session_state.get(K_SUMMARY_LENGTH_CUSTOM_CHARS)

        target_chars = resolve_target_chars(
            mode=current_minutes_mode,
            selected_label=selected_length_label,
            custom_chars=custom_chars,
        )

        summary_length_instruction = build_summary_length_instruction(
            mode=current_minutes_mode,
            target_chars=target_chars,
        )

    extra_text_for_prompt = merge_extra_with_summary_length_instruction(
        extra_text=st.session_state.get("minutes_extra_text", "") or "",
        summary_length_instruction=summary_length_instruction,
    )

    # ------------------------------------------------------------
    # プロンプト組み立て
    # ------------------------------------------------------------
    combined = build_prompt(
        mandatory_all,
        merged_preset,
        extra_text_for_prompt,
        src,
    )

    # ------------------------------------------------------------
    # 初期化（表示用）
    # ------------------------------------------------------------
    st.session_state[K_LAST_IN_TOK] = None
    st.session_state[K_LAST_OUT_TOK] = None
    st.session_state[K_LAST_COST_OBJ] = None
    st.session_state[K_LAST_COST_NOTE] = ""
    st.session_state[K_LAST_MODEL] = chosen_model
    st.session_state[K_LAST_PROVIDER] = provider

    try:
        with busy_run(
            projects_root=PROJECTS_ROOT,
            user_sub=str(sub),
            app_name=str(APP_NAME),
            page_name=str(PAGE_NAME),
            task_type="text",
            provider=str(provider),
            model=str(chosen_model),
            meta={
                "feature": "minutes_maker",
                "action": "generate_minutes",
                "input_method": str(picked_method),
                "prompt_chars": len(combined or ""),
                "src_chars": len(src or ""),
                "minutes_mode": str(st.session_state.get("minutes_mode") or ""),
                "visual_mode": str(st.session_state.get("minutes_visual_mode") or ""),
                "model_key": str(model_key),
            },
        ) as br:
            with st.spinner("議事録を生成中…"):
                res = call_text(
                    provider=str(provider),
                    model=str(chosen_model),
                    prompt=str(combined),
                    system=None,
                    temperature=None,
                    max_output_tokens=None,
                    extra=None,
                )

            text = (getattr(res, "text", "") or "").strip()

            # ---- usage（推計しない：取れたら反映）----
            in_tok, out_tok = _extract_usage_tokens_from_result(res)
            st.session_state[K_LAST_IN_TOK] = in_tok
            st.session_state[K_LAST_OUT_TOK] = out_tok

            if isinstance(in_tok, int) and isinstance(out_tok, int):
                br.set_usage(int(in_tok), int(out_tok))

            # ---- cost（推計しない：res.cost があれば反映）----
            cost_obj = getattr(res, "cost", None)
            st.session_state[K_LAST_COST_OBJ] = cost_obj

            usd = getattr(cost_obj, "usd", None) if cost_obj is not None else None
            jpy = getattr(cost_obj, "jpy", None) if cost_obj is not None else None
            if isinstance(usd, (int, float)) and isinstance(jpy, (int, float)):
                br.set_cost(float(usd), float(jpy))

            br.add_finish_meta(note="ok")

            st.session_state[K_LAST_RUN_ID] = br.run_id
            st.session_state[K_LAST_RUN_ACTION] = "generate_minutes"

    except Exception as e:
        st.error(f"AI 呼び出しでエラー: {e}")
        st.stop()

    if text.strip():
        st.session_state["minutes_raw_output"] = text

        visual_mode = st.session_state.get("minutes_visual_mode", "見た目1：横線あり")
        processed_text = apply_visual_mode(text, visual_mode)
        st.session_state["minutes_final_output"] = processed_text
    else:
        st.warning("⚠️ モデルから空の応答が返されました。")

    st.success("生成が完了しました。")

# ============================================================
# 生成結果の表示 ＆ ダウンロード
# ============================================================
raw_text = (st.session_state.get("minutes_raw_output") or "").strip()
final_text = (st.session_state.get("minutes_final_output") or "").strip()


# ============================================================
# 議事録プレビュー用 CSS
# - GPT出力は変更しない
# - 画面表示だけ h1/h2/h3 を小さくする
# ============================================================
st.markdown(
    """
    <style>
    .minutes-preview h1 {
        font-size: 1.6rem !important;
        margin-top: 0.8rem !important;
        margin-bottom: 0.6rem !important;
    }

    .minutes-preview h2 {
        font-size: 1.35rem !important;
        margin-top: 0.7rem !important;
        margin-bottom: 0.5rem !important;
    }

    .minutes-preview h3 {
        font-size: 1.15rem !important;
        margin-top: 0.6rem !important;
        margin-bottom: 0.4rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if final_text:
    # ------------------------------------------------------------
    # Markdown表示用：単一改行だけを「強制改行」に変換（空行は段落として保持）
    # ------------------------------------------------------------
    final_text_md = re.sub(r"(?<!\n)\n(?!\n)", "  \n", final_text)

    st.markdown("### 📝 生成結果")
    #st.markdown(final_text_md)
    st.markdown(
        f'<div class="minutes-preview">\n\n{final_text_md}\n\n</div>',
        unsafe_allow_html=True,
    )


    st.subheader("④ 議事録の保存")

    input_name = st.session_state.get("minutes_input_filename", "")
    input_stem = safe_filename(Path(input_name).stem) if input_name else "minutes"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    mode_label_for_name = st.session_state.get("minutes_mode", "議事録")
    safe_label = safe_filename(mode_label_for_name)

    # --- TXT 保存（生出力優先） ---
    # 方針：
    # - 正本は raw_text（モデル生出力）を優先
    # - ただし「空行（段落）」は保存では不要なので、\n\n 以上を \n に畳む
    base_for_txt = (raw_text or final_text)

    # 保存用：空行を除去（段落を作らない）
    base_for_txt_save = re.sub(r"\n\n+", "\n", base_for_txt).strip() + "\n"

    txt_bytes = base_for_txt_save.encode("utf-8")
    st.download_button(
        label="💾 テキストで保存 (.txt)",
        data=txt_bytes,
        file_name=f"{input_stem}_{safe_label}_{timestamp}.txt",
        mime="text/plain",
        key="dl_txt_minutes",
    )


    # --- DOCX 保存（lib のヘルパーに委譲） ---
    if HAS_DOCX and build_minutes_docx is not None:
        try:
            mode_label = st.session_state.get("minutes_mode", "議事録")
            visual_label = st.session_state.get("minutes_visual_mode", "")
            extra_prompt = (st.session_state.get("minutes_extra_text", "") or "").strip()
            used_model_key = str(st.session_state.get(K_MODEL_KEY, "—"))
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

            meta_info_lines = [
                "【生成メタ情報】",
                f"- 作成日時：{now_str}",
                f"- 使用モデル：{used_model_key}",
                f"- 議事録の種類：{mode_label}",
                f"- 見た目のスタイル：{visual_label}",
            ]

            minutes_group = get_group(MINUTES_MAKER)
            selected_keys = st.session_state.get("minutes_selected_preset_keys", [])

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
            # --- Word 保存用：空行を除去（TXT と同一の正本ルール） ---
            base_for_docx = raw_text or final_text
            base_for_docx_save = re.sub(r"\n\n+", "\n", base_for_docx).strip()

            final_text_with_meta = meta_info + base_for_docx_save
            docx_buffer = build_minutes_docx(final_text_with_meta)


            st.download_button(
                label="💾 Wordで保存 (.docx)",
                data=docx_buffer,
                file_name=f"{input_stem}_{safe_label}_{timestamp}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="dl_docx_minutes",
            )
        except Exception as e:
            st.error(f"Word 出力でエラーが発生しました: {e}")
else:
    st.info("逐語録を入力して『📝 議事録を生成』を実行してください。")

# ============================================================
# 📊 実行時間（get_run 正本）＋ cost UI（正本）＋ 実行サマリ（共通UI）
# ============================================================
last_run_id = str(st.session_state.get(K_LAST_RUN_ID) or "").strip()
has_run = bool(last_run_id)

if has_run:

    # ------------------------------------------------------------
    # 実行サマリ（共通UI）
    # ------------------------------------------------------------
    render_run_summary_compact(
        projects_root=PROJECTS_ROOT,
        run_id=last_run_id,
        model=st.session_state.get(K_LAST_MODEL),
        in_tokens=st.session_state.get(K_LAST_IN_TOK),
        out_tokens=st.session_state.get(K_LAST_OUT_TOK),
        cost=st.session_state.get(K_LAST_COST_OBJ),
        note="",
        show_divider=True,
    )
