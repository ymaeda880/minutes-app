# -*- coding: utf-8 -*-
# minutes_app/pages/08_要約逐語録作成.py
# ============================================================
# 📝 要約逐語録作成
# ============================================================

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Tuple
from types import SimpleNamespace

import streamlit as st

# ============================================================
# sys.path 調整
# ============================================================
_THIS = Path(__file__).resolve()
PROJECTS_ROOT = _THIS.parents[3]
if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

# ============================================================
# ページ設定
# ============================================================
st.set_page_config(
    page_title="Minutes Maker / 要約逐語録作成",
    page_icon="📝",
    layout="wide",
)

# ============================================================
# common_lib（正本）
# ============================================================
from common_lib.ui.page_header import render_standard_page_header
from common_lib.busy import busy_run
from common_lib.ai.routing import call_text
from common_lib.io.text import normalize_newlines
from common_lib.inbox.inbox_ui.file_picker import render_inbox_file_picker_no_toggle
from common_lib.inbox.inbox_ui.file_picker import InboxPickedFile
from common_lib.ui.run_summary import render_run_summary_compact

# ============================================================
# モデル選択（正本）
# ============================================================
from common_lib.ui.model_picker import render_text_model_picker
from common_lib.ai.models import TEXT_MODEL_CATALOG, DEFAULT_TEXT_MODEL_KEY

# ============================================================
# 既存プロンプト資産（07と共通）
# ============================================================
from lib.prompts import (
    MINUTES_MAKER,
    MINUTES_STYLE,
    get_group,
)

# ============================================================
# 08専用プロンプト
# ============================================================
from lib.summarized_transcript.summary_verbatim_prompts import build_summary_verbatim_prompt

# ============================================================
# 要約逐語録 説明UI
# ============================================================
from lib.summarized_transcript.explanation import (
    render_summarized_transcript_page_intro,
    render_summarized_transcript_help_expander,
)

# ============================================================
# .docx 読み取り
# ============================================================
try:
    from docx import Document  # type: ignore
    HAS_DOCX = True
except Exception:
    Document = None  # type: ignore
    HAS_DOCX = False


# ============================================================
# Gemini 利用可否チェック
# ============================================================
def _gemini_available() -> bool:
    try:
        from google import genai
        _ = genai
        return True
    except Exception:
        return False


# ============================================================
# provider:model 分解
# ============================================================
def _parse_model_key(model_key: str) -> Tuple[str, str]:
    if ":" not in model_key:
        return ("openai", model_key)
    prov, mdl = model_key.split(":", 1)
    return (prov.strip(), mdl.strip())


# ============================================================
# ページ専用キー
# ============================================================
APP_NAME = _THIS.parents[1].name
PAGE_NAME = _THIS.stem

SESSION_KEY_SOURCE = f"{PAGE_NAME}__source_text"
K_CONFIRMED_TEXT = f"{PAGE_NAME}__confirmed_text"

K_INPUT_METHOD = f"{PAGE_NAME}__input_method"
INPUT_FILE = "📁 ファイルから"
INPUT_PASTE = "📝 貼り付けテキスト"
INPUT_INBOX = "📥 Inboxから"

K_INBOX_BYTES = f"{PAGE_NAME}__inbox_bytes"
K_INBOX_NAME = f"{PAGE_NAME}__inbox_name"
K_INBOX_KIND = f"{PAGE_NAME}__inbox_kind"
K_INBOX_ITEM = f"{PAGE_NAME}__inbox_item_id"
K_INBOX_ADDED = f"{PAGE_NAME}__inbox_added_at"

K_MODEL_KEY = f"{PAGE_NAME}__model_key"

K_MAX_UTTERANCES = f"{PAGE_NAME}__max_utterances"
K_SEPARATOR_MODE = f"{PAGE_NAME}__separator_mode"
K_CUSTOM_SEPARATOR = f"{PAGE_NAME}__custom_separator"

K_LAST_RUN_ID = f"{PAGE_NAME}__last_run_id"
K_LAST_IN_TOK = f"{PAGE_NAME}__last_in_tok"
K_LAST_OUT_TOK = f"{PAGE_NAME}__last_out_tok"
K_LAST_COST_OBJ = f"{PAGE_NAME}__last_cost_obj"
K_LAST_MODEL = f"{PAGE_NAME}__last_model"
K_LAST_PROVIDER = f"{PAGE_NAME}__last_provider"

K_SUMMARY_OUTPUT = f"{PAGE_NAME}__summary_output"
K_NUMBERED_SUMMARY_OUTPUT = f"{PAGE_NAME}__numbered_summary_output"
K_EXPLANATION_OUTPUT = f"{PAGE_NAME}__explanation_output"
K_NUMBERED_OUTPUT = f"{PAGE_NAME}__numbered_output"
K_CHUNK_INFO_OUTPUT = f"{PAGE_NAME}__chunk_info_output"


# ============================================================
# データ構造
# ============================================================
@dataclass(frozen=True)
class Utterance:
    no: int
    speaker: str
    body: str
    raw: str


@dataclass(frozen=True)
class Chunk:
    index: int
    total: int
    start_no: int
    end_no: int
    utterances: list[Utterance]


# ============================================================
# 共通ヘッダー
# ============================================================
sub, theme, BANNER_KEY, settings = render_standard_page_header(
    st_module=st,
    projects_root=PROJECTS_ROOT,
    app_dir=_THIS.parents[1],
    app_name=APP_NAME,
    page_name=PAGE_NAME,
    title="📝 要約逐語録作成",
    subtitle_text="逐語録から発言の時系列を維持した要約逐語録を作成",
    default_banner_key="light_green",
)

# ============================================================
# ページ説明
# ============================================================
render_summarized_transcript_page_intro()

render_summarized_transcript_help_expander(
    theme=theme,
    banner_key=BANNER_KEY,
)

# ============================================================
# セッション初期化
# ============================================================
st.session_state.setdefault(SESSION_KEY_SOURCE, "")
st.session_state.setdefault(K_CONFIRMED_TEXT, st.session_state.get(SESSION_KEY_SOURCE, ""))
st.session_state.setdefault(K_INPUT_METHOD, INPUT_FILE)

st.session_state.setdefault(K_INBOX_BYTES, b"")
st.session_state.setdefault(K_INBOX_NAME, "")
st.session_state.setdefault(K_INBOX_KIND, "")
st.session_state.setdefault(K_INBOX_ITEM, "")
st.session_state.setdefault(K_INBOX_ADDED, "")

st.session_state.setdefault(K_MODEL_KEY, DEFAULT_TEXT_MODEL_KEY)

st.session_state.setdefault(K_MAX_UTTERANCES, 80)
st.session_state.setdefault(K_SEPARATOR_MODE, "デフォルト（全角コロン・半角コロン）")
st.session_state.setdefault(K_CUSTOM_SEPARATOR, "")

st.session_state.setdefault(K_LAST_RUN_ID, "")
st.session_state.setdefault(K_LAST_IN_TOK, None)
st.session_state.setdefault(K_LAST_OUT_TOK, None)
st.session_state.setdefault(K_LAST_COST_OBJ, None)
st.session_state.setdefault(K_LAST_MODEL, "")
st.session_state.setdefault(K_LAST_PROVIDER, "")

st.session_state.setdefault(K_SUMMARY_OUTPUT, "")
st.session_state.setdefault(K_NUMBERED_SUMMARY_OUTPUT, "")
st.session_state.setdefault(K_EXPLANATION_OUTPUT, "")
st.session_state.setdefault(K_NUMBERED_OUTPUT, "")
st.session_state.setdefault(K_CHUNK_INFO_OUTPUT, "")

st.session_state.setdefault(
    "summary_verbatim_selected_preset_keys",
    ["dearu_style"],
)
st.session_state.setdefault("summary_verbatim_preset_text", "")
st.session_state.setdefault("summary_verbatim_extra_text", "")


# ============================================================
# 補助関数：ファイル名
# ============================================================
def safe_filename(s: str) -> str:
    bad = '\\/:*?"<>|'
    for ch in bad:
        s = s.replace(ch, "_")
    return s


# ============================================================
# 補助関数：bytes decode
# ============================================================
def _decode_text_bytes(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return b.decode("cp932")
        except Exception:
            return b.decode("utf-8", errors="replace")


# ============================================================
# 補助関数：docx 読み取り
# ============================================================
def _text_from_docx_bytes(data: bytes) -> str:
    if not HAS_DOCX or Document is None:
        return ""
    try:
        doc = Document(BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""


# ============================================================
# 補助関数：usage 抽出
# ============================================================
def _extract_usage_tokens_from_result(res: Any) -> tuple[Optional[int], Optional[int]]:
    usage = getattr(res, "usage", None)
    if usage is None:
        return (None, None)

    def _get(u: Any, key: str) -> Optional[int]:
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
# 補助関数：区切り文字取得
# ============================================================
def get_separators() -> list[str]:
    mode = str(st.session_state.get(K_SEPARATOR_MODE, ""))

    if mode.startswith("デフォルト"):
        return ["：", ":"]

    custom = str(st.session_state.get(K_CUSTOM_SEPARATOR, "") or "").strip()
    if not custom:
        return ["：", ":"]

    return [custom]


# ============================================================
# 補助関数：発言者行判定
# ============================================================
def _match_speaker_line(line: str, separators: list[str]) -> tuple[str, str] | None:
    line = line.rstrip()
    if not line.strip():
        return None

    for sep in separators:
        if sep not in line:
            continue

        left, right = line.split(sep, 1)
        speaker = left.strip()
        body = right.strip()

        if not speaker:
            continue

        if len(speaker) > 80:
            continue

        if "\t" in speaker:
            continue

        return (speaker, body)

    return None


# ============================================================
# 補助関数：発言抽出（AI不使用）
# ============================================================
def parse_utterances(text: str, separators: list[str]) -> list[Utterance]:
    src = normalize_newlines(text or "")
    lines = src.splitlines()

    items: list[tuple[str, list[str]]] = []
    current_speaker: str | None = None
    current_lines: list[str] = []

    for line in lines:
        matched = _match_speaker_line(line, separators)

        if matched is not None:
            if current_speaker is not None:
                items.append((current_speaker, current_lines))

            current_speaker = matched[0]
            current_lines = [matched[1]] if matched[1] else []
            continue

        if current_speaker is not None:
            if line.strip():
                current_lines.append(line.strip())

    if current_speaker is not None:
        items.append((current_speaker, current_lines))

    utterances: list[Utterance] = []

    for idx, (speaker, body_lines) in enumerate(items, start=1):
        body = "\n".join(body_lines).strip()
        raw = f"{speaker}：{body}".strip()
        utterances.append(
            Utterance(
                no=idx,
                speaker=speaker,
                body=body,
                raw=raw,
            )
        )

    return utterances


# ============================================================
# 補助関数：番号付き逐語録
# ============================================================
def build_numbered_transcript(utterances: list[Utterance]) -> str:
    blocks: list[str] = []

    for u in utterances:
        blocks.append(
            f"【{u.no:04d}】\n"
            f"{u.speaker}：{u.body}".strip()
        )

    return "\n\n".join(blocks).strip()


# ============================================================
# 補助関数：チャンク化（AI不使用）
# ============================================================
def split_chunks(utterances: list[Utterance], max_utterances: int) -> list[Chunk]:
    if max_utterances <= 0:
        max_utterances = 80

    raw_chunks = [
        utterances[i:i + max_utterances]
        for i in range(0, len(utterances), max_utterances)
    ]

    total = len(raw_chunks)
    chunks: list[Chunk] = []

    for idx, us in enumerate(raw_chunks, start=1):
        chunks.append(
            Chunk(
                index=idx,
                total=total,
                start_no=us[0].no,
                end_no=us[-1].no,
                utterances=us,
            )
        )

    return chunks


# ============================================================
# 補助関数：チャンク情報
# ============================================================
def build_chunk_info(chunks: list[Chunk]) -> str:
    lines: list[str] = []
    lines.append(f"分割数：{len(chunks)}")
    lines.append("")

    for ch in chunks:
        lines.append(
            f"チャンク{ch.index}/{ch.total}："
            f"{ch.start_no:04d}～{ch.end_no:04d} "
            f"（発言数：{len(ch.utterances)}）"
        )

    return "\n".join(lines).strip()


# ============================================================
# 補助関数：AI出力分離
# ============================================================
def split_ai_output(text: str) -> tuple[str, str, str]:
    src = (text or "").strip()

    marker_summary = "【要約逐語録】"
    marker_numbered_summary = "【要約逐語録_発言番号付き】"
    marker_explanation = "【処理説明】"

    if (
        marker_summary in src
        and marker_numbered_summary in src
        and marker_explanation in src
    ):
        after_summary = src.split(marker_summary, 1)[1]
        summary_part, rest = after_summary.split(marker_numbered_summary, 1)
        numbered_summary_part, explanation_part = rest.split(marker_explanation, 1)

        return (
            summary_part.strip(),
            numbered_summary_part.strip(),
            explanation_part.strip(),
        )

    if marker_summary in src and marker_explanation in src:
        after_summary = src.split(marker_summary, 1)[1]
        summary_part, explanation_part = after_summary.split(marker_explanation, 1)

        return (
            summary_part.strip(),
            "",
            explanation_part.strip(),
        )

    return (
        src,
        "",
        "省略\n- 不明\n  理由：AI出力に【処理説明】区分が含まれていませんでした。\n\n統合\n- 不明\n  理由：AI出力に【処理説明】区分が含まれていませんでした。",
    )

# ============================================================
# サイドバー：モデル・分割設定
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

    st.divider()
    st.subheader("発言分割設定")

    st.number_input(
        "1ファイルの最大発言数",
        min_value=1,
        step=10,
        key=K_MAX_UTTERANCES,
        help="デフォルトは80です。発言者行を検出し、次の発言者行までを1発言として数えます。",
    )

    st.radio(
        "発言者と本文の区切り文字",
        options=[
            "デフォルト（全角コロン・半角コロン）",
            "カスタム",
        ],
        key=K_SEPARATOR_MODE,
    )

    if str(st.session_state.get(K_SEPARATOR_MODE, "")).startswith("カスタム"):
        st.text_input(
            "カスタム区切り文字",
            key=K_CUSTOM_SEPARATOR,
            placeholder="例：｜ / | / → / -",
        )



# ============================================================
# ① 逐語録入力
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
# ファイル入力
# ------------------------------------------------------------
if picked_method == INPUT_FILE:
    up = st.file_uploader(
        "逐語録（.txt / .docx）をアップロード",
        type=["txt", "docx"],
        accept_multiple_files=False,
        key=f"{PAGE_NAME}__uploader",
    )

    do_set_file = st.button(
        "セット（ファイル）",
        type="primary",
        disabled=(up is None),
        key=f"{PAGE_NAME}__btn_set_file",
    )

    text_from_file = ""

    if up is not None:
        used_file_name = up.name
        data = up.read()

        if up.name.lower().endswith(".docx"):
            text_from_file = _text_from_docx_bytes(data)
            if (not text_from_file) and (not HAS_DOCX):
                st.error("`.docx` を読むには python-docx が必要です。")
        else:
            text_from_file = _decode_text_bytes(data)

        st.caption(
            f"選択中: {used_file_name} / "
            f"length: {len((text_from_file or '').strip()):,} chars"
        )

    if do_set_file:
        if not (text_from_file or "").strip():
            st.warning("ファイルからテキストを取得できませんでした。")
            st.stop()

        st.session_state[SESSION_KEY_SOURCE] = normalize_newlines(text_from_file).strip()
        st.session_state[K_CONFIRMED_TEXT] = st.session_state[SESSION_KEY_SOURCE]
        st.session_state[f"{PAGE_NAME}__input_filename"] = used_file_name or "input.txt"
        st.success("✅ 入力テキストをセットしました。")

# ------------------------------------------------------------
# 貼り付け入力
# ------------------------------------------------------------
elif picked_method == INPUT_PASTE:
    pasted = st.text_area(
        "ここに逐語録を貼り付け",
        height=260,
        key=f"{PAGE_NAME}__pasted_text",
        placeholder="例：\n渡辺会長：では始めましょう。\n山田委員：今回の議題は何ですか。",
    )

    do_set_paste = st.button(
        "セット（貼り付け）",
        type="primary",
        key=f"{PAGE_NAME}__btn_set_paste",
    )

    if do_set_paste:
        if not (pasted or "").strip():
            st.warning("テキストを貼り付けてください。")
            st.stop()

        st.session_state[SESSION_KEY_SOURCE] = normalize_newlines(pasted).strip()
        st.session_state[K_CONFIRMED_TEXT] = st.session_state[SESSION_KEY_SOURCE]
        st.session_state[f"{PAGE_NAME}__input_filename"] = "pasted_text.txt"
        st.success("✅ 入力テキストをセットしました。")

# ------------------------------------------------------------
# Inbox入力
# ------------------------------------------------------------
else:
    st.caption("Inbox（kind=text）から読み込みます。")

    picked: InboxPickedFile | None = render_inbox_file_picker_no_toggle(
        projects_root=PROJECTS_ROOT,
        user_sub=sub,
        key_prefix=f"{PAGE_NAME}__inbox_picker",
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
        st.success("✅ Inbox から読み込みました。")

    kept_bytes: bytes = st.session_state.get(K_INBOX_BYTES, b"") or b""
    kept_name: str = st.session_state.get(K_INBOX_NAME, "") or ""

    if kept_bytes:
        st.caption(f"(保持中) name={kept_name} / size={len(kept_bytes):,} bytes")
    else:
        st.caption("(保持中) まだ選択されていません。")

    do_set_inbox = st.button(
        "セット（Inbox）",
        type="primary",
        disabled=(not bool(kept_bytes)),
        key=f"{PAGE_NAME}__btn_set_inbox",
    )

    if do_set_inbox:
        if not kept_bytes:
            st.warning("Inbox からテキストを選択してください。")
            st.stop()

        if kept_name.lower().endswith(".docx"):
            txt = _text_from_docx_bytes(kept_bytes)
        else:
            txt = _decode_text_bytes(kept_bytes)

        if not (txt or "").strip():
            st.warning("テキストが空でした。")
            st.stop()

        st.session_state[SESSION_KEY_SOURCE] = normalize_newlines(txt).strip()
        st.session_state[K_CONFIRMED_TEXT] = st.session_state[SESSION_KEY_SOURCE]
        st.session_state[f"{PAGE_NAME}__input_filename"] = kept_name or "inbox_text.txt"
        st.success("✅ 入力テキストをセットしました。")


# ============================================================
# 確定済み入力
# ============================================================
src = st.text_area(
    "確定済み逐語録（この内容を処理します）",
    height=220,
    key=K_CONFIRMED_TEXT,
)


# ============================================================
# ② 設定
# ============================================================
st.divider()
st.subheader("② 要約逐語録の設定")

group = get_group(MINUTES_MAKER)
style_group = get_group(MINUTES_STYLE)

# ------------------------------------------------------------
# スタイル調整
# ------------------------------------------------------------
st.markdown("##### スタイル調整")

st.radio(
    "横線",
    options=["横線あり", "横線なし"],
    key=f"{PAGE_NAME}__visual_mode",
)

# ------------------------------------------------------------
# 追記プリセット
# ------------------------------------------------------------
st.markdown("##### 追記プリセット")

prev_selected_keys = st.session_state.get("summary_verbatim_selected_preset_keys", [])
current_selected_keys = []

for preset in group.presets:
    checked = st.checkbox(
        preset.label,
        value=preset.key in prev_selected_keys,
        key=f"{PAGE_NAME}__preset_{preset.key}",
    )
    if checked:
        current_selected_keys.append(preset.key)

if set(current_selected_keys) != set(prev_selected_keys):
    st.session_state["summary_verbatim_selected_preset_keys"] = current_selected_keys
    combined_body_parts = [
        p.body
        for p in group.presets
        if p.key in current_selected_keys and p.body.strip()
    ]
    st.session_state["summary_verbatim_preset_text"] = "\n\n".join(combined_body_parts).strip()

# ------------------------------------------------------------
# 追加指示
# ------------------------------------------------------------
st.markdown("##### 追加指示")

st.text_area(
    "追加指示（任意）",
    height=88,
    key="summary_verbatim_extra_text",
)


# ============================================================
# ③ 事前解析
# ============================================================
st.divider()
st.subheader("③ 発言解析")

separators = get_separators()
utterances = parse_utterances(src, separators)

if src.strip():
    if utterances:
        max_utterances = int(st.session_state.get(K_MAX_UTTERANCES, 80) or 80)
        chunks = split_chunks(utterances, max_utterances)
        numbered_transcript = build_numbered_transcript(utterances)
        chunk_info = build_chunk_info(chunks)

        st.success(f"発言を {len(utterances)} 件検出しました。")
        st.text(chunk_info)

        with st.expander("発言番号付き逐語録プレビュー", expanded=False):
            st.text(numbered_transcript[:8000])

    else:
        st.warning(
            "発言者行を検出できませんでした。"
            "区切り文字が正しいか確認してください。"
        )
else:
    st.info("逐語録を入力すると、発言数と分割予定を確認できます。")


# ============================================================
# ④ AI実行
# ============================================================
st.divider()
st.subheader("④ 要約逐語録を作成")

run_btn = st.button(
    "📝 要約逐語録を作成",
    type="primary",
    key=f"{PAGE_NAME}__run",
)

if run_btn:
    #st.session_state[SESSION_KEY_SOURCE] = src

    if not src.strip():
        st.warning("逐語録を入力してください。")
        st.stop()

    separators = get_separators()
    utterances = parse_utterances(src, separators)

    if not utterances:
        st.error("発言者行を検出できませんでした。AI処理は実行しません。")
        st.stop()

    max_utterances = int(st.session_state.get(K_MAX_UTTERANCES, 80) or 80)
    chunks = split_chunks(utterances, max_utterances)

    numbered_transcript_all = build_numbered_transcript(utterances)
    chunk_info = build_chunk_info(chunks)

    st.session_state[K_NUMBERED_OUTPUT] = numbered_transcript_all
    st.session_state[K_CHUNK_INFO_OUTPUT] = chunk_info

    model_key = str(st.session_state.get(K_MODEL_KEY, DEFAULT_TEXT_MODEL_KEY))
    provider, chosen_model = _parse_model_key(model_key)

    st.session_state[K_LAST_IN_TOK] = None
    st.session_state[K_LAST_OUT_TOK] = None
    st.session_state[K_LAST_COST_OBJ] = None
    st.session_state[K_LAST_MODEL] = chosen_model
    st.session_state[K_LAST_PROVIDER] = provider

    base_preset = st.session_state.get("summary_verbatim_preset_text", "") or ""

    style_body = ""
    if style_group.presets:
        style_body = style_group.presets[0].body or ""

    if style_body:
        merged_preset = base_preset.strip() + "\n\n【見た目のスタイル指示】\n" + style_body.strip()
    else:
        merged_preset = base_preset.strip()

    extra_text = st.session_state.get("summary_verbatim_extra_text", "") or ""

    summary_blocks: list[str] = []
    numbered_summary_blocks: list[str] = []
    explanation_blocks: list[str] = []

    total_in = 0
    total_out = 0
    total_usd = 0.0
    total_jpy = 0.0

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
                "feature": "summary_verbatim",
                "action": "generate_summary_verbatim",
                "input_method": str(picked_method),
                "src_chars": len(src or ""),
                "utterances": len(utterances),
                "chunks": len(chunks),
                "max_utterances": max_utterances,
                "model_key": str(model_key),
            },
        ) as br:

            progress = st.progress(0)
            status = st.empty()

            with st.spinner("要約逐語録を作成中です..."):

                for i, ch in enumerate(chunks, start=1):

                    target_range_label = f"{ch.start_no:04d}～{ch.end_no:04d}"

                    chunk_numbered = build_numbered_transcript(
                        ch.utterances
                    )

                    prompt = build_summary_verbatim_prompt(
                        preset_body=merged_preset,
                        extra_text=extra_text,
                        target_range_label=target_range_label,
                        numbered_transcript=chunk_numbered,
                    )

                    status.info(
                        f"チャンク {ch.index}/{ch.total} "
                        f"（{ch.start_no:04d}～{ch.end_no:04d}）を処理中..."
                    )

                    res = call_text(
                        provider=str(provider),
                        model=str(chosen_model),
                        prompt=str(prompt),
                        system=None,
                        temperature=None,
                        max_output_tokens=None,
                        extra=None,
                    )

                    text = (getattr(res, "text", "") or "").strip()

                    # summary_part, explanation_part = split_ai_output(
                    #     text
                    # )
                    summary_part, numbered_summary_part, explanation_part = split_ai_output(text)

                    summary_blocks.append(
                        "\n".join(
                            [
                                "============================================================",
                                f"要約逐語録（{ch.index}/{ch.total}）",
                                f"対象発言：{ch.start_no:04d}～{ch.end_no:04d}",
                                f"発言数：{len(ch.utterances)}",
                                "============================================================",
                                "",
                                summary_part.strip(),
                            ]
                        ).strip()
                    )

                    numbered_summary_blocks.append(
                        "\n".join(
                            [
                                "============================================================",
                                f"要約逐語録_発言番号付き（{ch.index}/{ch.total}）",
                                f"対象発言：{ch.start_no:04d}～{ch.end_no:04d}",
                                f"発言数：{len(ch.utterances)}",
                                "============================================================",
                                "",
                                numbered_summary_part.strip(),
                            ]
                        ).strip()
                    )

                    explanation_blocks.append(
                        "\n".join(
                            [
                                "============================================================",
                                f"チャンク{ch.index}（{ch.start_no:04d}～{ch.end_no:04d}）",
                                "============================================================",
                                "",
                                explanation_part.strip(),
                            ]
                        ).strip()
                    )

                    in_tok, out_tok = _extract_usage_tokens_from_result(
                        res
                    )

                    if isinstance(in_tok, int):
                        total_in += in_tok

                    if isinstance(out_tok, int):
                        total_out += out_tok

                    cost_obj = getattr(res, "cost", None)

                    usd = getattr(cost_obj, "usd", None) if cost_obj is not None else None
                    jpy = getattr(cost_obj, "jpy", None) if cost_obj is not None else None

                    if isinstance(usd, (int, float)):
                        total_usd += float(usd)

                    if isinstance(jpy, (int, float)):
                        total_jpy += float(jpy)

                    progress.progress(i / len(chunks))

            if total_in > 0 or total_out > 0:
                st.session_state[K_LAST_IN_TOK] = total_in
                st.session_state[K_LAST_OUT_TOK] = total_out
                br.set_usage(
                    int(total_in),
                    int(total_out),
                )

            if total_usd > 0 or total_jpy > 0:
                st.session_state[K_LAST_COST_OBJ] = SimpleNamespace(
                    usd=total_usd,
                    jpy=total_jpy,
                )
                br.set_cost(
                    float(total_usd),
                    float(total_jpy),
                )

            br.add_finish_meta(note="ok")

            st.session_state[K_LAST_RUN_ID] = br.run_id

    except Exception as e:
        st.error(f"AI 呼び出しでエラー: {e}")
        st.stop()

    final_summary = "\n\n\n".join(summary_blocks).strip()
    final_numbered_summary = "\n\n\n".join(numbered_summary_blocks).strip()
    final_explanation = "\n\n\n".join(explanation_blocks).strip()

    visual_mode = st.session_state.get(f"{PAGE_NAME}__visual_mode", "横線あり")
    if visual_mode == "横線なし":
        final_summary = "\n".join(
            line for line in final_summary.splitlines()
            if line.strip() != "============================================================"
        )
    if visual_mode == "横線なし":
        final_numbered_summary = "\n".join(
            line for line in final_numbered_summary.splitlines()
            if line.strip() != "============================================================"
        )

    st.session_state[K_SUMMARY_OUTPUT] = final_summary
    st.session_state[K_NUMBERED_SUMMARY_OUTPUT] = final_numbered_summary
    st.session_state[K_EXPLANATION_OUTPUT] = final_explanation

    st.success("要約逐語録の作成が完了しました。")


# ============================================================
# ⑤ 結果表示・ダウンロード
# ============================================================
summary_output = (st.session_state.get(K_SUMMARY_OUTPUT) or "").strip()
numbered_summary_output = (st.session_state.get(K_NUMBERED_SUMMARY_OUTPUT) or "").strip()
explanation_output = (st.session_state.get(K_EXPLANATION_OUTPUT) or "").strip()
numbered_output = (st.session_state.get(K_NUMBERED_OUTPUT) or "").strip()
chunk_info_output = (st.session_state.get(K_CHUNK_INFO_OUTPUT) or "").strip()

if summary_output:
    st.divider()
    st.subheader("⑤ 生成結果")

    st.markdown("### 要約逐語録")
    st.text_area(
        "要約逐語録",
        value=summary_output,
        height=420,
        key=f"{PAGE_NAME}__summary_preview",
    )

    with st.expander("要約逐語録_発言番号付き", expanded=False):
        st.text(numbered_summary_output)

    with st.expander("要約処理説明", expanded=False):
        st.text(explanation_output)

    with st.expander("発言番号付き逐語録", expanded=False):
        st.text(numbered_output[:15000])

    st.subheader("⑥ ダウンロード")

    input_name = st.session_state.get(f"{PAGE_NAME}__input_filename", "")
    input_stem = safe_filename(Path(input_name).stem) if input_name else "summary_verbatim"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    st.download_button(
        label="💾 要約逐語録を保存 (.txt)",
        data=(summary_output.strip() + "\n").encode("utf-8"),
        file_name=f"{input_stem}_要約逐語録_{timestamp}.txt",
        mime="text/plain",
        key=f"{PAGE_NAME}__dl_summary",
    )

    st.download_button(
        label="💾 要約逐語録_発言番号付きを保存 (.txt)",
        data=(numbered_summary_output.strip() + "\n").encode("utf-8"),
        file_name=f"{input_stem}_要約逐語録_発言番号付き_{timestamp}.txt",
        mime="text/plain",
        key=f"{PAGE_NAME}__dl_numbered_summary",
    )

    st.download_button(
        label="💾 要約処理説明を保存 (.txt)",
        data=(explanation_output.strip() + "\n").encode("utf-8"),
        file_name=f"{input_stem}_要約処理説明_{timestamp}.txt",
        mime="text/plain",
        key=f"{PAGE_NAME}__dl_explanation",
    )

    st.download_button(
        label="💾 発言番号付き逐語録を保存 (.txt)",
        data=(numbered_output.strip() + "\n").encode("utf-8"),
        file_name=f"{input_stem}_発言番号付き逐語録_{timestamp}.txt",
        mime="text/plain",
        key=f"{PAGE_NAME}__dl_numbered",
    )

    if chunk_info_output:
        st.download_button(
            label="💾 分割情報を保存 (.txt)",
            data=(chunk_info_output.strip() + "\n").encode("utf-8"),
            file_name=f"{input_stem}_分割情報_{timestamp}.txt",
            mime="text/plain",
            key=f"{PAGE_NAME}__dl_chunk_info",
        )

else:
    st.info("逐語録を入力して『📝 要約逐語録を作成』を実行してください。")


# ============================================================
# 実行サマリ
# ============================================================
last_run_id = str(st.session_state.get(K_LAST_RUN_ID) or "").strip()

if last_run_id:
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