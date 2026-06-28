# -*- coding: utf-8 -*-
# minutes_app/pages/15_議事録ヘッダー作成.py
# ============================================================
# 議事録ヘッダー作成
# - xlsx / csv を読み込む
# - 議事録ヘッダー用の Word を作成する
# - Word ダウンロード / Inbox 保存に対応
# ============================================================

from __future__ import annotations

# ============================================================
# imports
# ============================================================
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import streamlit as st


# ============================================================
# パス設定
# ============================================================
_THIS = Path(__file__).resolve()
PROJECTS_ROOT = _THIS.parents[3]

if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

APP_DIR = _THIS.parents[1]
APP_NAME = APP_DIR.name
PAGE_NAME = _THIS.stem


# ============================================================
# ページ設定
# ============================================================
st.set_page_config(
    page_title="Minutes Maker / 議事録ヘッダー作成",
    page_icon="📄",
    layout="wide",
)


# ============================================================
# common_lib
# ============================================================
from common_lib.ui.page_header import render_standard_page_header
from common_lib.ui.input_source import render_input_source

from common_lib.inbox.inbox_ops.ingest import ingest_to_inbox
from common_lib.inbox.inbox_common.types import (
    IngestRequest,
    InboxNotAvailable,
    QuotaExceeded,
    IngestFailed,
)


# ============================================================
# app lib
# ============================================================
from lib.header_generation.explanation import (
    render_header_generation_page_intro,
    render_header_generation_help_expander,
)
from lib.header_generation.excel_parser import (
    parse_csv_blocks,
    parse_excel_blocks,
)
from lib.header_generation.docx_builder import build_docx_bytes


# ============================================================
# 共通ヘッダー
# ============================================================
sub, theme, BANNER_KEY, settings = render_standard_page_header(
    st_module=st,
    projects_root=PROJECTS_ROOT,
    app_dir=APP_DIR,
    app_name=APP_NAME,
    page_name=PAGE_NAME,
    title="📄 議事録ヘッダー作成",
    subtitle_text="Excel / CSV から議事録ヘッダー用 Word を作成",
    default_banner_key="light_green",
)


# ============================================================
# ページ説明
# ============================================================
render_header_generation_page_intro()

render_header_generation_help_expander(
    theme=theme,
    banner_key=BANNER_KEY,
)


# ============================================================
# セッションキー
# ============================================================
K_PARSED_BLOCKS = f"{PAGE_NAME}__parsed_blocks"
K_DOCX_BYTES = f"{PAGE_NAME}__docx_bytes"
K_DOCX_NAME = f"{PAGE_NAME}__docx_name"
K_INPUT_SIG = f"{PAGE_NAME}__input_sig"
K_SOURCE_NAME = f"{PAGE_NAME}__source_name"

st.session_state.setdefault(K_PARSED_BLOCKS, [])
st.session_state.setdefault(K_DOCX_BYTES, b"")
st.session_state.setdefault(K_DOCX_NAME, "")
st.session_state.setdefault(K_INPUT_SIG, "")
st.session_state.setdefault(K_SOURCE_NAME, "")


# ============================================================
# helper
# ============================================================
def safe_filename(s: str) -> str:
    """ファイル名に使えない文字を置換する。"""

    bad = '\\/:*?"<>|'
    for ch in bad:
        s = s.replace(ch, "_")

    return s.strip() or "議事録ヘッダー"


def parse_blocks_from_input(
    *,
    data_bytes: bytes,
    file_name: str,
    pasted_text: str,
    source_type: str,
) -> list[dict[str, Any]]:
    """入力ソースから表ブロックを抽出する。"""

    lower_name = str(file_name or "").lower()

    # ------------------------------------------------------------
    # 貼り付けの場合
    # - CSVテキストとして扱う
    # ------------------------------------------------------------
    if source_type == "paste":
        text = str(pasted_text or "").strip()
        if not text:
            return []

        return parse_csv_blocks(
            text.encode("utf-8")
        )

    # ------------------------------------------------------------
    # upload / inbox
    # ------------------------------------------------------------
    if not data_bytes:
        return []

    if lower_name.endswith(".xlsx"):
        return parse_excel_blocks(data_bytes)

    if lower_name.endswith(".csv"):
        return parse_csv_blocks(data_bytes)

    raise RuntimeError(
        "対応している形式は .xlsx / .csv です。"
    )


def clear_generated_docx() -> None:
    """入力変更時に生成済み Word をクリアする。"""

    st.session_state[K_DOCX_BYTES] = b""
    st.session_state[K_DOCX_NAME] = ""


# ============================================================
# ① 入力
# ============================================================
st.divider()
st.subheader("① 入力ファイルの設定")

input_result = render_input_source(
    projects_root=PROJECTS_ROOT,
    user_sub=sub,
    page_name=PAGE_NAME,
    key_prefix=f"{PAGE_NAME}__header_input",
    allowed_sources=["paste", "upload", "inbox"],
    upload_types=["xlsx", "csv"],
    inbox_kinds=None,
    inbox_extensions=["xlsx", "csv"],
    input_label="入力方法",
    paste_label="CSV形式の表データを貼り付け",
    upload_label="Excel / CSV ファイルをアップロード",
    inbox_page_size=8,
)

if not input_result.confirmed:
    st.info("まず Excel / CSV を設定してください。")
    st.stop()

# ============================================================
# 入力確定時：表ブロック解析
# ============================================================
input_sig = (
    f"{input_result.source_type}|"
    f"{input_result.file_name}|"
    f"{len(input_result.data_bytes or b'')}|"
    f"{len(input_result.text or '')}"
)

current_blocks = st.session_state.get(K_PARSED_BLOCKS, [])

# 同じファイルでも，表ブロックが未作成なら再解析する
need_parse = (
    st.session_state.get(K_INPUT_SIG, "") != input_sig
    or not current_blocks
)

if need_parse:
    try:
        blocks = parse_blocks_from_input(
            data_bytes=input_result.data_bytes or b"",
            file_name=input_result.file_name or "",
            pasted_text=input_result.text or "",
            source_type=input_result.source_type,
        )
    except Exception as e:
        st.session_state[K_PARSED_BLOCKS] = []
        clear_generated_docx()
        st.error(f"入力データの読み込みに失敗しました: {e}")
        st.stop()

    st.session_state[K_PARSED_BLOCKS] = blocks
    st.session_state[K_INPUT_SIG] = input_sig
    st.session_state[K_SOURCE_NAME] = (
        input_result.file_name
        or "pasted_header.csv"
    )

    clear_generated_docx()

    if not blocks:
        st.warning(
            "表ブロックを検出できませんでした。"
            "見出し行と表の形式を確認してください。"
        )
    else:
        st.success(
            f"{len(blocks)} 件の表ブロックを検出しました。"
        )

# ============================================================
# ② プレビュー
# ============================================================
blocks = st.session_state.get(K_PARSED_BLOCKS, [])

if blocks:
    st.divider()
    st.subheader("② 読み込んだ表")

    for idx, block in enumerate(blocks, start=1):
        title = str(block.get("title", ""))
        header = list(block.get("header", []))
        data = list(block.get("data", []))

        st.markdown(f"##### {idx}. {title}")

        try:
            preview_df = pd.DataFrame(
                data,
                columns=header,
            )
            st.dataframe(
                preview_df,
                hide_index=True,
            )
        except Exception:
            st.write(
                {
                    "title": title,
                    "header": header,
                    "data": data,
                }
            )


# ============================================================
# ③ Word 作成
# ============================================================
if blocks:
    st.divider()
    st.subheader("③ Wordを作成")

    if st.button(
        "Wordを作成",
        type="primary",
        key=f"{PAGE_NAME}__make_docx",
    ):
        try:
            docx_bytes = build_docx_bytes(blocks)

            source_name = str(
                st.session_state.get(K_SOURCE_NAME)
                or "議事録ヘッダー"
            )
            source_stem = safe_filename(
                Path(source_name).stem
            )

            st.session_state[K_DOCX_BYTES] = docx_bytes
            st.session_state[K_DOCX_NAME] = (
                f"{source_stem}_議事録ヘッダー.docx"
            )

            st.success("Wordを作成しました。")

        except Exception as e:
            st.error(f"Word作成に失敗しました: {e}")
            st.stop()


# ============================================================
# ④ ダウンロード / Inbox保存
# ============================================================
docx_bytes = st.session_state.get(K_DOCX_BYTES, b"")
docx_name = st.session_state.get(K_DOCX_NAME, "")

if docx_bytes:
    st.divider()
    st.subheader("④ Wordの保存")

    col_dl, col_inbox = st.columns(2)

    with col_dl:
        st.download_button(
            "💾 Wordをダウンロード",
            data=docx_bytes,
            file_name=docx_name or "議事録ヘッダー.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            key=f"{PAGE_NAME}__download_docx",
            on_click="ignore",
        )

    with col_inbox:
        if st.button(
            "📥 WordをInboxへ保存",
            type="secondary",
            key=f"{PAGE_NAME}__save_docx_to_inbox",
        ):
            try:
                if not docx_bytes:
                    st.error(
                        "❌ Wordが生成されていないため保存できません。"
                    )
                    st.stop()

                ingest_to_inbox(
                    projects_root=PROJECTS_ROOT,
                    req=IngestRequest(
                        user_sub=sub,
                        filename=docx_name or "議事録ヘッダー.docx",
                        data=docx_bytes,
                        tags_json='["minutes/header_generation/word"]',
                        origin={
                            "app": APP_NAME,
                            "page": PAGE_NAME,
                            "source_filename": str(
                                st.session_state.get(K_SOURCE_NAME)
                                or ""
                            ),
                            "action": "minutes_header_generation_word",
                        },
                    ),
                )

                st.success("WordをInboxに保存しました。")

            except InboxNotAvailable:
                st.error(
                    "❌ Inbox が存在しません。ストレージ接続を確認してください。"
                )

            except QuotaExceeded as e:
                st.error(
                    f"❌ 容量オーバーです。"
                    f" 現在={e.current} / 追加={e.incoming} / 上限={e.quota}"
                )

            except IngestFailed as e:
                st.error(
                    f"❌ Inbox への保存に失敗しました: {e}"
                )