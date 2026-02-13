# -*- coding: utf-8 -*-
# minutes_app/pages/80_議事録ポータル.py
# ============================================================
# 🗂️ 議事録ポータル（テスト用）
#
# 目的：
# - Minutes Maker の「入口」を1ページに集約して見やすくする
# - 推奨フロー（Step 1〜4）をカードで提示し、Pages内の各機能へ誘導する
#
# 運用方針（康男さんルール準拠）：
# - st.form は使わない
# - use_container_width は使わない
# - st.button()/st.download_button() に width 引数は使わない
# ============================================================

from __future__ import annotations

# ============================================================
# imports（stdlib）
# ============================================================
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import sys


# ============================================================
# imports（third-party）
# ============================================================
import streamlit as st

# ============================================================
# paths（minutes_app: pages 共通）
# - projects_root を確定し、common_lib を import できるようにする
# ============================================================
_THIS = Path(__file__).resolve()
APP_ROOT = _THIS.parents[1]              # .../minutes_app
APP_NAME = APP_ROOT.name                 # "minutes_app"
PROJECTS_ROOT = _THIS.parents[3]         # .../projects

if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

# ============================================================
# imports（common_lib）
# ============================================================
from common_lib.ui.ui_basics import subtitle
from common_lib.ui.banner_lines import render_banner_line_by_key
from common_lib.sessions.page_entry import page_session_heartbeat


# ============================================================
# constants
# ============================================================
PAGE_NAME = "80_minutes_portal"


# ============================================================
# data models
# ============================================================
@dataclass(frozen=True)
class PortalCard:
    step: str
    title: str
    body: str
    hint: str
    page_path: Optional[str] = None  # 例: "pages/20_音声ファイル分割_ストレージ対応.py"
    page_label: Optional[str] = None  # 例: "▶ 音声分割へ"
    badge: Optional[str] = None  # 例: "推奨"


# ============================================================
# ui helpers
# ============================================================
def _render_login_badge(*, sub: str) -> None:
    """
    右上のログイン表示（sub を正本として表示）
    """
    left, right = st.columns([2, 1])
    with left:
        st.title("🎛️ Minutes Maker")
        subtitle("議事録作成アプリ")
    with right:
        st.success(f"✅ ログイン中: **{sub}**")




def _render_hero() -> None:
    """
    ページ冒頭の説明（短く、導線重視）
    """
    st.markdown("### はじめに（おすすめ手順）")
    st.write(
        "音声をアップロードして分割 → 文字起こし → 話者分離 → 重複検出で整形し、"
        "最後に「正本（確定版）」から用途別の議事録を生成します。"
    )
    st.info("このアプリは **Human in the Loop**（AI＋人手確認）前提です。最終成果物は必ず目視で確認してください。")


def _render_page_link(path: str, label: str) -> None:
    """
    page_link:
    - main（app.py）から見た相対パスで指定する
    - 実在しないと StreamlitAPIException で落ちる
    → 存在確認して、無ければ案内にフォールバックする
    """
    # pages/ 配下に実在するかチェック（minutes_app が cwd 前提）
    p = Path(path)
    if not p.exists():
        st.caption("※ ページリンク先が見つかりません。左の Pages メニューから対象ページを開いてください。")
        st.code(path, language="text")
        return

    # page_link が使える環境ならリンクを出す
    if hasattr(st, "page_link"):
        st.page_link(path, label=label)
        return

    # page_link が無い場合も落とさない
    st.caption("※ この環境ではページリンクが利用できません。左の Pages メニューから対象ページを開いてください。")
    st.code(path, language="text")



def _render_card(card: PortalCard) -> None:
    """
    ステップカード（余白と階層だけで“それっぽく”見せる）
    """
    with st.container(border=True):
        top_l, top_r = st.columns([8, 2], vertical_alignment="center")
        with top_l:
            st.markdown(f"**{card.step}  {card.title}**")
        with top_r:
            if card.badge:
                st.caption(f"🟩 {card.badge}")

        st.write(card.body)
        st.caption(card.hint)

        if card.page_path and card.page_label:
            _render_page_link(card.page_path, card.page_label)


def _render_sections() -> None:
    """
    推奨フロー（4ステップ）
    ※ page_path は、実ファイル名に合わせて適宜変更してください。
    """
    st.markdown("### 使い方（推奨フロー）")

    cards = [
        PortalCard(
            step="①",
            title="音声をアップロード → 自動で分割・保存",
            body="長時間の録音は処理しやすい単位に分割し、ストレージに保存します。以降は保存済み成果物を入力として進めます。",
            hint="次に押す場所：Pages → 音声分割",
            page_path="pages/20_音声ファイル分割_ストレージ対応.py",
            page_label="▶ 音声分割へ",
            badge="最初にここ",
        ),
        PortalCard(
            step="②",
            title="文字起こし（分割音声からテキスト化）",
            body="分割された音声を順に文字起こしして、ストレージへ保存します。",
            hint="次に押す場所：Pages → 文字起こし",
            page_path="pages/30_文字起こし_ストレージ対応.py",
            page_label="▶ 文字起こしへ",
        ),
        PortalCard(
            step="③",
            title="話者分離 → 重複箇所検出（整形）",
            body="話者名付与や重複・不整合の検出を行い、確認・修正しやすい状態にします。",
            hint="次に押す場所：Pages → 話者分離 → 重複箇所検出",
            page_path="pages/23_重複箇所検出_storage対応.py",
            page_label="▶ 重複箇所検出へ",
        ),
        PortalCard(
            step="④",
            title="正本（確定版）から議事録を生成",
            body="確認・修正したテキスト（正本）を入力し、用途別テンプレ（要点整理／決定事項／アクション等）の議事録を作成します。",
            hint="次に押す場所：Pages → 議事録作成",
            page_path="pages/40_議事録作成_ストレージ対応.py",
            page_label="▶ 議事録作成へ",
            badge="ゴール",
        ),
    ]

    for c in cards:
        _render_card(c)


def _render_quick_notes() -> None:
    """
    注意事項（短く）
    """
    st.markdown("### 注意事項")
    st.write(
        "- 途中で話題が切り替わって戻る会議は、時系列が崩れやすいので必ず目視確認してください。\n"
        "- 誤変換や話者名の誤りは、重複箇所検出後に修正して「正本」を確定してください。"
    )


# ============================================================
# page
# ============================================================
def main() -> None:
    # ------------------------------------------------------------
    # page config（最初に1回だけ）
    # ------------------------------------------------------------
    st.set_page_config(
        page_title="🗂️ 議事録ポータル",
        page_icon="🗂️",
        layout="wide",
    )

    # ============================================================
    # バナー（アプリ共通・最上段）
    # ============================================================
    render_banner_line_by_key("light_green")

    # ============================================================
    # ログイン + セッション heartbeat（pages 共通）
    # ============================================================
    sub = page_session_heartbeat(
        st,
        PROJECTS_ROOT,
        app_name=APP_NAME,
        page_name=PAGE_NAME,
    )

    # ------------------------------------------------------------
    # header
    # ------------------------------------------------------------
    _render_login_badge(sub=str(sub))
    st.divider()

    # ------------------------------------------------------------
    # hero + main sections
    # ------------------------------------------------------------
    _render_hero()
    _render_sections()

    # ------------------------------------------------------------
    # notes
    # ------------------------------------------------------------
    st.divider()
    _render_quick_notes()


# ============================================================
# entrypoint
# ============================================================
if __name__ == "__main__":
    main()
