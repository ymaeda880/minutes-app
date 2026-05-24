# app.py
from __future__ import annotations

# ============================================================
# imports
# ============================================================
import sys
from pathlib import Path

import streamlit as st

from config.config import DEFAULT_USDJPY, get_openai_api_key
from ui.sidebar import init_metrics_state
from ui.style import hide_anchor_links


# ============================================================
# パスの取得と common_lib 読み込み
# ============================================================
_THIS = Path(__file__).resolve()
APP_ROOT = _THIS.parent
APP_NAME = APP_ROOT.name
PROJECTS_ROOT = _THIS.parents[2]

if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))


# ============================================================
# common_lib imports
# ============================================================
from common_lib.sessions.app_entry import app_session_heartbeat
from common_lib.ui.banner_lines import render_banner_line_by_key
from common_lib.ui.intro_panel import (
    render_hero_panel,
    render_intro_css,
    render_two_column_cards,
)
from common_lib.ui.theme_colors import get_theme_colors_from_banner_key
from common_lib.ui.ui_basics import subtitle


# ============================================================
# page config
# ============================================================
st.set_page_config(
    page_title="Minutes Maker",
    layout="wide",
)


# ============================================================
# theme / banner
# ============================================================
BANNER_KEY = "light_green"

render_banner_line_by_key(BANNER_KEY)

theme = get_theme_colors_from_banner_key(BANNER_KEY)
render_intro_css(theme)


# ============================================================
# custom expander css
# ============================================================
expander_border = theme.get("border", "rgba(34, 197, 94, 0.32)")
expander_accent = theme.get("accent", "#15803d")
expander_bg = theme.get("bg", "#f7fff9")

st.markdown(
    f"""
    <style>

    /* =======================================================
       custom usage expander only
    ======================================================= */

    .usage-expander div[data-testid="stExpander"] {{
        border: 1px solid {expander_border};
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #ffffff 0%,
            {expander_bg} 100%
        );
        box-shadow: 0 10px 28px rgba(15, 118, 110, 0.10);
        overflow: hidden;
    }}

    .usage-expander div[data-testid="stExpander"] details {{
        border-radius: 18px;
    }}

    .usage-expander div[data-testid="stExpander"] summary {{
        padding: 18px 22px !important;
    }}

    .usage-expander div[data-testid="stExpander"] summary p {{
        font-size: 1.05rem !important;
        font-weight: 800 !important;
        color: {expander_accent} !important;
        letter-spacing: 0.02em;
    }}

    .usage-expander div[data-testid="stExpander"] summary:hover {{
        background: rgba(34, 197, 94, 0.06);
    }}

    .usage-expander div[data-testid="stExpanderDetails"] {{
        padding: 0 24px 22px 24px;
        border-top: 1px solid {expander_border};
        background: rgba(255,255,255,0.60);
    }}

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 鎖アイコンを非表示
# ============================================================
# hide_anchor_links()


# ============================================================
# ログイン + セッション heartbeat（app 共通）
# ============================================================
sub = app_session_heartbeat(
    st,
    PROJECTS_ROOT,
    app_name=APP_NAME,
)

user = sub


# ============================================================
# ヘッダ
# ============================================================
left, right = st.columns([2, 1])

with left:
    st.title("🎛️ Minutes Maker")

with right:
    st.success(f"✅ ログイン中: **{sub}**")

subtitle("議事録作成アプリ")


# ============================================================
# 初期化
# ============================================================
init_metrics_state()

if "usd_jpy" not in st.session_state:
    st.session_state["usd_jpy"] = DEFAULT_USDJPY


# ============================================================
# API キー確認
# ============================================================
OPENAI_API_KEY = get_openai_api_key()

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY が .streamlit/secrets.toml に設定されていません。")


# ============================================================
# short description
# ============================================================
st.caption("Transcribe・Separate speakers・Clean transcript・Generate minutes.")

st.markdown(
    """
    左サイドバーまたは Pages メニューから、利用したい機能を選択してください。  
    まずは **文字起こし** から順に処理を進めてください。
    """
)

# # ============================================================
# # usage expander
# # ============================================================
# st.markdown(
#     '<div style="height:16px;"></div>',
#     unsafe_allow_html=True,
# )
# with st.expander("使い方（推奨フロー）", expanded=False):
#     st.markdown(
#         """
# 1. **音声ファイルをアップロードし、分割します。**  
#    長時間の録音は処理しやすい単位に分割され、ストレージに保存されます。

# 2. **文字起こし → 話者分離 → 重複箇所検出** の順に処理を進めます。  
#    各ステップの結果は自動的に保存され、次の工程に引き継がれます。

# 3. **重複箇所検出後のテキストをダウンロードし、人手で確認・修正し、逐語録を作成します。**  
#    実際に音声を聞きながら、文字起こしの変換ミス、話者の誤り、文脈上不自然な箇所を修正します。

# 4. **「議事録作成」** に逐語録を入力し、議事録を作成します。  
#    確定した逐語録を基に、用途に応じた複数タイプの議事録を作成できます。
# """
#     )

# ============================================================
# usage expander
# ============================================================
st.markdown(
    '<div style="height:16px;"></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="usage-expander">',
    unsafe_allow_html=True,
)

with st.expander(
    "💡 使い方（推奨フロー）",
    expanded=False,
):
    st.markdown(
        """
1. **音声ファイルをアップロードし、分割します。**  
   長時間の録音は処理しやすい単位に分割され、ストレージに保存されます。

2. **文字起こし → 話者分離 → 重複箇所検出** の順に処理を進めます。  
   各ステップの結果は自動的に保存され、次の工程に引き継がれます。

3. **重複箇所検出後のテキストをダウンロードし、人手で確認・修正し、逐語録を作成します。**  
   実際に音声を聞きながら、文字起こしの変換ミス、話者の誤り、文脈上不自然な箇所を修正します。

4. **「議事録作成」** に逐語録を入力し、議事録を作成します。  
   確定した逐語録を基に、用途に応じた複数タイプの議事録を作成できます。
"""
    )

st.markdown(
    '</div>',
    unsafe_allow_html=True,
)

# ============================================================
# intro message
# ============================================================
render_hero_panel(
    kicker="MINUTES MAKER",
    title="音声から、確認済み逐語録へ。逐語録から、使える議事録へ。",
    body_html='<span class="ts-highlight">Minutes Maker</span> は、音声分割・文字起こし・話者分離・重複箇所検出・議事録作成を一連の流れで扱うためのAI議事録作成ワークスペースです。<br>AIによる自動処理と人手による確認を組み合わせた <span class="ts-highlight">Human in the Loop</span> 前提の運用を想定しています。',
    chips=[
        "Transcribe",
        "Speaker",
        "Clean",
        "Minutes",
    ],
)


# ============================================================
# flow cards
# ============================================================
render_two_column_cards(
    left_title="① 音声処理から逐語録作成まで",
    left_body_html="音声ファイルをアップロードして分割し、<span class=\"ts-highlight\">文字起こし → 話者分離 → 重複箇所検出</span> の順に処理します。<br><br>各ステップの成果物はストレージに保存され、次の工程に引き継がれます。",
    right_title="② 人手確認後に議事録を作成",
    right_body_html="重複箇所検出後のテキストをダウンロードし、音声を聞きながら誤変換・話者名・文脈を確認します。<br><br>確認済みの <span class=\"ts-highlight\">逐語録（正本）</span> を入力として、議事録を作成します。",
)


# ============================================================
# development cards
# ============================================================
#st.divider()
st.markdown(
    '<div style="height:24px;"></div>',
    unsafe_allow_html=True,
)

render_two_column_cards(
    left_title="🚧 現在も開発中です",
    left_body_html="本アプリケーションは、業務効率を高めることを目的として継続的に改良しています。<br><br>議事録の形式や使い勝手について、気づいた点・改善要望・不具合がありましたら、ぜひフィードバックをお寄せください。",
    right_title="📝 既存サービスとの併用も可能です",
    right_body_html="当面の運用として、AI議事録・Notta など既存サービスで作成した逐語録を取り込み、<span class=\"ts-highlight\">Minutes Maker の議事録生成機能</span> のみを利用する方法も有効です。<br><br>今後は、複数の議事録フォーマット選択や匿名化処理にも対応していく予定です。",
)


# ============================================================
# サイドバー
# ============================================================
# render_sidebar()