# minutes_app/app.py
from __future__ import annotations

# ============================================================
# imports
# ============================================================
import re
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

PAGE_NAME = _THIS.stem

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
from common_lib.ui.expander_style import render_theme_expander_css

# ============================================================
# page config
# ============================================================
st.set_page_config(
    page_title="Minutes Maker",
    layout="wide",
)


# ============================================================
# page key
# ============================================================
SAFE_PAGE_KEY = re.sub(r"[^0-9a-zA-Z_]+", "_", PAGE_NAME)
K_EXPANDER_THEME_AREA = f"{SAFE_PAGE_KEY}_expander_theme_area"

# ============================================================
# theme / banner
# ============================================================
BANNER_KEY = "light_green"
render_banner_line_by_key(BANNER_KEY)
theme = get_theme_colors_from_banner_key(BANNER_KEY)

# ============================================================
# theme  css
# ============================================================
render_theme_expander_css(
    container_key=K_EXPANDER_THEME_AREA,
    theme=theme,
)
render_intro_css(theme)

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

# ============================================================
# usage expander（theme expander）
# ============================================================
theme_area = st.container(
    key=K_EXPANDER_THEME_AREA,
)

with theme_area:
    with st.expander(
        "💡 Minutes Maker の使い方と使用上の注意",
        expanded=False,
    ):
        st.markdown(
            """
### 1. 使い方（推奨フロー）

1. **音声ファイルをアップロードし、分割します。**  
   長時間の録音は処理しやすい単位に分割され、ストレージに保存されます。

2. **文字起こし → 話者分離 → 重複箇所検出** の順に処理を進めます。  
   各ステップの結果は自動的に保存され、次の工程に引き継がれます。

3. **重複箇所検出後のテキストをダウンロードし、人手で確認・修正し、逐語録を作成します。**  
   実際に音声を聞きながら、文字起こしの変換ミス、話者の誤り、文脈上不自然な箇所を修正します。

4. **「議事録作成」** に逐語録を入力し、議事録を作成します。  
   確定した逐語録を基に、用途に応じた複数タイプの議事録を作成できます。

### 2. 使用上の注意

- AIの使用に際しては、個人情報や機密情報の取り扱いに注意してください。
    1. 文字起こしには、文字起こし用のAIを使用します。
    2. 話者分離には、生成AIを使用します。
    3. 議事録作成には、生成AIを使用します。
"""
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