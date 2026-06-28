# minutes_app/app.py
from __future__ import annotations


# ============================================================
# imports
# ============================================================
import streamlit as st

# ============================================================
# パス設定（app.py 用）
# ============================================================
from pathlib import Path
import sys

_THIS = Path(__file__).resolve()

APP_DIR = _THIS.parent
PROJ_DIR = _THIS.parents[1]
MONO_ROOT = _THIS.parents[2]

for p in (MONO_ROOT, PROJ_DIR, APP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

PROJECTS_ROOT = MONO_ROOT
APP_NAME = APP_DIR.name
PAGE_NAME = _THIS.stem

# ============================================================
# navigation icons
# ============================================================
from common_lib.ui.nav_icons import (
    NAV_HOME_ICON, 
    NAV_PORTAL_RETURN_ICON,
    NAV_PROCESS_ICON, 
    NAV_CONSTRUCTION_ICON,
    PAGE_HOME_ICON,
    PAGE_PORTAL_RETURN_ICON,
    NAV_STOP_ICON,
)


# ============================================================
# page config
# ============================================================
st.set_page_config(
    page_title="Minutes Maker",
    layout="wide",
)


#BANNER_KEY = "light_green"

# ============================================================
# navigation
# ============================================================
pg = st.navigation(
    {
        f"{NAV_HOME_ICON}": [
            st.Page(
                "pages/00_トップ.py",
                title="トップ",
                icon=PAGE_HOME_ICON,
                default=True,
            ),
        ],
        f"{NAV_PROCESS_ICON} 議事録": [
            st.Page(
                "pages/07_議事録作成.py",
                title="議事録作成",
                icon="📝",
            ),
            st.Page(
                "pages/08_要約逐語録作成.py",
                title="要約逐語録作成",
                icon="📝",
            ),

            st.Page(
                "pages/15_議事録ヘッダー作成.py",
                title="議事録ヘッダー作成",
                icon="📄",
            ),
        ],
        f"{NAV_PROCESS_ICON} 逐語録までの処理": [
            st.Page(
                "pages/20_音声ファイル分割_ストレージ対応.py",
                title="音声ファイル分割",
                icon="🎧",
            ),
            st.Page(
                "pages/21_文字起こし_ストレージ対応.py",
                title="文字起こし",
                icon="✍️",
            ),
            st.Page(
                "pages/22_話者分離_ストレージ対応.py",
                title="話者分離",
                icon="👥",
            ),
            st.Page(
                "pages/30_重複箇所検出_ストレージ対応.py",
                title="重複箇所検出",
                icon="🔎",
            ),
            st.Page(
                "pages/35_音声ファイル一括処理.py",
                title="音声ファイル一括処理",
                icon="🔄",
            ),
            # st.Page(
            #     "pages/36_音声ファイル一括処理OLD.py",
            #     title="音声ファイル一括処理（古い）",
            #     icon="🔄",
            # ),

        ],
        "🕰️ 旧ページ": [
            st.Page(
                "pages/61_音声ファイル分割.py",
                title="音声ファイル分割（旧）",
                icon="🎧",
            ),
            st.Page(
                "pages/63_文字起こし（連続対応）.py",
                title="文字起こし 連続対応",
                icon="✍️",
            ),
            st.Page(
                "pages/65_重複箇所検出.py",
                title="重複箇所検出（旧）",
                icon="🔎",
            ),
            st.Page(
                "pages/67_話者分離.py",
                title="話者分離（旧）",
                icon="👥",
            ),
            # st.Page(
            #     "pages/68_話者分離（Gemini）.py",
            #     title="話者分離 Gemini",
            #     icon="✨",
            # ),
        ],
        f"{NAV_PORTAL_RETURN_ICON}": [
            st.Page(
                "pages/50_ポータルへ戻る.py",
                title="ポータルへ戻る",
                icon="↩️",
            ),

        ],

        f"{NAV_STOP_ICON} 開発・管理": [
            st.Page(
                "pages/999_開発用管理者ログイン.py",
                title="開発用 管理者ログイン",
                icon="🔐",
                url_path="999_開発用管理者ログイン",
            ),
        ],       
            # st.Page(
            #     "pages/80_議事録ポータル.py",
            #     title="議事録ポータル",
            #     icon="📚",
            # ),
            # st.Page(
            #     "pages/100_ログインテスト.py",
            #     title="ログインテスト",
            #     icon="🔐",
            # ),
            # st.Page(
            #     "pages/101_ログインテスト2.py",
            #     title="ログインテスト2",
            #     icon="🔐",
            # ),
    }
)


# ============================================================
# run
# ============================================================
pg.run()