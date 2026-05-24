# text_studio_app/pages/999_開発用管理者ログイン.py
# ============================================================
# 🔐 開発用 管理者ログイン
# ============================================================

from __future__ import annotations

# ============================================================
# imports
# ============================================================
from pathlib import Path
import sys
import json
import tomllib

import streamlit as st
import extra_streamlit_components as stx

# ============================================================
# page config
# ============================================================
st.set_page_config(
    page_title="開発用 管理者ログイン",
    page_icon="🔐",
    layout="wide",
)

# ============================================================
# paths
# ============================================================
_THIS = Path(__file__).resolve()
APP_DIR = _THIS.parents[1]
PROJ_DIR = _THIS.parents[2]
MONO_ROOT = _THIS.parents[3]

for p in (MONO_ROOT, PROJ_DIR, APP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

PROJECTS_ROOT = MONO_ROOT
APP_NAME = APP_DIR.name
PAGE_NAME = _THIS.stem

AUTH_APP_DIR = MONO_ROOT / "auth_portal_project" / "auth_portal_app"
AUTH_USERS_FILE = AUTH_APP_DIR / "data" / "users.json"
AUTH_SETTINGS_FILE = AUTH_APP_DIR / ".streamlit" / "settings.toml"

APP_SETTINGS_FILE = APP_DIR / ".streamlit" / "settings.toml"

# ============================================================
# common_lib
# ============================================================
from common_lib.ui.banner_lines import render_banner_line_by_key
from common_lib.ui.theme_colors import get_theme_colors_from_banner_key
from common_lib.ui.intro_panel import (
    render_intro_css,
    render_info_card_compact,
)
from common_lib.auth.admin_local_login import render_admin_local_login_form

# ============================================================
# helper：toml/json 読み込み
# ============================================================
def load_toml_file(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"settings.toml が見つかりません: {path}")

    with open(path, "rb") as f:
        return tomllib.load(f)


def load_json_file(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"users.json が見つかりません: {path}")

    return json.loads(path.read_text(encoding="utf-8"))

# ============================================================
# settings
# ============================================================
app_settings = load_toml_file(APP_SETTINGS_FILE)
access_settings = load_toml_file(AUTH_SETTINGS_FILE)
users_db = load_json_file(AUTH_USERS_FILE)

# ============================================================
# banner / theme
# ============================================================
BANNER_KEY = app_settings.get(
    "BANNER_KEY",
    "navy_dark",
)

render_banner_line_by_key(BANNER_KEY)

theme = get_theme_colors_from_banner_key(BANNER_KEY)
render_intro_css(theme)

# ============================================================
# title
# ============================================================
current_user = st.session_state.get("current_user")

left, right = st.columns([2, 1])

with left:
    st.title("🔐 開発用 管理者ログイン")

with right:
    if current_user:
        st.success(f"✅ ログイン中: **{current_user}**")
    else:
        st.info("未ログイン")

# ============================================================
# info
# ============================================================
render_info_card_compact(
    body_html="""
このページは <b>開発用の管理者ログインページ</b> です．<br>
auth_portal_app を起動しなくても，この app 単体で管理者ログインできます．
""",
)

render_info_card_compact(
    body_html="""
管理者以外のユーザー名では，パスワードが正しくてもログインできません．
""",
)

st.markdown(
    "<div style='height:16px'></div>",
    unsafe_allow_html=True,
)

# ============================================================
# login form
# ============================================================
cm = stx.CookieManager(
    key=f"{PAGE_NAME}_cookie_manager",
)

render_admin_local_login_form(
    cookie_manager=cm,
    users_db=users_db,
    access_settings=access_settings,
    title="🔐 管理者ログイン",
)

# ============================================================
# footer
# ============================================================
st.divider()

st.caption("開発専用ページです。")