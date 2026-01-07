# pages/101_ログインテスト2.py
from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

# --- sys.path 調整（pages/13_ボット に倣う） ---
_THIS = Path(__file__).resolve()
PROJECTS_ROOT = _THIS.parents[3]
if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

from common_lib.auth.auth_helpers import get_current_user_from_session_or_cookie

# ============================================================
# UI（最小）
# ============================================================
st.set_page_config(
    page_title="ログインテスト2（pages/13と同じ）",
    page_icon="🧪",
    layout="centered",
)

st.title("🧪 ログインテスト2")
st.caption("pages/13_ボット（ログ管理拡張版）と完全に同じログイン判定で表示します。")

# ============================================================
# pages/13 と同じ「ログイン判定」部分だけ
# ============================================================
current_user, _ = get_current_user_from_session_or_cookie(st)
if current_user:
    st.success(f"ログイン中: **{current_user}**")
else:
    st.warning("未ログイン")
