# app.py
import streamlit as st
from config.config import get_openai_api_key, DEFAULT_USDJPY
from ui.sidebarOld import init_metrics_state, render_sidebar
from ui.style import hide_anchor_links

st.set_page_config(page_title="Minutes Maker — Home", layout="wide")
# 鎖アイコンを非表示にする
hide_anchor_links()
st.title("🎛️ Minutes Maker — Home")

# 初期化
init_metrics_state()
if "usd_jpy" not in st.session_state:
    st.session_state["usd_jpy"] = DEFAULT_USDJPY

# API キー確認
OPENAI_API_KEY = get_openai_api_key()
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY が .streamlit/secrets.toml に設定されていません。")
else:
    st.success("OPENAI_API_KEY が設定されています。")

st.markdown(
    """
### 使い方
1. 左の **USD/JPY** を必要に応じて変更  
2. 上部のタブ（サイドバーの下の「Pages」メニュー）から  
   **「01 文字起こし」→「02 議事録作成」** の順に進みます。  
3. どのページでもサイドバーに **処理時間・概算料金（USD/JPY）** を表示します。
"""
)

# サイドバー（どのページからでも同じ表示）
# render_sidebar()
