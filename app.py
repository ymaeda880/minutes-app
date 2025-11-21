# app.py
import streamlit as st
from config.config import get_openai_api_key, DEFAULT_USDJPY
from ui.sidebarOld import init_metrics_state, render_sidebar
from ui.style import hide_anchor_links

st.set_page_config(page_title="議事録作成アプリ", layout="wide")
# 鎖アイコンを非表示にする
hide_anchor_links()
st.title("🎛️ 議事録作成アプリ（Minutes Maker）")

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
1. 音声ファイルを分割します．
2. 上部のタブ（サイドバーの下の「Pages」メニュー）から  
   **「文字起こし」→「話者分離」** の順に進みます． 
3. （人手で）実際に録音を聞いて話者分離されたテキストの間違っている箇所を修正します．  
    この作業によって逐語録が作成されます．
4. **「議事録作成」** で，逐語録から議事録を作成します．
"""
)

# サイドバー（どのページからでも同じ表示）
# render_sidebar()
