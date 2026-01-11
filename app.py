# app.py
import streamlit as st
from config.config import get_openai_api_key, DEFAULT_USDJPY
from ui.sidebar import init_metrics_state, render_sidebar
from ui.style import hide_anchor_links


# ============================================================
# パスの取得とcommon_lib読み込み（app.pyにおけるコード）
# ============================================================
from pathlib import Path
import sys
import streamlit as st

_THIS = Path(__file__).resolve()
APP_ROOT = _THIS.parent
APP_NAME = APP_ROOT.name                  # ← app_name を自動取得
PROJECTS_ROOT = _THIS.parents[2]

if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

from common_lib.sessions import SessionConfig, init_session, heartbeat_tick
from common_lib.auth.auth_helpers import require_login



st.set_page_config(page_title="議事録作成アプリ", layout="wide")
# 鎖アイコンを非表示にする
#hide_anchor_links()
#st.title("🎛️ 議事録作成アプリ（Minutes Maker）")

# ============================================================
# Session heartbeat（全ページ共通・app.py）
# ============================================================
SESSIONS_DB = (
    PROJECTS_ROOT / "Storages" / "_admin" / "sessions" / "sessions.db"
)
CFG = SessionConfig()  # heartbeat=30s, TTL=120s（既定）

# ───────────────── ログイン必須 ─────────────────

sub = require_login(st)
if not sub:
    st.stop()

# ───────────────── ヘッダ ─────────────────
left, right = st.columns([2, 1])
with left:
    st.title("🎛️ 議事録作成アプリ（Minutes Maker）")
with right:
    st.success(f"✅ ログイン中: **{sub}**")

user = sub

# ───────────────── sessions（初期化 + heartbeat） ─────────────────
init_session(db_path=SESSIONS_DB, cfg=CFG, user_sub=user, app_name=APP_NAME)
heartbeat_tick(db_path=SESSIONS_DB, cfg=CFG, user_sub=user, app_name=APP_NAME)



# 初期化
init_metrics_state()
if "usd_jpy" not in st.session_state:
    st.session_state["usd_jpy"] = DEFAULT_USDJPY

# API キー確認
OPENAI_API_KEY = get_openai_api_key()
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY が .streamlit/secrets.toml に設定されていません。")
#else:
    #st.success("OPENAI_API_KEY が設定されています。")

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

st.markdown("""
## 🚧 このアプリケーションは現在 **開発中** です

本アプリケーションシステム **Minutes Maker** は、皆様の業務効率を高めることを目的として、継続的に改良を進めています。
            実際にご利用いただき、**気づいた点・改善してほしい点・不具合** などについて、ぜひフィードバックをお寄せください。

現在は、音声ファイルを手動で分割し、それぞれを文字起こしする方式となっています。将来的には、**音声ファイルをドロップするだけで**、

- 自動で適切に分割  
- 各ファイルを順次文字起こし  
- 文字起こし結果の自動結合  
- 話者分離（speaker diarization）の自動実行  

までを一括で行えるようにする予定です。それまではご不便をおかけしますが、どうかご了承ください。

また、当面の運用としては **既存のクラウドサービス（AI議事録、Notta など）で作成した逐語録を本システムに取り込み、Minutes Maker の議事録生成機能のみを利用する方法** も有効です。

議事録の形式については、現在既に**社内で使用している議事録スタイル** に近づけるようプロンプトの調整を進めていく予定です。最終的には、複数の議事録フォーマットから選択できるようにし、用途に応じて使い分けられる仕組みを整えていきます。

さらに、AIガバナンスの観点から、**議事録生成時の匿名化処理（名称・個人情報のマスキング等）** にも対応していく計画です。

将来的には、どなたにも使いやすく、組織全体で活用できる議事録作成ツールへと成長させたいと考えております。そのためにも、**議事録の形式や使い勝手に関するフィードバック** を特に歓迎しております。
いただいたご意見をもとにプログラムの改善を重ね、より使いやすく、業務に役立つツールへと発展させてまいります。

ご協力のほど、どうぞよろしくお願いいたします。
""")


# サイドバー（どのページからでも同じ表示）
# render_sidebar()
