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


from common_lib.sessions.app_entry import app_session_heartbeat
from common_lib.ui.ui_basics import subtitle
from common_lib.ui.banner_lines import render_banner_line_by_key


st.set_page_config(page_title="Minutes Maker", layout="wide")
render_banner_line_by_key("light_green")
# 鎖アイコンを非表示にする
#hide_anchor_links()
#st.title("🎛️ 議事録作成アプリ（Minutes Maker）")

# ============================================================
# ログイン + セッション heartbeat（app 共通）
# ============================================================
sub = app_session_heartbeat(
    st,
    PROJECTS_ROOT,
    app_name=APP_NAME,
)

# ───────────────── ヘッダ ─────────────────
left, right = st.columns([2, 1])
with left:
    st.title("🎛️ Minutes Maker")
with right:
    st.success(f"✅ ログイン中: **{sub}**")
subtitle("議事録作成アプリ")
user = sub


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

# st.markdown(
#     """
# ### 使い方
# 1. 音声ファイルを分割します．
# 2. 上部のタブ（サイドバーの下の「Pages」メニュー）から  
#    **「文字起こし」→「話者分離」** の順に進みます． 
# 3. （人手で）実際に録音を聞いて話者分離されたテキストの間違っている箇所を修正します．  
#     この作業によって逐語録が作成されます．
# 4. **「議事録作成」** で，逐語録から議事録を作成します．
# """
# )


st.markdown(
    """
### 使い方（推奨フロー）

1. **音声ファイルをアップロードし、分割します。**  
   長時間の録音は処理しやすい単位に分割され、ストレージに保存されます．以降の処理はすべてストレージ上の成果物を入力として進みます。

2. 上部のタブ（サイドバー下の「Pages」メニュー）から、  
   **「文字起こし」→「話者分離」→「重複箇所検出」** の順に処理を進めます。  
   各ステップの結果は自動的にストレージに保存され、次の工程に引き継がれます。

3. **重複箇所検出後のテキストをダウンロードし、人手で確認・修正します。**  
   実際に音声を聞きながら、  
   - 文字起こしの変換ミス  
   - 話者の誤りや話者名の付与  
   - 文脈上不自然な箇所  
   を修正します。  
   この作業によって、**逐語録（正本）** が確定します。

4. **「議事録作成」** に逐語録を入力し、議事録を作成します。  
   確定した逐語録を基に、用途に応じた **複数タイプの議事録**（要点整理、決定事項、アクション項目など）を作成できます。

> 本アプリは、AIによる自動処理と人手による確認を組み合わせた**Human in the Loop 前提の議事録作成フロー**を採用しています。
""")


st.markdown("""
## 🚧 このアプリケーションは現在 **開発中** です

本アプリケーションシステム **Minutes Maker** は、皆様の業務効率を高めることを目的として、継続的に改良を進めています。
            実際にご利用いただき、**気づいた点・改善してほしい点・不具合** などについて、ぜひフィードバックをお寄せください。

現在は、まだ開発中です．完成まではご不便をおかけしますが、どうかご了承ください。

また、当面の運用としては **既存のクラウドサービス（AI議事録、Notta など）で作成した逐語録を本システムに取り込み、Minutes Maker の議事録生成機能のみを利用する方法** も有効です。

議事録の形式については、現在既に**社内で使用している議事録スタイル** に近づけるようプロンプトの調整を進めていく予定です。最終的には、複数の議事録フォーマットから選択できるようにし、用途に応じて使い分けられる仕組みを整えていきます。

さらに、AIガバナンスの観点から、**議事録生成時の匿名化処理（名称・個人情報のマスキング等）** にも対応していく計画です。

将来的には、どなたにも使いやすく、組織全体で活用できる議事録作成ツールへと成長させたいと考えております。そのためにも、**議事録の形式や使い勝手に関するフィードバック** を特に歓迎しております。
いただいたご意見をもとにプログラムの改善を重ね、より使いやすく、業務に役立つツールへと発展させてまいります。

ご協力のほど、どうぞよろしくお願いいたします。
""")


# サイドバー（どのページからでも同じ表示）
# render_sidebar()
