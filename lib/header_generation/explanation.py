# -*- coding: utf-8 -*-
# minutes_app/lib/header_generation/explanation.py
# ============================================================
# 議事録ヘッダー作成ページ 説明UI
# ============================================================

from __future__ import annotations

from typing import Any

import streamlit as st

from common_lib.ui.help_expander import render_themed_help_expander
from common_lib.ui.intro_panel import (
    render_info_card_compact,
    render_info_card_bullets_compact_custom,
)


def render_header_generation_page_intro() -> None:
    """ページ上部の説明UIを表示する。"""

     # ------------------------------------------------------------
    # AI利用
    # ------------------------------------------------------------
    render_info_card_compact(
        body_html="""
        🟢 このページでは，<b>AIは使用しません</b>．<b>個人情報</b>も入力できます．
        """,
            )


    render_info_card_compact(
        body_html="""
このページでは，Excel または CSV 形式の表から，議事録ヘッダー用の Word 文書を作成します．
"""
    )

    render_info_card_bullets_compact_custom(
        title="使い方",
        items=[
            ("①", "入力方法を選択し，<b>Excel (.xlsx) または CSV (.csv)</b> の内容を読み込みます．"),
            ("", "貼り付け，パソコンからのアップロード，Inboxから選択できます．"),
            ("②", "最初のシート内に縦に並んだ表ブロックを検出します．"),
            ("③", "<b>「Wordを作成」</b>ボタンを押します．"),
            ("④", "作成した Wordファイルはパソコンに保存するか，Inboxに保存できます．"),
        ],
    )

    st.markdown(
        "<div style='height:16px'></div>",
        unsafe_allow_html=True,
    )


def render_header_generation_help_expander(
    *,
    theme: dict[str, Any] | None = None,
    banner_key: str = "light_green",
) -> None:
    """詳細説明 expander を表示する。"""

    render_themed_help_expander(
        expander_key=HELP_EXPANDER_KEY,
        expander_title=HELP_EXPANDER_TITLE,
        tabs=HELP_TABS,
        theme=theme,
        banner_key=banner_key,
        expanded=False,
    )


HELP_EXPANDER_KEY = "header_generation_help_expander"
HELP_EXPANDER_TITLE = "詳細説明（クリックで展開）"


HEADER_USAGE_TEXT = """

##### 1. このページの役割

Excel または CSV で作成された出席者一覧・配布資料一覧などを読み込み，
議事録の冒頭に貼り付けるための Word ヘッダーを作成します．

---

##### 2. 入力方法

入力方法は次の3つです．

1. 貼り付け
2. ファイルアップロード
3. Inbox から選択

Excel は `.xlsx`，CSV は `.csv` に対応します．

---

##### 3. 想定する表形式

1つのシート内に，次のような表ブロックが縦に並んでいる形式を想定しています．

- 出席者（委員）
- 出席者（事務局）
- 欠席者
- 配布資料

各ブロックは，見出し行の次に表が続く形式です．

---

##### 4. 出力

作成した Word は，

- パソコンへダウンロード
- Inbox へ保存

のどちらにも対応します．

</div>
"""

HEADER_FORMAT_TEXT = """

##### 入力表の形式

このページでは，Excel または CSV に記載された表から議事録ヘッダーを作成します．

1つのシート内に，複数の表を縦方向に並べてください．

---

##### 入力例

出席者（委員）

| 氏名 | 所属・役職等 | 専門分野等 | 備考 |
|------|-------------|-----------|------|
| 山田 太郎 | ○○商工会 前会長 | 商工 | 会長 |
| 鈴木 一郎 | ○○協会 理事 | 地域振興 | |

（空行）

出席者（事務局）

| 所属名 | 職名 | 氏名 |
|--------|------|------|
| ○○市教育委員会 | 教育長 | 山田 花子 |
| ○○市教育委員会 | 主査 | 佐藤 次郎 |

（空行）

欠席者

| 氏名 | 所属・役職等 | 専門分野等 | 備考 |
|------|-------------|-----------|------|
| 高橋 三郎 | ○○大学教授 | 歴史学 | |

---

##### 判定ルール

- 1つのセルだけに文字が入力されている行を表タイトルとして扱います．
- タイトル行の次の行を表ヘッダーとして扱います．
- その下の行を表データとして扱います．
- 空行が現れると，その表は終了したものと判定します．
- 次のタイトル行が現れると，新しい表として読み込みます．

---

##### 注意事項

- 1つのシート内に複数の表を配置できます．
- 表と表の間には空行を入れてください．
- タイトル行は1セルのみ入力してください．
- Excel（.xlsx）および CSV（.csv）に対応しています．

</div>
"""

HELP_TABS = [
    ("使い方", HEADER_USAGE_TEXT),
    ("入力形式", HEADER_FORMAT_TEXT),
]