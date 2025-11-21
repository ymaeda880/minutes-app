
# lib/prompts.py
# ------------------------------------------------------------
# 統一プロンプトレジストリ
# - 話者分離（Speaker Prep）
# - 議事録作成（Minutes Maker）
# どちらも同じデータモデルとヘルパーで扱えるように実装。
# ------------------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass
from textwrap import dedent
from typing import List, Dict, Optional

# ===== カテゴリ識別子（ページ毎の名前空間） =====
SPEAKER_PREP = "speaker_prep"
MINUTES_MAKER = "minutes_maker"
MINUTES_STYLE = "minutes_style"  # ← 見た目スタイル用グループを追加

@dataclass(frozen=True)
class PromptPreset:
    key: str      # 内部キー
    label: str    # UI表示名
    body: str     # 追記本文

@dataclass(frozen=True)
class PromptGroup:
    group_key: str           # 例: SPEAKER_PREP / MINUTES_MAKER
    title: str               # UI/管理用タイトル
    mandatory_default: str   # 必須パート
    presets: List[PromptPreset]
    default_preset_key: str

    # ---- ユーティリティ（インスタンスメソッド） ----
    def preset_labels(self) -> List[str]:
        return [p.label for p in self.presets]

    def body_for_label(self, label: str) -> str:
        for p in self.presets:
            if p.label == label:
                return p.body
        return ""

    def label_for_key(self, key: str) -> str:
        for p in self.presets:
            if p.key == key:
                return p.label
        # fallback: 先頭のラベル
        return self.preset_labels()[0] if self.presets else ""

# ============================================================
#  話者分離（Speaker Prep）
# ============================================================
SPEAKER_MANDATORY = dedent("""        あなたは日本語の会議文字起こしを整形する専門家です。
    以下の「生の文字起こしテキスト」を読み、話者を推定しながら発話ごとに改行して可読化してください。

    必須要件:
    1) 話者ラベルは S1:, S2:, S3: ... の形式
    2) 司会者は発言内容から特定し「司会者:」とする
    3) 文字は一字一句変えない

    注意:
    - 文字は変更しない・付け加えない
    - 会話の順序を維持
    - 話者ごとに改行し、さらに1行空ける
""").strip()

SPEAKER_PRESETS: List[PromptPreset] = [
    PromptPreset("none", "追記なし（基本のみ）", ""),
    PromptPreset(
        "strict_break",
        "文字起こしの誤りの指摘",
        dedent("""                出力は以下の 2 部構成:
            【整形テキスト】…（S1:, S2: で始まる行で構成）
            【メモ】話者数の推定 / 主要トピック3点 / 用語ゆらぎの正規化例
            文字起こしの誤りと思われる箇所があれば列挙してください。
        """).strip()
    ),
    PromptPreset(
        "keep_ts",
        "タイムスタンプ保持",
        "入力に含まれる [hh:mm:ss] 等のタイムスタンプは削除せず各発話の先頭に残してください。"
    ),
    PromptPreset(
        "keep_noise",
        "原文完全保持（ノイズ・重複も残す）",
        "咳払い・えー/あー・（笑）なども削除せず、話者推定と改行のみ行ってください。"
    ),
    PromptPreset(
        "shrink_spaces",
        "空白縮約のみ",
        "文言や記号は変更せず、全角/半角スペースの連続のみ1つに縮約してください（句読点の前後は変更不可）。"
    ),
]

SPEAKER_GROUP = PromptGroup(
    group_key=SPEAKER_PREP,
    title="話者分離・整形",
    mandatory_default=SPEAKER_MANDATORY,
    presets=SPEAKER_PRESETS,
    default_preset_key="none",
)
# ============================================================
#  議事録作成（Minutes Maker）
# ============================================================
# モード別 mandatory プロンプト
MINUTES_MANDATORY_MODES: Dict[str, str] = {
    "逐語録の作成": dedent("""
        あなたは日本語の会議逐語録を整える専門家です。
        以下の「整形済みテキスト」（話者ごとに分離されたテキスト）をもとに、
        元の発話内容をできるだけ忠実に残した「逐語録」を作成してください。

        要件:
        - 発言の順序は変えないこと。
        - 語句の変更は行わないこと．
                     
        出力フォーマット（見出し例）:
        # 会議概要
        - 会議名（分かる範囲で）
        - 主なテーマ
        - 日時・場所（分かる範囲で）
        - 参加者（役職や氏名が分かる範囲で）
                     
        # 逐語録:           
        - 見出しやサマリーは付けず、シンプルな逐語録として出力してください。
        - 発言ごとの通し番号を「（0001）」などと見やすく振ってください
        - 発言ごとに空白行を入れて見やすくしてください．
    """).strip(),

    "簡易議事録の作成": dedent("""
        あなたは日本語の会議議事録を作成する専門家です。
        以下の「整形済みテキスト」（話者ごとに分離されたテキスト）をもとに、
        会議の要点を押さえた「簡易議事録」を作成してください。

        基本方針:
        - 発言内容をそのまま書き写すのではなく、論点・意見・確認事項・決定事項を整理してまとめてください。
        - 脱線した雑談や細かな言い回しは省略して構いません。
        - あとから参加者が読み返したときに「どのような議論があり、何が決まったか」が把握できるレベルを目指します。
        - 全体として A4 1〜2枚程度のボリューム感に収まる密度で要約してください。

        出力フォーマット（見出し例）:
        # 会議概要
        - 会議名（分かる範囲で）
        - 主なテーマ
        - 日時・場所（分かる範囲で）
        - 参加者（役職や氏名が分かる範囲で）

        # 議論の概要
        - 議論された主なトピックを箇条書きで整理（1トピックあたり1〜3行）

        # 決定事項
        - 箇条書きで簡潔に記述（1行で要約）
        - 必要に応じて背景を1文だけ補足してもよい

        # TODO
        - [担当: 氏名または部署, 期限: 分かる範囲で] 作業内容
        - 期限や担当が不明な場合は「担当・期限: 未確定」と明記

        # メモ・補足（任意）
        - 今後の検討事項や、論点として残しておくべき内容があれば簡潔に記載

        制約:
        - 事実の改変は禁止（推測が入るときは「〜と考えられる」「〜と思われる」といった表現を用いる）。
        - 日付・時刻・数量は半角で統一してください。
        - 固有名詞はできるだけ元の表記を維持してください。
    """).strip(),


    "詳細議事録の作成": dedent("""
        あなたは会議の議事録作成の専門家です。与えられた整形済みテキストから、
        重要事項、決定事項、TODO（担当者・期限）、論点、論拠、未解決事項を正確に抽出し、
        わかりやすく構造化した日本語の詳細議事録を作成してください。

        出力フォーマット（見出しはこの順序・文言で出力）:
        # サマリー（3〜5行）
        # 決定事項
        - 箇条書き（番号付き）。各項目に背景/根拠を1文で添付。
        # TODO（担当・期限つき）
        - 例）[担当: 氏名, 期限: YYYY-MM-DD] 具体的な作業内容
        # 重要トピック（最大5つ）
        - 各トピックの論点と結論を短く
        # 補足・論拠
        - 参照すべき資料・データがあれば列挙
        # 未解決事項・次回アジェンダ

        制約:
        - 事実の改変は禁止（聞き違いの推測は「不確実」と明記）。
        - 日付・数量は半角、固有名詞は元の表記を維持。
        - 箇条書きは簡潔、1行80字程度を目安に適宜改行。
        - 必要に応じて、重要な発言については「誰が・何を主張したか」が分かるように要約してください。
    """).strip(),
}

# デフォルト（後方互換用）は「簡易議事録の作成」
MINUTES_MANDATORY = MINUTES_MANDATORY_MODES["簡易議事録の作成"]

MINUTES_PRESETS: List[PromptPreset] = [
    PromptPreset("none", "追記なし（基本のみ）", ""),
    PromptPreset(
        "with_summary_points",
        "サマリを箇条書きで厳密化",
        dedent("""
            サマリーは必ず5行以内とし、文頭に「・」を付けた箇条書き形式で、
            会議の目的・結論・重要な決定事項が一目で分かるように簡潔に要約してください。
        """).strip()
    ),
    PromptPreset(
        "add_risks",
        "リスク/懸念の抽出を追加",
        dedent("""
            # リスク・懸念
            - コスト、スケジュール、品質、セキュリティ、法務の観点で潜在的リスクを抽出してください。
            - 各リスクについて「発生確率（低/中/高）」と「影響度（低/中/高）」をタグとして付与してください。
        """).strip()
    ),
    PromptPreset(
        "exec_brief",
        "経営向けブリーフ（A4半頁）",
        dedent("""
            経営層向けにA4半頁相当のブリーフも併記してください。
            見出し: 「エグゼクティブ・ブリーフ」
            内容: 目的 / 現状 / 意思決定事項 / 次アクションを3〜6行で簡潔にまとめてください。
        """).strip()
    ),
    PromptPreset(
        "inline_timecodes",
        "元テキストのタイムコード参照付き",
        dedent("""
            元テキストに [hh:mm:ss] 等のタイムコードが含まれている場合、
            可能な範囲で関連する項目の末尾に括弧付きで付与してください（例: （参照: 00:12:34））。
        """).strip()
    ),
]

MINUTES_GROUP = PromptGroup(
    group_key=MINUTES_MAKER,
    title="議事録作成",
    mandatory_default=MINUTES_MANDATORY,
    presets=MINUTES_PRESETS,
    default_preset_key="none",
)


# ============================================================
#  議事録の見た目スタイル（Minutes Style）
# ============================================================
MINUTES_STYLE_MANDATORY = dedent("""
    これは議事録の内容ではなく「レイアウト・見た目」に関するスタイル指定です。
    内容の要約や構造化に関する指示は別のプロンプト（逐語録 / 簡易 / 詳細 など）に従い、
    ここでは Word に貼り付けたときの見た目を整えるためのルールだけを指定します。
""").strip()

MINUTES_STYLE_PRESETS: List[PromptPreset] = [
    # 見た目1：ベーシック（シンプルな Markdown / Word 貼り付け用）
    PromptPreset(
        key="style_basic",
        label="見た目1：ベーシック（シンプル）",
        body=dedent("""
            【見た目スタイル：ベーシック】

            - 見出しは「# 見出し名」「## 見出し名」の形式で統一してください。
            - 各見出しの前後には 1 行空行を入れて、セクションの境界をはっきりさせてください。
            - 箇条書きは「- 」で統一し、入れ子の箇条書きは「Tab + - 」で表現してください。
              （Word に貼り付けたときに自動的に階層付き箇条書きになります。）
            - 行頭に不要な全角スペース・半角スペースを入れないでください。
            - 長すぎる段落は 2〜4 行程度で読みやすく改行して構いません。
            - セクションとセクションの間は 1 行だけ空け、詰まりすぎない読みやすいレイアウトにしてください。
        """).strip(),
    ),

    # 見た目2：横線＋「== 見出し ==」区切り
    PromptPreset(
        key="style_hr_heading",
        label="見た目2：横線＋装飾見出し",
        body=dedent("""
            【見た目スタイル：横線＋装飾見出し】

            - 文書の最上部に「-----------------------------」を 1 行入れてください。
            - 主要なセクション（会議概要・サマリー・決定事項・TODO・議論のポイントなど）の
              見出しは「== 見出し名 ==」の形式で出力してください。
            - 各主要セクションの直前には必ず「-----------------------------」を 1 行入れて区切ってください。
            - 見出しと本文の間、およびセクションとセクションの間には 1 行以上の空行を入れてください。
            - 箇条書きは「- 」で統一し、行頭のインデントは入れず左揃えにしてください。
            - 強調したい語句には **太字** を使っても構いません（Word に貼り付けた際に視認性が上がります）。
        """).strip(),
    ),

    # 見た目3：Word の見出しスタイル対応（厳密 # / ##）
    PromptPreset(
        key="style_word_headings",
        label="見た目3：Word見出しスタイル向け（改良版）",
        body=dedent("""
            【見た目スタイル：Word 見出しスタイル向け（改良版）】

            ▼ 見出しの書式（Wordの見出し1/2/3に安全に変換できる形式）
            - 見出しは Markdown の見出し記法を使用する (#, ##, ###)。
            - 見出し行の前後には必ず1行の空行を入れること。
            - 見出し行の先頭は「#」から始め、前にスペースを置かない（Wordが箇条書き判定しない）。

            ▼ 箇条書き（Wordのオートフォーマット誤作動防止）
            - 箇条書きは必ず「- 」で統一する。
            - 箇条書き行の前には**全角スペース1つを入れてから**「- 」を書くこと。
            例）"　- スケジュールの確認"
            - 入れ子（下位階層）は全角スペース2つ入れてから「- 」を書くこと。
            - 箇条書きの上下には最低でも1行の余白を入れ、上位箇条目（●）と
            下位箇条目（-）が密着しないようにする。

            ▼ 行頭記号の誤認防止（Word対策）
            - 行頭に「*」「●」「■」などの記号を置かない。
            - 箇条書き以外の行の行頭にも記号を置かない。
            - 見出しの直後に記号を置くとWordが自動で箇条書きに変換するため禁止。

            ▼ 段落の整形
            - 文章は左揃えに統一し、行頭インデントは付けない。
            - 不要な連続改行は避け、論理的なまとまりごとに段落をまとめる。
            - 見出し直後に文章を書く場合でも、1行の余白は必ず確保する。

            ▼ 総合ルール
            - この書式は Word の「スタイル」機能で見出し1/2/3を適用しやすいように設計している。
            - Word 貼り付け後の行頭の小さな四角（■）が表示されないように、
            行頭記号の使い方は上記の通り厳密に従うこと。
        """).strip(),
)]


MINUTES_STYLE_GROUP = PromptGroup(
    group_key=MINUTES_STYLE,
    title="議事録見た目スタイル",
    mandatory_default=MINUTES_STYLE_MANDATORY,
    presets=MINUTES_STYLE_PRESETS,
    default_preset_key="style_basic",
)


# ============================================================
#  レジストリ（ページ横断で参照）
# ============================================================
_REGISTRY: Dict[str, PromptGroup] = {
    SPEAKER_PREP: SPEAKER_GROUP,
    MINUTES_MAKER: MINUTES_GROUP,
    MINUTES_STYLE: MINUTES_STYLE_GROUP,  # ← 追加
}


def get_group(group_key: str) -> PromptGroup:
    if group_key not in _REGISTRY:
        raise KeyError(f"Unknown prompt group: {group_key}")
    return _REGISTRY[group_key]

def build_prompt(mandatory: str, preset_body: str, extra: str, src_text: str) -> str:
    """実際にモデルへ渡すプロンプトを組み立て（共通関数）"""
    parts = [mandatory.strip()]
    if preset_body and preset_body.strip():
        parts.append(preset_body.strip())
    if extra and extra.strip():
        parts.append("【追加指示】\n" + extra.strip())
    parts.append("【入力テキスト】\n" + src_text)
    return "\n\n".join(parts)
