
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
MINUTES_MANDATORY = dedent("""        あなたは会議の議事録作成の専門家です。与えられた整形済みテキストから、
    重要事項、決定事項、TODO（担当者・期限）、論点、論拠、未解決事項を正確に抽出し、
    わかりやすく構造化した日本語の議事録を作成してください。

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
    - 事実の改変は禁止（聞き違いの推測は「不確実」と明記）
    - 日付・数量は半角、固有名詞は元の表記を維持
    - 箇条書きは簡潔、1行80字程度を目安に改行
""").strip()

MINUTES_PRESETS: List[PromptPreset] = [
    PromptPreset("none", "追記なし（基本のみ）", ""),
    PromptPreset(
        "with_summary_points",
        "サマリを箇条書きで厳密化",
        dedent("""                サマリーは必ず5行以内、文頭に「・」を付けて簡潔に要点化してください。
        """).strip()
    ),
    PromptPreset(
        "add_risks",
        "リスク/懸念の抽出を追加",
        dedent("""                # リスク・懸念
            - コスト、スケジュール、品質、セキュリティ、法務の観点で潜在的リスクを抽出
            - リスクごとに発生確率（低/中/高）と影響度（低/中/高）をタグ付け
        """).strip()
    ),
    PromptPreset(
        "exec_brief",
        "経営向けブリーフ（A4半頁）",
        dedent("""                経営層向けにA4半頁相当のブリーフも併記してください。
            見出し: 「エグゼクティブ・ブリーフ」
            内容: 目的/現状/意思決定事項/次アクションを3〜6行で要約
        """).strip()
    ),
    PromptPreset(
        "inline_timecodes",
        "元テキストのタイムコード参照付き",
        dedent("""                元テキストの [hh:mm:ss] 等のタイムコードを可能な範囲で各項目の末尾に付与してください。
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
#  レジストリ（ページ横断で参照）
# ============================================================
_REGISTRY: Dict[str, PromptGroup] = {
    SPEAKER_PREP: SPEAKER_GROUP,
    MINUTES_MAKER: MINUTES_GROUP,
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
