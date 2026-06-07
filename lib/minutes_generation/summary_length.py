# -*- coding: utf-8 -*-
# lib/minutes_generation/summary_length.py
# ============================================================
# 議事録作成：要約長設定ユーティリティ
#
# 機能：
# - 議事録モードごとの要約長設定を管理する
# - UI表示用の選択肢を返す
# - 選択された長さから目標文字数を決定する
# - 目標文字数をAIプロンプト用の内部指示文に変換する
#
# 方針：
# - Streamlitには依存しない
# - pages側はこの定義を使ってradio/number_inputを描画する
# - build_prompt()は変更せず、追加指示として連結する
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


# ============================================================
# 定数：共通ラベル
# ============================================================

CUSTOM_LABEL = "カスタム"


# ============================================================
# データ構造
# ============================================================

@dataclass(frozen=True)
class SummaryLengthConfig:
    mode: str
    title: str
    custom_label: str
    default_label: str
    default_custom_chars: int
    min_chars: int
    step_chars: int
    options: Dict[str, int]
    instruction_kind: str


# ============================================================
# モード別設定
# ============================================================

SUMMARY_LENGTH_CONFIGS: Dict[str, SummaryLengthConfig] = {
    "話者別要約": SummaryLengthConfig(
        mode="話者別要約",
        title="話者別要約の長さ",
        custom_label="1話者あたりの目標文字数",
        default_label="標準（500字程度）",
        default_custom_chars=500,
        min_chars=100,
        step_chars=100,
        options={
            "短い（200字程度）": 200,
            "標準（500字程度）": 500,
            "長い（1000字程度）": 1000,
        },
        instruction_kind="speaker_summary",
    ),
    "議事録": SummaryLengthConfig(
        mode="議事録",
        title="サマリーの長さ",
        custom_label="サマリーの目標文字数",
        default_label="標準（600字程度）",
        default_custom_chars=600,
        min_chars=100,
        step_chars=100,
        options={
            "短い（300字程度）": 300,
            "標準（600字程度）": 600,
            "長い（1000字程度）": 1000,
        },
        instruction_kind="minutes_summary",
    ),
    "会議全体要約": SummaryLengthConfig(
        mode="会議全体要約",
        title="会議全体要約の長さ",
        custom_label="全体要約の目標文字数",
        default_label="標準（1000字程度）",
        default_custom_chars=1000,
        min_chars=300,
        step_chars=100,
        options={
            "短い（500字程度）": 500,
            "標準（1000字程度）": 1000,
            "長い（2000字程度）": 2000,
        },
        instruction_kind="whole_meeting_summary",
    ),
}


# ============================================================
# 判定：要約長設定の対象モードか
# ============================================================

def is_summary_length_mode(mode: str) -> bool:
    return mode in SUMMARY_LENGTH_CONFIGS


# ============================================================
# 取得：モード別設定
# ============================================================

def get_summary_length_config(mode: str) -> Optional[SummaryLengthConfig]:
    return SUMMARY_LENGTH_CONFIGS.get(mode)


# ============================================================
# 取得：UI表示用選択肢
# ============================================================

def get_summary_length_options(mode: str) -> list[str]:
    config = get_summary_length_config(mode)
    if config is None:
        return []

    return list(config.options.keys()) + [CUSTOM_LABEL]


# ============================================================
# 解決：選択ラベルから目標文字数を決定
# ============================================================

def resolve_target_chars(
    *,
    mode: str,
    selected_label: str,
    custom_chars: int | None,
) -> Optional[int]:
    config = get_summary_length_config(mode)
    if config is None:
        return None

    if selected_label == CUSTOM_LABEL:
        if custom_chars is None:
            return config.default_custom_chars

        value = int(custom_chars)
        if value < config.min_chars:
            return config.min_chars
        return value

    return config.options.get(selected_label)


# ============================================================
# 生成：AIプロンプト用の内部指示
# ============================================================

def build_summary_length_instruction(
    *,
    mode: str,
    target_chars: int | None,
) -> str:
    config = get_summary_length_config(mode)
    if config is None or target_chars is None:
        return ""

    if config.instruction_kind == "speaker_summary":
        return f"""
【要約長設定】
話者別要約では、各話者について、1話者あたりおおむね{target_chars}字程度で要約してください。
ただし、発言量が少ない話者については、無理に字数を増やさないでください。
発言量が多い話者については、重要な発言内容、意見、質問、回答、懸念事項、提案事項を優先して整理してください。
同じ趣旨の繰り返しは統合し、発言内容の方向性や事実関係は変更しないでください。
""".strip()

    if config.instruction_kind == "minutes_summary":
        return f"""
【要約長設定】
正式議事録の「サマリー」パートは、おおむね{target_chars}字程度で作成してください。
決定事項、作業項目、重要トピック、補足・論拠、未解決事項・次回アジェンダは、サマリーとは別パートとして整理してください。
サマリーでは、会議の目的、主要論点、結論、今後の対応が読み手に分かるように簡潔に整理してください。
""".strip()

    if config.instruction_kind == "whole_meeting_summary":
        return f"""
【要約長設定】
会議全体の内容を、おおむね{target_chars}字程度で要約してください。
単なる短縮ではなく、会議の目的、主要論点、結論、未解決事項、次に取るべき対応が分かるように整理してください。
話者別ではなく、会議全体の流れと内容を一つのまとまりとして要約してください。
""".strip()

    return ""


# ============================================================
# 生成：追加指示への連結
# ============================================================

def merge_extra_with_summary_length_instruction(
    *,
    extra_text: str,
    summary_length_instruction: str,
) -> str:
    parts: list[str] = []

    if extra_text and extra_text.strip():
        parts.append(extra_text.strip())

    if summary_length_instruction and summary_length_instruction.strip():
        parts.append(summary_length_instruction.strip())

    return "\n\n".join(parts).strip()