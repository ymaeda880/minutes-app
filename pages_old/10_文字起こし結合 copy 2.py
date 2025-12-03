# -*- coding: utf-8 -*-
# pages/10_文字起こし結合.py
#
# 「2分オーバーラップの同じ話題部分」を正確に検出し、
# 前半 → 「ここから重複」
# 後半 → 「ここまでが重複部分」
# を正しい位置に入れる処理。
#
# ポイント：
#   - 後半セグメントの *最初の700文字付近* を重視
#   - b（next_head 内位置）が最小の一致ブロックを採用
#   - 長さ MIN_MATCH_SIZE 以上のものだけ採用
#   - 文字列は削除しない。マーカーだけ入れる。
#
# Streamlit UI 付き（複数の .txt ファイル対応）


from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple

import streamlit as st


# ============================================================
# 設定値
# ============================================================

# 「オーバーラップあり」とみなす最低一致長（文字数）
MIN_MATCH_SIZE = 40   # 700字前提なら 40〜60 が妥当

# つなぎ目を示す行のパターン
MARKER_PATTERN = re.compile(
    r"^-{3,}\s*ここがつなぎ目です（(.*?)）.*$",
    re.MULTILINE,
)

# マーカー
OVERLAP_START = "-----ここから重複-----"
OVERLAP_END   = "-----ここまでが重複部分-----"
NO_OVERLAP_MARK = "[-----重複はありませんでした-----]"


# ============================================================
# セグメント分割
# ============================================================

def split_by_markers(text: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    segments = []
    markers = []

    prev_end = 0
    for m in MARKER_PATTERN.finditer(text):
        seg = text[prev_end:m.start()]
        segments.append(seg)

        markers.append(
            {
                "file_name": m.group(1),
                "marker_text": m.group(0),
                "start": m.start(),
                "end": m.end(),
            }
        )
        prev_end = m.end()

    segments.append(text[prev_end:])
    return segments, markers


# ============================================================
# ★ 改良版 オーバーラップ検出ロジック
# ============================================================

def find_overlap(prev_seg: str, next_seg: str, overlap_chars: int) -> Dict[str, Any]:
    """
    改良版：
    - next_head の最初（先頭側）に近い一致ブロックほど優先
    - b（next_head 側の開始位置）が小さいものを最優先
    - かつ size が MIN_MATCH_SIZE 以上のもの
    """

    prev_tail = prev_seg[-overlap_chars:]
    next_head = next_seg[:overlap_chars]

    tail_offset = len(prev_seg) - len(prev_tail)

    sm = SequenceMatcher(None, prev_tail, next_head)
    blocks = sm.get_matching_blocks()

    # 候補の中から「b が小さい & size が大きい」ブロックを選ぶ
    good_blocks = [
        (a, b, size)
        for (a, b, size) in blocks
        if size >= MIN_MATCH_SIZE
    ]

    if not good_blocks:
        return {
            "match_size": 0,
            "prev_start_idx": None,
            "next_end_idx": None,
        }

    # ★（改良点）b が小さいほど「後半の冒頭に近い一致」と解釈できる
    #   b の小ささ → size の大きさ の順で優先
    best = sorted(good_blocks, key=lambda t: (t[1], -t[2]))[0]

    a, b, size = best

    prev_start_idx = tail_offset + a
    next_end_idx = b + size

    return {
        "match_size": size,
        "prev_start_idx": prev_start_idx,
        "next_end_idx": next_end_idx,
    }


# ============================================================
# 文字列への挿入ユーティリティ
# ============================================================

def apply_insertions(base: str, inserts: List[Tuple[int, str]]) -> str:
    if not inserts:
        return base

    inserts_sorted = sorted(inserts, key=lambda x: x[0])
    offset = 0
    s = base
    for pos, text in inserts_sorted:
        real_pos = max(0, min(len(s), pos + offset))
        s = s[:real_pos] + text + s[real_pos:]
        offset += len(text)
    return s


# ============================================================
# 全体結合
# ============================================================

def build_report_and_merged_text(text: str, overlap_chars: int) -> Tuple[str, str]:
    segments, markers = split_by_markers(text)

    n_seg = len(segments)
    n_mark = len(markers)

    if n_mark == 0:
        return "マーカーがありません。", text

    # 挿入指示
    seg_insertions = [[] for _ in range(n_seg)]
    has_overlap = [False] * n_mark

    report_lines = []
    report_lines.append("【オーバーラップ推定レポート】")
    report_lines.append(f"セグメント数: {n_seg} / マーカー数: {n_mark}")
    report_lines.append(f"解析範囲: 前後 {overlap_chars} 文字")
    report_lines.append("")

    # ---- 各つなぎ目処理 ----
    for idx, marker in enumerate(markers):
        prev_seg = segments[idx]
        next_seg = segments[idx + 1]

        ol = find_overlap(prev_seg, next_seg, overlap_chars)

        report_lines.append("=" * 70)
        report_lines.append(f"■ つなぎ目 {idx+1}: {marker['file_name']}")
        report_lines.append(marker["marker_text"])
        report_lines.append(f"- 一致長: {ol['match_size']} 文字")
        report_lines.append("")

        if ol["match_size"] >= MIN_MATCH_SIZE:
            has_overlap[idx] = True

            prev_pos = ol["prev_start_idx"]
            next_pos = ol["next_end_idx"]

            if prev_pos is not None:
                seg_insertions[idx].append((prev_pos, "\n" + OVERLAP_START + "\n"))

            if next_pos is not None:
                seg_insertions[idx + 1].append((next_pos, "\n" + OVERLAP_END + "\n"))

            report_lines.append("→ 有意なオーバーラップを検出しました。")
        else:
            report_lines.append("→ 重複は検出されませんでした。")
        report_lines.append("")

    # ---- セグメントごとに挿入を反映 ----
    modified_segments = [
        apply_insertions(seg, seg_insertions[i])
        for i, seg in enumerate(segments)
    ]

    # ---- マーカーと結合 ----
    merged_parts = []
    for i in range(n_seg):
        merged_parts.append(modified_segments[i])
        if i < n_mark:
            merged_parts.append("\n" + markers[i]["marker_text"] + "\n")
            if not has_overlap[i]:
                merged_parts.append(NO_OVERLAP_MARK + "\n")

    merged_text = "".join(merged_parts)
    report_text = "\n".join(report_lines)

    return report_text, merged_text


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(page_title="📝文字起こし結合（重複検出）", layout="wide")

st.title("📝 文字起こし結合（2分重複を正確に検出）")

st.markdown(
    """
**AI文字起こしの揺れを考慮した“2分オーバーラップ領域”の正確な検出ツールです。**

- 後半セグメントの「最初の700文字」を基準に重複を判定  
- 「ここから重複」「ここまでが重複部分」を本文に挿入  
- 文字列は削除せず全文保持  
"""
)

overlap_chars = st.slider(
    "重複として見る文字数（前後それぞれ）",
    min_value=300, max_value=2000, step=100,
    value=700,  # ← 初期値 700 に変更
)

uploaded = st.file_uploader(
    "文字起こしテキスト (.txt) をアップロード（複数可）",
    type=["txt"], accept_multiple_files=True
)

run = st.button("▶ 重複検出を実行")

if run:
    if not uploaded:
        st.warning("先にファイルをアップロードしてください。")
    else:
        for up in uploaded:
            st.subheader(f"📄 {up.name}")

            raw = up.read()
            try:
                text = raw.decode("utf-8")
            except:
                text = raw.decode("cp932", errors="replace")

            report_text, merged_text = build_report_and_merged_text(text, overlap_chars)

            with st.expander("🔍 オーバーラップレポート", expanded=True):
                st.text(report_text)

            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    "📥 レポートをダウンロード (.txt)",
                    report_text.encode("utf-8"),
                    file_name=f"{up.name}_overlap_report.txt",
                )

            with col2:
                st.download_button(
                    "📥 重複マーク付き結合 (.txt)",
                    merged_text.encode("utf-8"),
                    file_name=f"{up.name}_merged_with_marks.txt",
                )

        st.success("完了しました。結果を確認してください。")
