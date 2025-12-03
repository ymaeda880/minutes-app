# -*- coding: utf-8 -*-
# pages/10_重複箇所検出.py
#
# 後半セグメントの「最初の数行」をキーに、
# 前半セグメントの「最後の2分相当（文字数）」の中から
# もっともよく一致する位置を探し、
# 前半側にだけ「-----ここから重複-----」を挿入する。
#
# 後半側には「ここまでが重複部分」は入れない。

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Dict, Tuple

import streamlit as st

# ============================================================
# 設定値
# ============================================================

# 「前後セグメントの比較範囲」＝オーバーラップ長（前半の最後の何文字を見るか）
OVERLAP_CHARS = 700  # おおよそ2分相当の目安

# 「後半の最初の数行」として使う最大文字数
HEAD_CHARS = 400

# 「後半の最初の数行」として区切る最大文数
HEAD_SENTENCES = 3

# 一致とみなす最低文字数
MIN_MATCH_SIZE = 30

# 境界マーカー行
MARKER_PATTERN = re.compile(
    r"^-{3,}\s*ここがつなぎ目です（(.*?)）.*$",
    re.MULTILINE,
)

BEGIN_TAG = "-----ここから重複-----"


# ============================================================
# セグメント分割
# ============================================================

def split_by_markers(text: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    全文から「つなぎ目マーカー行」で分割して
    segments と markers を返す。

    [seg0][marker0][seg1][marker1][seg2]...
    → segments = [seg0, seg1, seg2, ...]
      markers  = [marker0, marker1, ...]
    """
    segments: List[str] = []
    markers: List[Dict[str, str]] = []

    prev_end = 0

    for m in MARKER_PATTERN.finditer(text):
        seg = text[prev_end:m.start()]
        segments.append(seg)

        markers.append(
            {
                "file_name": m.group(1),
                "marker_text": m.group(0),
            }
        )
        prev_end = m.end()

    segments.append(text[prev_end:])
    return segments, markers


# ============================================================
# 「後半の最初の数行」を抜き出す
# ============================================================

def extract_head_phrase(next_seg: str) -> str:
    """
    後半セグメントから「最初の数行」を取り出す。

    - 先頭 HEAD_CHARS 文字を対象
    - その中で「。」「？」「！」「改行」を文の区切りとみなし、
      HEAD_SENTENCES 文まで含めた部分を head_phrase として返す。
    """
    if not next_seg:
        return ""

    s = next_seg[:HEAD_CHARS]
    count = 0
    end = len(s)

    for i, ch in enumerate(s):
        if ch in "。？！\n":
            count += 1
            if count >= HEAD_SENTENCES:
                end = i + 1  # 区切り記号も含める
                break

    return s[:end]


# ============================================================
# 重複開始位置の探索（前半側のみ）
# ============================================================

def find_overlap_start(prev_seg: str, next_seg: str) -> Tuple[int, int]:
    """
    prev_seg(前半)末尾と next_seg(後半)先頭の共通部を探す。
    ロジック：

      1. head_phrase = 後半の「最初の数行」
      2. prev_tail   = 前半の「最後の OVERLAP_CHARS 文字」
      3. SequenceMatcher(head_phrase, prev_tail) で
         head_phrase が prev_tail のどこに一番よくはまりそうか探索。
      4. 一致ブロックのうち size >= MIN_MATCH_SIZE のものの中で、
         「head_phrase 側の先頭に近い（a が小さい）」
         ＆「prev_tail 側の開始位置（b）も小さい」
         ＆「size が大きい」ものを優先的に採用。
      5. 見つからなければ (-1, 0) を返す。

    戻り値:
        (prev_start_global, size)
        prev_start_global < 0 のときは重複なし。
    """
    if not prev_seg or not next_seg:
        return -1, 0

    head_phrase = extract_head_phrase(next_seg)
    if len(head_phrase) < MIN_MATCH_SIZE:
        return -1, 0

    prev_tail = prev_seg[-OVERLAP_CHARS:]

    sm = SequenceMatcher(None, head_phrase, prev_tail)
    blocks = sm.get_matching_blocks()

    candidates: List[Tuple[int, int, int]] = []
    for b in blocks:
        if b.size >= MIN_MATCH_SIZE:
            # a: head_phrase 側の開始位置
            # b: prev_tail 側の開始位置
            candidates.append((b.a, b.b, b.size))

    if not candidates:
        return -1, 0

    # 優先順位：
    #  1) head_phrase 側の開始 a が小さい   → head_phrase の先頭に近い一致
    #  2) prev_tail 側の開始 b が小さい     → prev_tail の中でも早い位置
    #  3) 一致長 size が長い
    a, b, size = sorted(candidates, key=lambda t: (t[0], t[1], -t[2]))[0]

    # head_phrase[a] と prev_tail[b] が対応しているので、
    # 「head_phrase 全体の先頭」に合わせるなら、
    # prev_tail 側では (b - a) 付近が本来の開始と考えられる。
    start_in_tail = max(0, b - a)

    # prev_seg 全体での位置へ変換
    global_prev_start = len(prev_seg) - len(prev_tail) + start_in_tail

    return global_prev_start, size


# ============================================================
# 結合処理（前半側だけマーク）
# ============================================================

def build_merged_text(text: str) -> str:
    """
    つなぎ目ごとに前半セグメント末尾の重複開始位置を探し、
    そこに「-----ここから重複-----」を挿入する。

    後半セグメントはそのまま（「ここまで重複」は付けない）。
    """
    segments, markers = split_by_markers(text)

    if len(segments) <= 1 or not markers:
        # マーカーがない場合はそのまま返す
        return text

    merged: List[str] = []

    # 最初のセグメントを一旦そのまま入れておく
    merged.append(segments[0])

    for idx, marker in enumerate(markers):
        prev_seg = segments[idx]
        next_seg = segments[idx + 1]

        # 重複開始位置を探索
        start_pos, size = find_overlap_start(prev_seg, next_seg)

        if start_pos < 0 or size <= 0:
            # 重複が検出できなかった場合：
            #   直前に入れていた前セグメントはそのまま、
            #   マーカーと後セグメントをそのまま繋ぐ
            merged.append("\n" + marker["marker_text"] + "\n")
            merged.append(next_seg)
            continue

        # 前半側の重複開始位置でタグを挿入した新しい前セグメントを作る
        new_prev = (
            prev_seg[:start_pos]
            + "\n" + BEGIN_TAG + "\n"
            + prev_seg[start_pos:]
        )

        # 直前の merged[-1]（古い prev_seg）を差し替え
        merged[-1] = new_prev

        # マーカー行
        merged.append("\n" + marker["marker_text"] + "\n")

        # 後半側は何も削らず、そのまま続ける
        merged.append(next_seg)

    return "".join(merged)


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="📝 文字起こし結合（重複検出：後半の最初の数行ベース）",
    page_icon="📝",
    layout="wide",
)

st.title("📝 重複箇所検出（後半の最初の数行から重複開始を検出）")

st.markdown(
    """
- 後半セグメントの **「最初の数行」** をキーにして、  
  前半セグメントの **「最後の2分相当」** の中から  
  もっともよく一致する位置を探し、前半側にだけ  
  `-----ここから重複-----` を挿入します。

- 後半セグメントには「ここまで重複部分」は挿入しません。
"""
)

uploaded_files = st.file_uploader(
    "文字起こしテキスト (.txt) をドラッグ＆ドロップしてください（複数可）",
    type=["txt"],
    accept_multiple_files=True,
)

run = st.button("▶️ 重複箇所検出を実行する", type="primary")

if run:
    if not uploaded_files:
        st.warning("先にファイルをアップロードしてください。")
    else:
        for up in uploaded_files:
            name = up.name
            raw = up.read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("cp932", errors="replace")

            merged = build_merged_text(text)

            st.subheader(f"📄 ファイル: {name}")
            with st.expander("📘 結果プレビュー", expanded=True):
                st.text(merged)

            st.download_button(
                "📥 結合テキストをダウンロード (.txt)",
                merged.encode("utf-8"),
                file_name=f"{name.rsplit('.',1)[0]}_重複箇所検出.txt",
                mime="text/plain",
            )

        st.success("処理が完了しました。")
