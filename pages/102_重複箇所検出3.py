# -*- coding: utf-8 -*-
# pages/12_重複箇所検出3.py
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
MIN_MATCH_SIZE = 15

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

# ============================================================
# 重複開始位置の探索（前半側のみ）
# ============================================================

# 正規化：句読点・改行・スペース除去（必要ならさらに追加可）
def normalize_text(s: str) -> str:
    # 全角スペース → 空
    s = s.replace("　", "")
    # 改行 → 空
    s = s.replace("\n", "")
    # 句読点・記号を除去
    s = re.sub(r"[、。！？,.!?]", "", s)
    # 余分なスペース除去
    s = s.replace(" ", "")
    return s


SIMILARITY_THRESHOLD = 0.75  # 類似度しきい値（0〜1）


def is_similar(a: str, b: str, threshold: float = SIMILARITY_THRESHOLD) -> bool:
    """2つの文字列の類似度がしきい値以上かどうかを判定"""
    return SequenceMatcher(None, a, b).ratio() >= threshold


def find_overlap_start(prev_seg: str, next_seg: str) -> Tuple[int, int]:
    """
    改良版：
      - MIN_MATCH_SIZE = 20 に変更（正規化後 head の最低長チェック用）
      - 正規化した head_phrase と prev_tail でスライドしながら類似度を計算
      - 類似度が SIMILARITY_THRESHOLD 以上で最大の位置を重複開始とみなす
      - その位置を元の非正規化テキストの index に戻して返す
    """

    if not prev_seg or not next_seg:
        return -1, 0

    from_head = extract_head_phrase(next_seg)
    if not from_head:
        return -1, 0

    # 前半側の末尾
    prev_tail = prev_seg[-OVERLAP_CHARS:]

    # 正規化テキスト作成
    norm_head = normalize_text(from_head)
    norm_prev = normalize_text(prev_tail)

    # head 側が短すぎる場合は判定しない
    if len(norm_head) < MIN_MATCH_SIZE:
        return -1, 0

    # ======== 正規化テキスト上での最良位置を探索（スライド窓） ========
    Lh = len(norm_head)
    if len(norm_prev) < Lh:
        return -1, 0

    best_score = 0.0
    best_b_norm = -1

    for b_norm in range(0, len(norm_prev) - Lh + 1):
        window = norm_prev[b_norm:b_norm + Lh]
        score = SequenceMatcher(None, norm_head, window).ratio()
        if score > best_score:
            best_score = score
            best_b_norm = b_norm

    # 類似度がしきい値を下回るなら重複なし
    if best_b_norm < 0 or best_score < SIMILARITY_THRESHOLD:
        return -1, 0

    # ======== 元テキストの index に戻すための処理 ========

    def build_index_map(raw: str, norm: str):
        """
        正規化前 raw の各文字が、正規化後 norm の
        どの index に対応するかの map を作る
        （norm_idx → raw_idx の対応ペアのリスト）
        """
        mapping: List[Tuple[int, int]] = []
        j = 0  # norm 側の index
        for i, ch in enumerate(raw):
            ch_norm = normalize_text(ch)
            if ch_norm == "":
                # 正規化で消える文字はスキップ
                continue
            if j < len(norm):
                mapping.append((j, i))
                j += 1
        return mapping

    head_map = build_index_map(from_head, norm_head)
    prev_map = build_index_map(prev_tail, norm_prev)

    def mapped_index(mapping: List[Tuple[int, int]], idx_norm: int) -> int:
        """
        正規化後の index から、対応する（または一番近い）元テキストの index を取得
        """
        candidates = [raw_idx for norm_idx, raw_idx in mapping if norm_idx == idx_norm]
        if candidates:
            return candidates[0]

        # 直接一致がなければ最も近い norm_idx を探す
        nearest = None
        best_dist = 10**9
        for norm_idx, raw_idx in mapping:
            d = abs(norm_idx - idx_norm)
            if d < best_dist:
                best_dist = d
                nearest = raw_idx
        return nearest if nearest is not None else 0

    # head 側は「正規化文字列の先頭」を基準にする
    head_raw_start = mapped_index(head_map, 0)
    # prev 側は「最良一致窓の先頭」best_b_norm を基準
    prev_raw_start = mapped_index(prev_map, best_b_norm)

    # 「head の先頭と align する」前提で補正
    start_in_tail = max(0, prev_raw_start - head_raw_start)

    global_prev_start = len(prev_seg) - len(prev_tail) + start_in_tail

    # size はあくまで参考値（類似度 × head長）として返す
    approx_size = int(len(norm_head) * best_score)

    return global_prev_start, approx_size

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

# 🔽 ロジック説明（初期は畳んでおく）
with st.expander("🔍 このツールの重複検出ロジック（概要）", expanded=False):
    st.markdown(
        """
1. **セグメント分割**  
   - テキスト全体を「--- ここがつなぎ目です（…）」というマーカー行で分割し、  
     `[前半セグメント][マーカー][後半セグメント]...` という形に分けます。

2. **後半側の「最初の数行」を抽出**  
   - 各つなぎ目について、後半セグメントの先頭から `HEAD_CHARS` 文字を取り出し、  
     「。」「？」「！」「改行」などで区切って **最大 `HEAD_SENTENCES` 文** までを  
     `head_phrase` として使います。

3. **前半側の「最後の2分相当」を取得**  
   - 前半セグメントの末尾から `OVERLAP_CHARS` 文字だけを切り出し、  
     これを `prev_tail` として比較対象にします。

4. **正規化して類似度を計算**  
   - `head_phrase` と `prev_tail` から  
     - 全角スペース  
     - 改行  
     - 句読点（、。！？,.!?）  
     - 半角スペース  
     を取り除いた **正規化テキスト**（`norm_head`, `norm_prev`）を作ります。  
   - `norm_head` が短すぎる場合は（`MIN_MATCH_SIZE = 20` 未満）そのつなぎ目はスキップします。

5. **類似度しきい値で最良の重複位置を探す**  
   - `norm_head` の長さと同じ長さの「窓」を `norm_prev` の中でスライドさせ、  
     各位置について `SequenceMatcher` の `ratio()` を用いて **類似度スコア** を計算します。  
   - 類似度がもっとも高い位置を取り出し、  
     そのスコアが `SIMILARITY_THRESHOLD`（例：0.82）以上であれば、  
     **「ここから重複している」と判定**します。  
   - しきい値未満であれば、そのつなぎ目では重複なしとみなします。

6. **正規化前の位置にマッピングしてタグを挿入**  
   - 元のテキストと正規化テキストの対応関係（インデックスマップ）を作り、  
     正規化テキスト上で見つかった一致開始位置を **元の非正規化テキスト上の index に戻します**。  
   - 前半セグメントのその位置に  
     `-----ここから重複-----`  
     という行を挿入し、後半セグメント側は一切削らずそのまま連結します。
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
