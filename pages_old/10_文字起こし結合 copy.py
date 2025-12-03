# -*- coding: utf-8 -*-
# pages/10_文字起こし結合.py
#
# 文字起こしテキスト(.txt)をアップロードして、
# 「----- ここがつなぎ目です（xxx と次のファイルの間）-----」
# という行を境目として前後のオーバーラップ部分を推定・表示するページ。
#
# ・つなぎ目ごとに、前のファイル側と次のファイル側の
#   「似ていそうな部分（オーバーラップ候補）」を抜き出して表示
# ・オーバーラップ情報レポートを .txt でダウンロード
#
# ・結合テキスト側では、テキストは削除せず、
#   - 前のセグメントの「重複開始位置」に
#       -----ここから重複-----
#   - 後ろのセグメントの「重複終了位置」に
#       -----ここまでが重複部分-----
#   を差し込む。
#
#   （重複部分自体は削らず、そのまま全文を残す）
#
# ★ オーバーラップとして見る範囲（前後の文字数）をスライダーで調整可能。

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple

import streamlit as st


# ============================================================
# 設定値
# ============================================================

# 「オーバーラップあり」とみなす最低一致長（文字数）
MIN_MATCH_SIZE = 50

# つなぎ目を示す行のパターン
# 例:
# ----- ここがつなぎ目です（音声（三春町）_part000_00000-02000.mp3 と次のファイルの間）-----
MARKER_PATTERN = re.compile(
    r"^-{3,}\s*ここがつなぎ目です（(.*?)）.*$",
    re.MULTILINE,
)

# 重複部分のマーカー文字列
OVERLAP_START = "-----ここから重複-----"
OVERLAP_END   = "-----ここまでが重複部分-----"
NO_OVERLAP_MARK = "[-----重複はありませんでした-----]"


# ============================================================
# ユーティリティ
# ============================================================

def split_by_markers(text: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    1つの文字起こしテキスト全体を、
    「つなぎ目行」で分割して、セグメントとマーカー情報を返す。

    例:
        [seg0][marker0][seg1][marker1][seg2]
    → segments = [seg0, seg1, seg2]
      markers  = [marker0, marker1]
    """
    segments: List[str] = []
    markers: List[Dict[str, Any]] = []

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

    # 最後のセグメント
    segments.append(text[prev_end:])

    return segments, markers


def find_overlap(prev_seg: str, next_seg: str, overlap_chars: int) -> Dict[str, Any]:
    """
    前後2つのセグメントからオーバーラップしていそうな部分を推定する。
    overlap_chars: 前後それぞれ何文字分を「オーバーラップ候補」として見るか。

    返り値:
        {
            "similarity": float,
            "match_size": int,
            "prev_overlap": str or "",
            "next_overlap": str or "",
            "prev_context": str or "",
            "next_context": str or "",
            "prev_start_idx": int or None,  # 前セグメント内での重複開始位置
            "next_start_idx": int or None,  # 後セグメント内での重複開始位置
            "next_end_idx": int or None,    # 後セグメント内での重複終了位置
        }
    """
    # 前セグメントの末尾 overlap_chars 文字、次セグメントの先頭 overlap_chars 文字のみを見る
    prev_tail = prev_seg[-overlap_chars:]
    next_head = next_seg[:overlap_chars]

    tail_offset = len(prev_seg) - len(prev_tail)  # prev_tail の開始位置（prev_seg 内）

    sm = SequenceMatcher(None, prev_tail, next_head)
    ratio = sm.quick_ratio()

    blocks = sm.get_matching_blocks()
    if not blocks:
        return {
            "similarity": ratio,
            "match_size": 0,
            "prev_overlap": "",
            "next_overlap": "",
            "prev_context": "",
            "next_context": "",
            "prev_start_idx": None,
            "next_start_idx": None,
            "next_end_idx": None,
        }

    # 最長一致ブロックを採用
    best = max(blocks, key=lambda b: b.size)
    if best.size < MIN_MATCH_SIZE:
        return {
            "similarity": ratio,
            "match_size": best.size,
            "prev_overlap": "",
            "next_overlap": "",
            "prev_context": "",
            "next_context": "",
            "prev_start_idx": None,
            "next_start_idx": None,
            "next_end_idx": None,
        }

    a, b, size = best.a, best.b, best.size  # a: prev_tail 内, b: next_head 内

    prev_overlap = prev_tail[a: a + size]
    next_overlap = next_head[b: b + size]

    # グローバルな開始/終了位置（セグメント全体に対する index）
    prev_start_idx = tail_offset + a
    next_start_idx = b
    next_end_idx = b + size

    # ========= レポート用コンテキスト（「2〜3行分」イメージ） =========
    ctx_margin = 200  # 周辺を「2〜3行」程度出したいイメージ

    # ----- 前のセグメント側：文の切れ目を優先して前文脈を取る -----
    prev_before_full = prev_tail[:a]
    last_punct_pos = -1
    for ch in ["。", "？", "！", "\n"]:
        p = prev_before_full.rfind(ch)
        if p > last_punct_pos:
            last_punct_pos = p

    if last_punct_pos != -1 and a - (last_punct_pos + 1) <= ctx_margin * 2:
        prev_start_ctx_local = last_punct_pos + 1
    else:
        prev_start_ctx_local = max(a - ctx_margin, 0)

    prev_before = prev_tail[prev_start_ctx_local:a]

    # ----- 次のセグメント側：後ろもできるだけ文単位で見せる -----
    next_after_full = next_head[b + size:]
    first_punct_rel = len(next_after_full)
    for ch in ["。", "？", "！", "\n"]:
        p = next_after_full.find(ch)
        if p != -1 and p < first_punct_rel:
            first_punct_rel = p + 1  # 記号も含めて表示

    if first_punct_rel != len(next_after_full) and first_punct_rel <= ctx_margin * 2:
        next_end_ctx_local = b + size + first_punct_rel
    else:
        next_end_ctx_local = min(b + size + ctx_margin, len(next_head))

    next_after = next_head[b + size: next_end_ctx_local]

    prev_context = (
        prev_before
        + "\n\n"
        + "「ここから重複」\n"
        + prev_overlap
    )

    next_context = (
        next_overlap
        + "\n\n"
        + "「ここから，新しい文章」\n"
        + next_after
    )

    return {
        "similarity": ratio,
        "match_size": size,
        "prev_overlap": prev_overlap,
        "next_overlap": next_overlap,
        "prev_context": prev_context,
        "next_context": next_context,
        "prev_start_idx": prev_start_idx,
        "next_start_idx": next_start_idx,
        "next_end_idx": next_end_idx,
    }


def apply_insertions(base: str, inserts: List[Tuple[int, str]]) -> str:
    """
    1つのセグメント文字列 base に対して、
    [(位置, 挿入文字列), ...] を index 昇順で適用する。
    （後ろからやると index がずれないが、ここでは offset を足しながら前から適用）
    """
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


def build_report_and_merged_text(text: str, overlap_chars: int) -> Tuple[str, str]:
    """
    元テキストから、オーバーラップ情報レポートと
    「重複部分にマーカーを挿入した結合テキスト」を生成する。

    overlap_chars: 各つなぎ目で、前後それぞれ何文字ぶんを「オーバーラップ候補」として見るか。

    ■ 結合テキストの組み立て方（重要）
    - segments, markers に分割
    - つなぎ目ごと（idx）に find_overlap をかけ、
        prev_start_idx（前セグメント中の重複開始位置）
        next_end_idx （後セグメント中の重複終了位置）
      を使って、
        segments[idx]   の prev_start_idx の前に OVERLAP_START を挿入
        segments[idx+1] の next_end_idx の後に OVERLAP_END   を挿入
      という「挿入指示」をまず集める。
    - あとで各セグメントごとに apply_insertions してから、
      seg0 + marker0 + seg1 + marker1 + seg2 + ... の順で再結合する。
    - 重複が見つからなかったつなぎ目には、
      marker 行の直後に NO_OVERLAP_MARK を差し込む。
    """
    segments, markers = split_by_markers(text)

    if len(segments) <= 1 or not markers:
        report = (
            "つなぎ目のマーカー行（"
            "\"----- ここがつなぎ目です（… と次のファイルの間）-----\"）"
            "が見つかりませんでした。"
        )
        return report, text

    n_seg = len(segments)
    n_mark = len(markers)

    # セグメントごとの挿入指示リスト
    seg_insertions: List[List[Tuple[int, str]]] = [[] for _ in range(n_seg)]
    # 各つなぎ目で重複が見つかったかどうか
    has_overlap: List[bool] = [False] * n_mark

    lines_report: List[str] = []

    lines_report.append("【つなぎ目オーバーラップ推定レポート】")
    lines_report.append("")
    lines_report.append(f"セグメント数: {n_seg} / マーカー数: {n_mark}")
    lines_report.append(f"オーバーラップとして見る範囲: 前後それぞれ {overlap_chars} 文字")
    lines_report.append("")

    # 1) まず各つなぎ目ごとにオーバーラップを推定し、
    #    レポートと「挿入位置」を決める
    for idx, marker in enumerate(markers):
        prev_seg = segments[idx]
        next_seg = segments[idx + 1]

        ol = find_overlap(prev_seg, next_seg, overlap_chars)

        lines_report.append("=" * 70)
        lines_report.append(f"■ つなぎ目 {idx + 1} : {marker['file_name']}")
        lines_report.append(marker["marker_text"])
        lines_report.append(f"- 類似度 (quick_ratio): {ol['similarity']:.3f}")
        lines_report.append(f"- 最長一致長: {ol['match_size']} 文字")
        lines_report.append("")

        if ol["match_size"] >= MIN_MATCH_SIZE and ol["prev_overlap"]:
            has_overlap[idx] = True

            # レポート用抜粋
            lines_report.append("[前のセグメント側 抜粋]")
            lines_report.append(ol["prev_context"])
            lines_report.append("")
            lines_report.append("[次のセグメント側 抜粋]")
            lines_report.append(ol["next_context"])
            lines_report.append("")

            # ===== 挿入位置の決定 =====
            prev_start_idx = ol["prev_start_idx"]
            next_end_idx = ol["next_end_idx"]

            if prev_start_idx is not None:
                # 前セグメントの重複開始位置の直前に「ここから重複」を挿入
                seg_insertions[idx].append(
                    (prev_start_idx, "\n" + OVERLAP_START + "\n")
                )

            if next_end_idx is not None:
                # 後ろセグメントの重複終了位置の直後に「ここまでが重複部分」を挿入
                seg_insertions[idx + 1].append(
                    (next_end_idx, "\n" + OVERLAP_END + "\n")
                )
        else:
            # 有意なオーバーラップなし
            lines_report.append(
                "⇒ 有意なオーバーラップは検出されませんでした（しきい値未満／沈黙など）。"
            )
            lines_report.append("")

    # 2) セグメントごとに挿入指示を適用して modified_segments を作る
    modified_segments: List[str] = []
    for i, seg in enumerate(segments):
        modified = apply_insertions(seg, seg_insertions[i])
        modified_segments.append(modified)

    # 3) 最後に seg0 + marker0 + (必要ならNO_OVERLAP_MARK) + seg1 + … で結合
    merged_parts: List[str] = []
    for i in range(n_seg):
        merged_parts.append(modified_segments[i])
        if i < n_mark:
            merged_parts.append("\n")
            merged_parts.append(markers[i]["marker_text"])
            merged_parts.append("\n")
            if not has_overlap[i]:
                # 重複なしの場合のみメッセージを付ける
                merged_parts.append(NO_OVERLAP_MARK + "\n")

    merged_text = "".join(merged_parts)
    report_text = "\n".join(lines_report)

    return report_text, merged_text


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="📝 文字起こし結合（オーバーラップ検出）",
    page_icon="📝",
    layout="wide",
)

st.title("📝 文字起こし結合（オーバーラップ検出付き）")

st.markdown(
    """
長時間の会議音声を分割して文字起こししたテキストを対象に、  
**「----- ここがつなぎ目です（… と次のファイルの間）-----」** という行を境目として、

- 境目の前後で **オーバーラップしていそうな部分** を推定して表示  
- 各境目ごとの **オーバーラップレポート (.txt)** を作成  
- 結合テキストでは、テキストは削除せず、  
  - 前のセグメントの重複開始位置に `ここから重複`  
  - 次のセグメントの重複終了位置に `ここまでが重複部分`  
  を差し込みます。
"""
)

# オーバーラップとして見る範囲（前後の文字数）をスライダーで指定
overlap_chars = st.slider(
    "オーバーラップとして見る範囲（前後の文字数）",
    min_value=500,
    max_value=6000,
    step=500,
    value=2000,  # 2分オーバーラップをざっくり想定した初期値
    help="音声ファイルのオーバーラップ時間が長いときは大きめに、短いときは小さめにしてください。",
)

uploaded_files = st.file_uploader(
    "文字起こしテキスト (.txt) をドラッグ＆ドロップしてください（複数可）",
    type=["txt"],
    accept_multiple_files=True,
)

run = st.button("▶️ オーバーラップ検出を実行する", type="primary")

if run:
    if not uploaded_files:
        st.warning("先に .txt ファイルをアップロードしてください。")
    else:
        for up in uploaded_files:
            st.subheader(f"📄 ファイル: {up.name}")

            raw = up.read()
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("cp932", errors="replace")

            report_text, merged_text = build_report_and_merged_text(text, overlap_chars)

            # ---- 画面表示 ----
            with st.expander("🔍 オーバーラップ推定レポート（画面表示）", expanded=True):
                st.text(report_text)

            # ---- ダウンロードボタン ----
            col1, col2 = st.columns(2)

            with col1:
                report_bytes = report_text.encode("utf-8")
                st.download_button(
                    label="📥 オーバーラップレポートをダウンロード (.txt)",
                    data=report_bytes,
                    file_name=f"{up.name.rsplit('.', 1)[0]}_overlap_report.txt",
                    mime="text/plain",
                )

            with col2:
                merged_bytes = merged_text.encode("utf-8")
                st.download_button(
                    label="📥 重複マーク付き結合テキストをダウンロード (.txt)",
                    data=merged_bytes,
                    file_name=f"{up.name.rsplit('.', 1)[0]}_merged_with_overlap_marks.txt",
                    mime="text/plain",
                )

        st.success("処理が完了しました。『ここから重複』『ここまでが重複部分』の位置を確認してみてください。")
