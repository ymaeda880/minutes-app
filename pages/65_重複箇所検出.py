# -*- coding: utf-8 -*-
# pages/03_重複箇所検出2.py
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
from typing import List, Dict, Tuple, Any

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

# 一致とみなす最低文字数（デフォルト値）※サイドバーで変更可能
DEFAULT_MIN_MATCH_SIZE = 20

# キーになるフレーズを「先頭1文字ずつずらして」試す最大回数
HEAD_SHIFT_TRIES = 3  # 例：今日は会議を→日は会議を→は会議を→会議を… のイメージ

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


def _match_with_phrase(
    prev_seg: str,
    phrase: str,
    min_match_size: int,
    use_autojunk: bool,
) -> Tuple[int, int]:
    """
    1つの「キーになるフレーズ（phrase）」に対して、
    前半セグメント prev_seg の末尾 OVERLAP_CHARS とマッチングを行い、
    一致開始位置と一致長を返す（見つからなければ -1, 0）。
    """
    if not prev_seg or not phrase:
        return -1, 0

    if len(phrase) < min_match_size:
        return -1, 0

    # 前半側の末尾
    prev_tail = prev_seg[-OVERLAP_CHARS:]

    # 正規化テキスト作成
    norm_head = normalize_text(phrase)
    norm_prev = normalize_text(prev_tail)

    if len(norm_head) < min_match_size:
        return -1, 0

    # 類似ブロックを抽出（★ autojunk を UI から制御）
    sm = SequenceMatcher(None, norm_head, norm_prev, autojunk=use_autojunk)
    blocks = sm.get_matching_blocks()

    # 候補抽出
    cand = [(b.a, b.b, b.size) for b in blocks if b.size >= min_match_size]
    if not cand:
        return -1, 0

    # 優先順位：
    #   head_phrase 側の開始が先 → prev_tail 側の開始が先 → size が大きい
    a, b, size = sorted(cand, key=lambda t: (t[0], t[1], -t[2]))[0]

    # ======== 元テキストの index に戻すための処理 ========

    # 正規化前の phrase と prev_tail とのマッピングを作る
    def build_index_map(raw: str, norm: str):
        """
        正規化前 raw の各文字が、正規化後 norm の
        どの index に対応するかを返す map
        """
        mapping = []
        j = 0
        for i, ch in enumerate(raw):
            # 正規化で消える文字は mapping に入れない
            ch_norm = normalize_text(ch)
            if ch_norm == "":
                continue
            if j < len(norm):
                mapping.append((j, i))
                j += 1
        return mapping

    head_map = build_index_map(phrase, norm_head)
    prev_map = build_index_map(prev_tail, norm_prev)

    # a, b は正規化後の index なので raw 側 index に逆変換する
    def mapped_index(mapping, idx_norm):
        # idx_norm に最も近い mapping の raw index を返す
        candidates = [raw_idx for norm_idx, raw_idx in mapping if norm_idx == idx_norm]
        if candidates:
            return candidates[0]
        # 直接一致がなければ近いものを探す
        nearest = None
        best_dist = 10**9
        for norm_idx, raw_idx in mapping:
            d = abs(norm_idx - idx_norm)
            if d < best_dist:
                best_dist = d
                nearest = raw_idx
        return nearest

    # 元のテキストでの開始位置
    head_raw_start = mapped_index(head_map, a)
    prev_raw_start = mapped_index(prev_map, b)

    # 「head の先頭と align する」前提で補正
    start_in_tail = max(0, prev_raw_start - head_raw_start)

    global_prev_start = len(prev_seg) - len(prev_tail) + start_in_tail

    return global_prev_start, size


def find_overlap_start(
    prev_seg: str,
    next_seg: str,
    min_match_size: int,
    use_autojunk: bool,
) -> Tuple[int, int, str, List[str], str]:
    """
    1. 後半セグメントから head_phrase（最初の数行）を作る。
    2. strip + rstrip で末尾の句読点を落として base_head にする。
    3. base_head と、そこから 1文字ずつ先頭を削ったフレーズを順に試す。
    4. どれかで見つかれば、その位置と長さ・base_head・shifted・matched_phrase を返す。
    """

    if not prev_seg or not next_seg:
        return -1, 0, "", [], ""

    # まず「生の head_phrase（最初の数行）」を作る
    raw_head = extract_head_phrase(next_seg)
    if not raw_head:
        return -1, 0, "", [], ""

    # 前後の空白を削り、末尾の句読点を落としてから使う
    base_head = raw_head.strip()
    base_head = base_head.rstrip("。？！!？，、,.")

    if not base_head:
        return -1, 0, "", [], ""

    # 試すフレーズの候補列を作る
    candidates: List[str] = []
    seen: set[str] = set()
    shifted_heads: List[str] = []

    def add_candidate(s: str, is_shifted: bool = False):
        if not s:
            return
        if s in seen:
            return
        seen.add(s)
        candidates.append(s)
        if is_shifted:
            shifted_heads.append(s)

    # 1. そのままの head_phrase
    add_candidate(base_head, is_shifted=False)

    # 2. 先頭を 1 文字ずつずらしたバリエーション（最大 HEAD_SHIFT_TRIES 回）
    current = base_head
    for _ in range(HEAD_SHIFT_TRIES):
        if len(current) <= 1:
            break
        current = current[1:]
        add_candidate(current, is_shifted=True)

    # 順番にマッチングを試す
    matched_phrase = ""
    for phrase in candidates:
        start_pos, size = _match_with_phrase(
            prev_seg, phrase, min_match_size, use_autojunk
        )
        if start_pos >= 0 and size > 0:
            matched_phrase = phrase
            return start_pos, size, base_head, shifted_heads, matched_phrase

    # どのフレーズでも見つからなかった
    return -1, 0, base_head, shifted_heads, ""


# ============================================================
# 結合処理（前半側だけマーク）＋ログ出力
# ============================================================

def build_merged_text(
    text: str,
    min_match_size: int,
    use_autojunk: bool,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    つなぎ目ごとに前半セグメント末尾の重複開始位置を探し、
    そこに「-----ここから重複-----」を挿入する。

    - merged_text: マーカー挿入済みの全文
    - logs: つなぎ目ごとの検出結果（成功/失敗を含む）

    後半セグメントはそのまま（「ここまで重複」は付けない）。
    """
    segments, markers = split_by_markers(text)

    if len(segments) <= 1 or not markers:
        # マーカーがない場合はそのまま返す（ログは空）
        return text, []

    merged: List[str] = []
    logs: List[Dict[str, Any]] = []

    # 最初のセグメントを一旦そのまま入れておく
    merged.append(segments[0])

    for idx, marker in enumerate(markers):
        prev_seg = segments[idx]
        next_seg = segments[idx + 1]

        # 重複開始位置を探索（内部でキー文のスライドも行う）
        start_pos, size, base_head, shifted_heads, matched_phrase = find_overlap_start(
            prev_seg, next_seg, min_match_size, use_autojunk
        )

        if start_pos < 0 or size <= 0:
            # ★ 失敗ログを追加
            logs.append(
                {
                    "つなぎ目番号": idx,
                    "ファイル名": marker.get("file_name", ""),
                    "検出結果": "見つからず",
                    "開始位置": None,
                    "一致文字数": 0,
                    "head_phrase": base_head,
                    "shifted_phrases": shifted_heads,
                    "matched_phrase": "",
                }
            )

            merged.append("\n" + marker["marker_text"] + "\n")
            merged.append(next_seg)
            continue

        # ★ 成功ログを追加
        logs.append(
            {
                "つなぎ目番号": idx,
                "ファイル名": marker.get("file_name", ""),
                "検出結果": "検出",
                "開始位置": start_pos,
                "一致文字数": size,
                "head_phrase": base_head,
                "shifted_phrases": shifted_heads,
                "matched_phrase": matched_phrase,
            }
        )

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

    return "".join(merged), logs


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="📝 文字起こし結合（重複検出：後半の最初の数行ベース）",
    page_icon="📝",
    layout="wide",
)

st.title("📝 重複箇所検出")

st.markdown(
    """
- 後半セグメントの **「最初の数行」** をキーにして、  
  前半セグメントの **「最後の2分相当」** の中から  
  もっともよく一致する位置を探し、前半側にだけ  
  `-----ここから重複-----` を挿入します。

- 後半セグメントには「ここまで重複部分」は挿入しません。
"""
)

# 🔧 サイドバー設定
with st.sidebar:
    st.header("検出パラメータ")

    min_match_size = st.slider(
        "一致とみなす最低文字数（MIN_MATCH_SIZE）",
        min_value=5,
        max_value=40,
        value=DEFAULT_MIN_MATCH_SIZE,
        step=1,
        help="重複とみなす連続一致の最低文字数です。値を小さくすると検出がゆるくなり、大きくすると厳しくなります。",
    )

    autojunk_option = st.radio(
        "autojunk（SequenceMatcher 自動ジャンク判定）",
        options=["ON（デフォルト）", "OFF（短い文でも精確に）"],
        index=0,
        help=(
            "ON: Python標準の自動ジャンク判定を使います（高速だが短いフレーズを落とすことがあります）。\n"
            "OFF: 短いフレーズの一致も取りこぼしにくくなりますが、わずかに遅くなります。"
        ),
    )
    use_autojunk = autojunk_option.startswith("ON")

# 🔽 ロジック説明（初期は畳んでおく）
with st.expander("🔍 このツールの重複検出ロジック（概要）", expanded=False):
    st.markdown(
        f"""
1. **セグメント分割**  
   - テキスト全体を「--- ここがつなぎ目です（…）」というマーカー行で分割し、  
     `[前半セグメント][マーカー][後半セグメント]...` という形に分けます。

2. **後半側の「最初の数行」を抽出**  
   - 各つなぎ目について、後半セグメントの先頭から `HEAD_CHARS` 文字を取り出し、  
     「。」「？」「！」「改行」などで区切って **最大 `HEAD_SENTENCES` 文** までを  
     head_phrase として使います（末尾の句読点は除去）。

3. **キー文のスライド**  
   - head_phrase そのものに加え、先頭を1文字ずつ削った `{HEAD_SHIFT_TRIES}` 個の候補も試します。

4. **前半側の「最後の2分相当」を取得**  
   - 前半セグメントの末尾から `OVERLAP_CHARS` 文字だけを切り出し、これを比較対象とします。

5. **正規化して類似度を計算**  
   - 全角スペース・改行・句読点・半角スペースを除去してから、`SequenceMatcher` で一致ブロックを調べます。  
   - autojunk の ON/OFF はサイドバーから切り替えられます。

6. **もっとも自然な一致位置を選び、タグを挿入**  
   - 連続一致が `{min_match_size}` 文字以上ある部分から、自然な位置を1箇所選び、  
     前半セグメント側にのみ `-----ここから重複-----` 行を挿入します。
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

            # ★ 結合テキストとログを受け取る
            merged, logs = build_merged_text(text, min_match_size, use_autojunk)

            st.subheader(f"📄 ファイル: {name}")

            # 🔍 つなぎ目ごとの検出結果（概要）
            with st.expander("🔍 つなぎ目ごとの検出ログ（概要）", expanded=True):
                if logs:
                    st.table(
                        [
                            {
                                "つなぎ目番号": item["つなぎ目番号"],
                                "ファイル名": item["ファイル名"],
                                "検出結果": item["検出結果"],
                                "開始位置": item["開始位置"],
                                "一致文字数": item["一致文字数"],
                            }
                            for item in logs
                        ]
                    )
                else:
                    st.info("つなぎ目マーカーが見つからなかったため、ログはありません。")

            # 🧩 head_phrase と shifted_phrases の詳細
            with st.expander("🧩 head_phrase と shifted_phrases の詳細", expanded=False):
                if not logs:
                    st.info("ログがないため表示できる情報がありません。")
                else:
                    for item in logs:
                        st.markdown(
                            f"### 🔹 つなぎ目 {item['つなぎ目番号']} — {item['検出結果']}"
                        )
                        st.markdown("**🔸 head_phrase（整形後のキー文）**")
                        st.code(item["head_phrase"] or "（空です）")

                        st.markdown("**🔸 shifted_phrases（先頭を1文字ずつずらした候補）**")
                        if item["shifted_phrases"]:
                            for s in item["shifted_phrases"]:
                                st.code(s)
                        else:
                            st.write("（shifted_phrases はありません）")

                        st.markdown("**🔸 matched_phrase（実際にマッチしたフレーズ）**")
                        st.code(item["matched_phrase"] or "（マッチなし）")

                        st.markdown("---")

            # 📘 テキスト結果プレビュー
            with st.expander("📘 結果プレビュー", expanded=False):
                st.text(merged)

            st.download_button(
                "📥 結合テキストをダウンロード (.txt)",
                merged.encode("utf-8"),
                file_name=f"{name.rsplit('.',1)[0]}_重複箇所検出.txt",
                mime="text/plain",
            )

        st.success("処理が完了しました。")
