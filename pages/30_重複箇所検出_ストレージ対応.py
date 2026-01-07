# -*- coding: utf-8 -*-
# pages/30_重複箇所検出_storage対応.py
#
# ✅ storage対応版（ログイン必須）
# - ログイン確認（pages/13 と同じ）
# - Storages/<user>/minutes_app/ 配下の「連結文字起こし（combined txt）」を列挙
# - radio で 1つ選択 → 重複検出（前半側にだけ BEGIN_TAG を挿入）
# - 結果（merged txt）＋検出ログ（json）を transcript/ に保存
#
# ※ common_lib は改変しない
# ※ use_container_width は使わない（方針に従う）

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import sys

import streamlit as st

_THIS = Path(__file__).resolve()
PROJECTS_ROOT = _THIS.parents[3]

if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

from common_lib.storage.external_ssd_root import resolve_storage_subdir_root
from common_lib.auth.auth_helpers import require_login


# ============================================================
# 設定値（既存）
# ============================================================

OVERLAP_CHARS = 700
HEAD_CHARS = 400
HEAD_SENTENCES = 3
DEFAULT_MIN_MATCH_SIZE = 20
HEAD_SHIFT_TRIES = 3

MARKER_PATTERN = re.compile(
    r"^-{3,}\s*ここがつなぎ目です（(.*?)）.*$",
    re.MULTILINE,
)

BEGIN_TAG = "-----ここから重複-----"


# ============================================================
# paths（PROJECTS_ROOT 基準）
# ============================================================
STORAGE_ROOT = resolve_storage_subdir_root(
    PROJECTS_ROOT,
    subdir="Storages",
)


# ============================================================
# utils
# ============================================================
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_text(p: Path, s: str) -> None:
    safe_mkdir(p.parent)
    p.write_text(s, encoding="utf-8")


def write_json(p: Path, obj: Any) -> None:
    safe_mkdir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_log(log_path: Path, msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_mkdir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def _sanitize_username_for_path(username: str) -> str:
    u = (username or "").strip()
    if not u:
        return "anonymous"
    u = re.sub(r"[^0-9A-Za-z_-]+", "_", u).strip("_")
    return u or "anonymous"


def _human_dt(s: str | None) -> str:
    if not s:
        return "—"
    try:
        return s.replace("T", " ").replace("+00:00", "Z")
    except Exception:
        return s


# ============================================================
# storage: combined txt listing
#   Storages/<user>/minutes_app/<YYYY-MM-DD>/job_*/transcript_speaker_separated_combined/*.txt
# ============================================================
@dataclass
class CombinedItem:
    label: str
    path: Path
    job_dir: Path
    transcript_dir: Path
    job_id: str
    date: str
    created_at: Optional[str]


def _read_job_json(job_dir: Path) -> dict:
    p = job_dir / "job.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def list_combined_texts(user_dir: str) -> list[CombinedItem]:
    base = STORAGE_ROOT / user_dir / "minutes_app"
    if not base.exists():
        return []

    items: list[CombinedItem] = []

    # 日付降順
    for day_dir in sorted(base.glob("*"), reverse=True):
        if not day_dir.is_dir():
            continue

        # job降順
        for job_dir in sorted(day_dir.glob("job_*"), reverse=True):
            if not job_dir.is_dir():
                continue

            meta = _read_job_json(job_dir)
            job_id = str(meta.get("job_id") or job_dir.name)
            date = str(meta.get("date") or day_dir.name)
            created_at = meta.get("created_at")

            # 入力元：transcript_speaker_separated_combined
            combined_dir = job_dir / "transcript_speaker_separated_combined"
            if not combined_dir.exists():
                continue

            combined_files = sorted(
                combined_dir.glob("*.txt"),
                key=lambda p: p.name.lower(),
                reverse=True,
            )
            for p in combined_files:
                label = f"{date} / {job_id} / {p.name} / created={_human_dt(created_at)}"
                items.append(
                    CombinedItem(
                        label=label,
                        path=p,
                        job_dir=job_dir,
                        transcript_dir=combined_dir,
                        job_id=job_id,
                        date=date,
                        created_at=created_at,
                    )
                )

    return items



# ============================================================
# セグメント分割（既存）
# ============================================================
def split_by_markers(text: str) -> Tuple[List[str], List[Dict[str, str]]]:
    segments: List[str] = []
    markers: List[Dict[str, str]] = []
    prev_end = 0

    for m in MARKER_PATTERN.finditer(text):
        seg = text[prev_end:m.start()]
        segments.append(seg)
        markers.append({"file_name": m.group(1), "marker_text": m.group(0)})
        prev_end = m.end()

    segments.append(text[prev_end:])
    return segments, markers


# ============================================================
# 「後半の最初の数行」を抜き出す（既存）
# ============================================================
# === 追加：話者ラベル（行頭）除去 ===
SPEAKER_PREFIX_PATTERN = re.compile(
    r"""^(
        \s*                                   # 先頭空白
        (?:司会|ＭＣ|MC|進行)                  # 日本語/MC系ラベル（必要なら増やす）
        \s*[:：]\s*                            # コロン
      |
        \s*\[?\s*[sS]\s*\d+\s*\]?\s*[:：]\s*   # S12: / [s12]: / S 12 :
    )""",
    re.VERBOSE,
)

def strip_leading_speaker_labels(text: str) -> str:
    """
    セグメント先頭に付いた話者ラベル（S4: / [s4]: / 司会: 等）を剥がす。
    先頭から連続して付くケースもあるので繰り返し除去する。
    """
    if not text:
        return ""
    s = text.lstrip("\ufeff")  # 念のためBOM除去
    # 先頭行に連続で付く場合を考慮してループ
    for _ in range(5):  # 無限ループ防止
        m = SPEAKER_PREFIX_PATTERN.match(s)
        if not m:
            break
        s = s[m.end():]
    return s


def extract_head_phrase(next_seg: str) -> str:
    if not next_seg:
        return ""

    # ★ 追加：先頭の話者ラベルを剥がしてからキー文を作る
    next_seg = strip_leading_speaker_labels(next_seg)

    s = next_seg[:HEAD_CHARS]
    count = 0
    end = len(s)

    for i, ch in enumerate(s):
        if ch in "。？！\n":
            count += 1
            if count >= HEAD_SENTENCES:
                end = i + 1
                break

    return s[:end]



# ============================================================
# 正規化 & マッチング（既存）
# ============================================================
def normalize_text(s: str) -> str:
    s = s.replace("　", "")
    s = s.replace("\n", "")
    s = re.sub(r"[、。！？,.!?]", "", s)
    s = s.replace(" ", "")
    return s


def _match_with_phrase(
    prev_seg: str,
    phrase: str,
    min_match_size: int,
    use_autojunk: bool,
) -> Tuple[int, int]:
    if not prev_seg or not phrase:
        return -1, 0
    if len(phrase) < min_match_size:
        return -1, 0

    prev_tail = prev_seg[-OVERLAP_CHARS:]
    norm_head = normalize_text(phrase)
    norm_prev = normalize_text(prev_tail)

    if len(norm_head) < min_match_size:
        return -1, 0

    sm = SequenceMatcher(None, norm_head, norm_prev, autojunk=use_autojunk)
    blocks = sm.get_matching_blocks()

    cand = [(b.a, b.b, b.size) for b in blocks if b.size >= min_match_size]
    if not cand:
        return -1, 0

    a, b, size = sorted(cand, key=lambda t: (t[0], t[1], -t[2]))[0]

    def build_index_map(raw: str, norm: str):
        mapping = []
        j = 0
        for i, ch in enumerate(raw):
            ch_norm = normalize_text(ch)
            if ch_norm == "":
                continue
            if j < len(norm):
                mapping.append((j, i))
                j += 1
        return mapping

    head_map = build_index_map(phrase, norm_head)
    prev_map = build_index_map(prev_tail, norm_prev)

    def mapped_index(mapping, idx_norm):
        candidates = [raw_idx for norm_idx, raw_idx in mapping if norm_idx == idx_norm]
        if candidates:
            return candidates[0]
        nearest = None
        best_dist = 10**9
        for norm_idx, raw_idx in mapping:
            d = abs(norm_idx - idx_norm)
            if d < best_dist:
                best_dist = d
                nearest = raw_idx
        return nearest

    head_raw_start = mapped_index(head_map, a)
    prev_raw_start = mapped_index(prev_map, b)

    start_in_tail = max(0, int(prev_raw_start) - int(head_raw_start))
    global_prev_start = len(prev_seg) - len(prev_tail) + start_in_tail
    return global_prev_start, int(size)


def find_overlap_start(
    prev_seg: str,
    next_seg: str,
    min_match_size: int,
    use_autojunk: bool,
) -> Tuple[int, int, str, List[str], str]:
    if not prev_seg or not next_seg:
        return -1, 0, "", [], ""

    raw_head = extract_head_phrase(next_seg)
    if not raw_head:
        return -1, 0, "", [], ""

    base_head = raw_head.strip().rstrip("。？！!？，、,.")
    if not base_head:
        return -1, 0, "", [], ""

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

    add_candidate(base_head, is_shifted=False)

    current = base_head
    for _ in range(HEAD_SHIFT_TRIES):
        if len(current) <= 1:
            break
        current = current[1:]
        add_candidate(current, is_shifted=True)

    matched_phrase = ""
    for phrase in candidates:
        start_pos, size = _match_with_phrase(prev_seg, phrase, min_match_size, use_autojunk)
        if start_pos >= 0 and size > 0:
            matched_phrase = phrase
            return start_pos, size, base_head, shifted_heads, matched_phrase

    return -1, 0, base_head, shifted_heads, ""


def build_merged_text(
    text: str,
    min_match_size: int,
    use_autojunk: bool,
) -> Tuple[str, List[Dict[str, Any]]]:
    segments, markers = split_by_markers(text)

    if len(segments) <= 1 or not markers:
        return text, []

    merged: List[str] = []
    logs: List[Dict[str, Any]] = []

    merged.append(segments[0])

    for idx, marker in enumerate(markers):
        prev_seg = segments[idx]
        next_seg = segments[idx + 1]

        start_pos, size, base_head, shifted_heads, matched_phrase = find_overlap_start(
            prev_seg, next_seg, min_match_size, use_autojunk
        )

        if start_pos < 0 or size <= 0:
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

        new_prev = prev_seg[:start_pos] + "\n" + BEGIN_TAG + "\n" + prev_seg[start_pos:]
        merged[-1] = new_prev

        merged.append("\n" + marker["marker_text"] + "\n")
        merged.append(next_seg)

    return "".join(merged), logs


# ============================================================
# Streamlit UI（storage対応）
# ============================================================
st.set_page_config(
    page_title="📝 重複箇所検出（storage対応）",
    page_icon="📝",
    layout="wide",
)

sub = require_login(st)
if not sub:
    st.stop()
left, right = st.columns([2, 1])
with left:
    st.title("📝 重複箇所検出（ストレージ対応）")
with right:
    st.success(f"✅ ログイン中: **{sub}**")
current_user=sub


st.markdown(
    """
- **ログイン必須**（Cookie/JWT）。
- Storages 内の **話者分離後の連結（transcript_speaker_separated_combined/*.txt）** を選択して処理します。
- 各つなぎ目で、後半セグメントの「最初の数行」をキーに前半末尾から一致位置を探し、
  **前半側にだけ** `-----ここから重複-----` を挿入します。
"""
)

user_dir = _sanitize_username_for_path(str(current_user))

# =========================
# Sidebar params
# =========================
with st.sidebar:
    st.header("検出パラメータ")

    min_match_size = st.slider(
        "一致とみなす最低文字数（MIN_MATCH_SIZE）",
        min_value=5,
        max_value=40,
        value=DEFAULT_MIN_MATCH_SIZE,
        step=1,
        help="小さいほど検出がゆるくなります。",
    )

    autojunk_option = st.radio(
        "autojunk（SequenceMatcher 自動ジャンク判定）",
        options=["ON（デフォルト）", "OFF（短い文でも精確に）"],
        index=0,
        help="OFF にすると短い一致を拾いやすくなります（少し遅くなることがあります）。",
    )
    use_autojunk = autojunk_option.startswith("ON")

    st.divider()
    st.caption("保存先は、選択ジョブの transcript/ です。")

# =========================
# combined txt selection
# =========================
items = list_combined_texts(user_dir)
if not items:
    st.info(
        "Storages に連結文字起こし（transcripts_combined_*.txt）が見つかりません。\n\n"
        "先に「文字起こし（storage対応）」で transcript を作成してください。"
    )
    st.stop()

labels = [it.label for it in items]
picked = st.radio("処理対象（combined txt）", options=labels, index=0)
it = items[labels.index(picked)]

st.caption(f"選択ファイル: {it.path}")

with st.expander("📌 選択ジョブ情報", expanded=False):
    st.write(
        {
            "job_dir": str(it.job_dir),
            "transcript_dir": str(it.transcript_dir),
            "job_id": it.job_id,
            "date": it.date,
            "created_at": it.created_at,
        }
    )

run = st.button("▶️ 重複箇所検出を実行する", type="primary")

if run:
    # 読む
    try:
        text = it.path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = it.path.read_text(encoding="cp932", errors="replace")

    # 実行
    with st.spinner("重複箇所を検出しています…"):
        merged, logs = build_merged_text(text, min_match_size, use_autojunk)

    # 保存（同じ transcript/）
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    # ★ 追加：保存先フォルダ（transcript_marked）
    marked_dir = it.job_dir / "transcript_marked"
    safe_mkdir(marked_dir)

    # ==============================
    # 出力ファイル名
    # ==============================
    out_txt = marked_dir / f"{it.path.stem}_marked.txt"
    out_log = marked_dir / f"{it.path.stem}_detect_log.json"

    write_text(out_txt, merged)
    write_json(
        out_log,
        {
            "input": str(it.path),
            "output_text": str(out_txt),
            "output_log": str(out_log),
            "params": {
                "OVERLAP_CHARS": OVERLAP_CHARS,
                "HEAD_CHARS": HEAD_CHARS,
                "HEAD_SENTENCES": HEAD_SENTENCES,
                "HEAD_SHIFT_TRIES": HEAD_SHIFT_TRIES,
                "min_match_size": int(min_match_size),
                "autojunk": bool(use_autojunk),
                "BEGIN_TAG": BEGIN_TAG,
            },
            "counts": {"markers": len(logs), "has_markers": bool(logs)},
            "logs": logs,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "user": str(current_user),
            "job_dir": str(it.job_dir),
        },
    )

    # jobの process.log にも軽く書く
    log_path = it.job_dir / "logs" / "process.log"
    append_log(log_path, "OVERLAP DETECT START")
    append_log(log_path, f"input={it.path.name}")
    append_log(log_path, f"output={out_txt.name}")
    append_log(log_path, f"log={out_log.name}")
    append_log(log_path, f"min_match_size={min_match_size} autojunk={use_autojunk}")
    append_log(log_path, "OVERLAP DETECT DONE")

    # 表示
    st.success("処理が完了しました（ストレージに保存しました）。")

    st.markdown("### 💾 保存先（ストレージ）")
    st.write({"merged_txt": str(out_txt), "log_json": str(out_log)})

    st.markdown("### 🔍 つなぎ目ごとの検出ログ（概要）")
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
        st.info("つなぎ目マーカーが見つかりませんでした（ログは空）。")

    with st.expander("🧩 head_phrase / matched_phrase 詳細", expanded=False):
        if not logs:
            st.info("ログがありません。")
        else:
            for item in logs:
                st.markdown(f"#### つなぎ目 {item['つなぎ目番号']} — {item['検出結果']}")
                st.markdown("**head_phrase**")
                st.code(item.get("head_phrase") or "（空）")
                st.markdown("**shifted_phrases**")
                shifted = item.get("shifted_phrases") or []
                if shifted:
                    for s in shifted:
                        st.code(s)
                else:
                    st.write("（なし）")
                st.markdown("**matched_phrase**")
                st.code(item.get("matched_phrase") or "（マッチなし）")
                st.markdown("---")

    with st.expander("📘 結果プレビュー（merged）", expanded=False):
        st.text(merged)

    st.download_button(
        "📥 結合テキストをダウンロード (.txt)",
        data=merged.encode("utf-8"),
        file_name=f"{it.path.stem}_重複箇所検出_{ts_tag}.txt",
        mime="text/plain",
    )
