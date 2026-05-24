# -*- coding: utf-8 -*-
# pages/30_重複箇所検出_storage対応.py
# ============================================================
# 🧩 重複箇所検出（ストレージ対応 / 非AI / セッション記録のみ）
#
# 目的：
# - Storages/<user>/minutes_app/ 配下の「連結テキスト（combined txt）」を列挙
# - radio で 1つ選択 → 重複検出（前半側にだけ BEGIN_TAG を挿入）
# - 結果（marked txt）＋検出ログ（json）を transcript_marked/ に保存
#
# 方針：
# - AI は一切使わない（busy / 実行時間測定は不要）
# - st.form は使わない
# - use_container_width は使わない
# - ログイン判定の正本は page_session_heartbeat
# ============================================================

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

import streamlit as st

# ============================================================
# sys.path（common_lib を import できるように）
# ============================================================
_THIS = Path(__file__).resolve()
APP_DIR = _THIS.parents[1]
PROJ_DIR = _THIS.parents[2]
MONO_ROOT = _THIS.parents[3]

for p in (MONO_ROOT, PROJ_DIR, APP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

PROJECTS_ROOT = MONO_ROOT
APP_NAME = _THIS.parents[1].name
PAGE_NAME = _THIS.stem

# ============================================================
# common_lib（正本）
# ============================================================
from common_lib.ui.page_header import render_standard_page_header

from lib.overlap_detect.explanation import (
    render_overlap_detect_page_intro,
    render_overlap_detect_help_expander,
)
from common_lib.storage.external_ssd_root import resolve_storage_subdir_root

# ============================================================
# session_state keys：ページ単位で名前空間化（他ページ汚染防止）
# ============================================================
PAGE_KEY_PREFIX = _THIS.stem  # e.g. "30_重複箇所検出_storage対応"


def k(name: str) -> str:
    return f"{PAGE_KEY_PREFIX}::{name}"


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

def get_base_from_original(job_dir: Path) -> tuple[str, str]:
    """
    job_dir/original/ の先頭ファイル名を表示用に拾う。
    - original_name: 元ファイル名（例: sample.wav）
    - base_stem: 拡張子を除いたベース名（例: sample）
    ない場合は ("none", "none") を返す。
    """
    original_dir = job_dir / "original"
    if not original_dir.exists():
        return "none", "none"

    files = [p for p in original_dir.iterdir() if p.is_file()]
    if not files:
        return "none", "none"

    # 安定化：名前順で先頭を採用
    files_sorted = sorted(files, key=lambda p: p.name.lower())
    p0 = files_sorted[0]
    return p0.stem, p0.name



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

            combined_dir = job_dir / "transcript_speaker_separated_combined"
            if not combined_dir.exists():
                continue

            combined_files = sorted(
                combined_dir.glob("*.txt"),
                key=lambda p: p.name.lower(),
                reverse=True,
            )
            for p in combined_files:
                #label = f"{date} / {job_id} / {p.name} / created={_human_dt(created_at)}"

                base_stem, original_name = get_base_from_original(job_dir)
                label = (
                    f"{date} / {job_id} / {p.name}\n"
                    f"  └ original: {original_name}\n"
                    f"  └ base: {base_stem} / created={_human_dt(created_at)}"
                )

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
SPEAKER_PREFIX_PATTERN = re.compile(
    r"""^(
        \s*
        (?:司会|ＭＣ|MC|進行)
        \s*[:：]\s*
      |
        \s*\[?\s*[sS]\s*\d+\s*\]?\s*[:：]\s*
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
    for _ in range(5):
        m = SPEAKER_PREFIX_PATTERN.match(s)
        if not m:
            break
        s = s[m.end() :]
    return s


def extract_head_phrase(next_seg: str) -> str:
    if not next_seg:
        return ""

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
# ページ設定（必須・統一）
# ============================================================
st.set_page_config(
    page_title="Minutes Maker",
    page_icon="📝",
    layout="wide",
)


# ============================================================
# 共通ヘッダー
# - settings.toml から BANNER_KEY を取得
# - banner / theme / intro CSS を描画
# - page_session_heartbeat を実行
# - title / subtitle / ログイン状態を描画
# ============================================================
sub, theme, BANNER_KEY, settings = render_standard_page_header(
    st_module=st,
    projects_root=PROJECTS_ROOT,
    app_dir=APP_DIR,
    app_name=APP_NAME,
    page_name=PAGE_NAME,
    title="📝 重複箇所検出",
    subtitle_text="連結テキストのつなぎ目重複を検出し，重複部分に目印を挿入します",
    default_banner_key="light_green",
)

# ============================================================
# ページ説明
# ============================================================
render_overlap_detect_page_intro()

# ============================================================
# 詳細説明
# ============================================================
render_overlap_detect_help_expander(
    theme=theme,
    banner_key=BANNER_KEY,
)


current_user = sub
user_dir = _sanitize_username_for_path(str(current_user))

# ============================================================
# セッションキー（結果保持）
# ============================================================
K_LAST_RESULT = f"{PAGE_NAME}__last_result"
st.session_state.setdefault(K_LAST_RESULT, None)

# ============================================================
# 入力
# ============================================================
st.subheader("① テキストファイルの設定")
st.caption("話者分離して結合したテキストファイルを読み込む先をここで指定します．"
    "「話者分離」を行った時にサーバー内部に自動保存されたテキストファイルを"
    "使用するときは「既存ジョブ」を選択してください．")

input_mode = st.radio(
    "どこから combined txt を読み込むか",
    options=[
        "既存ジョブ（storage から選択）",
        "新規アップロード",
    ],
    index=0,
    label_visibility="collapsed",
    key=k("input_mode"),
)

# drop用 state
K_UP_LAST_SIG = k("upload_last_sig")
K_UP_JOB_ID = k("upload_job_id")
K_UP_JOB_ROOT = k("upload_job_root")
K_UP_LOCKED = k("upload_job_locked")

st.session_state.setdefault(K_UP_LAST_SIG, None)
st.session_state.setdefault(K_UP_JOB_ID, None)
st.session_state.setdefault(K_UP_JOB_ROOT, None)
st.session_state.setdefault(K_UP_LOCKED, False)


def now_job_id() -> str:
    return "job_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_filename(name: str) -> str:
    s = (name or "text").strip()
    s = s.replace("\\", "_").replace("/", "_").replace(":", "_")
    s = re.sub(r"\s+", " ", s)
    return s or "text"


# ============================================================
# Sidebar params
# ============================================================
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
    st.caption("保存先は、選択ジョブの transcript_marked/ です。")


# ============================================================
# combined txt selection（既存 / drop 両対応）
# ============================================================
items: list[CombinedItem] = []

if input_mode.startswith("既存ジョブ"):
    items = list_combined_texts(user_dir)
    st.markdown("##### ジョブ選択（サーバー内部の保存ファイル）")
    st.write("直近の5つのジョブより古いものは自動的に消去されます")
    if not items:
        st.info(
            "Storages に連結テキスト（combined txt）が見つかりません。\n\n"
            "先に「話者分離（storage対応）」で combined を作成してください。"
        )
else:
    st.markdown("### 新規アップロード（drop）")
    uploaded = st.file_uploader(
        "連結テキスト（.txt）をドロップ/選択（1つ以上）",
        type=["txt"],
        accept_multiple_files=True,
        key=k("uploader_combined_txt"),
    )

    if uploaded:
        sig_parts = [f"{getattr(f, 'name', 'file')}:{getattr(f, 'size', 0)}" for f in uploaded]
        upload_sig = "|".join(sig_parts)

        # ファイルセットが変わったら「未確定」に戻す
        if st.session_state.get(K_UP_LAST_SIG) != upload_sig:
            st.session_state[K_UP_LAST_SIG] = upload_sig
            st.session_state[K_UP_JOB_ID] = None
            st.session_state[K_UP_JOB_ROOT] = None

        st.caption("※ まずは『保存して新規ジョブ作成』で確定してください。")

        existing_job_root = st.session_state.get(K_UP_JOB_ROOT)
        existing_job_id = st.session_state.get(K_UP_JOB_ID)

        if existing_job_root and existing_job_id and st.session_state.get(K_UP_LAST_SIG) == upload_sig:
            job_root = Path(existing_job_root)
            st.success(f"✅ 作成済みジョブを復元しました: {job_root}")

        if not (existing_job_root and existing_job_id and st.session_state.get(K_UP_LAST_SIG) == upload_sig):
            create_job_clicked = st.button(
                "📦 アップロードを combined として保存して新規ジョブを作成",
                type="primary",
                key=k("create_job_btn"),
                disabled=bool(st.session_state.get(K_UP_LOCKED, False)),
                help="押した時点で Storages に job_YYYYMMDD_HHMMSS を作り、transcript_speaker_separated_combined/ に保存します。",
            )

            if create_job_clicked:
                st.session_state[K_UP_LOCKED] = True
                try:
                    with st.spinner("新規ジョブを作成して combined を保存しています…"):
                        today_dir = datetime.now().strftime("%Y-%m-%d")
                        job_id = now_job_id()
                        job_root = STORAGE_ROOT / user_dir / "minutes_app" / today_dir / job_id

                        combined_dir = job_root / "transcript_speaker_separated_combined"
                        logs_dir = job_root / "logs"
                        safe_mkdir(combined_dir)
                        safe_mkdir(logs_dir)

                        log_path = logs_dir / "process.log"
                        append_log(log_path, "UPLOAD(COMBINED)->JOB START")
                        append_log(log_path, f"job_dir={job_root}")

                        saved: list[dict[str, Any]] = []
                        for i, uf in enumerate(uploaded, start=1):
                            name = safe_filename(getattr(uf, "name", f"combined_{i}.txt"))
                            out_name = f"{i:03d}_{name}"
                            out_path = combined_dir / out_name
                            b = uf.getvalue()
                            try:
                                text = b.decode("utf-8")
                            except UnicodeDecodeError:
                                text = b.decode("cp932", errors="ignore")
                            write_text(out_path, text)
                            saved.append({"order": i, "original_name": name, "saved_name": out_name})
                            append_log(log_path, f"saved combined -> {out_name}")

                        job_json = {
                            "job_id": job_id,
                            "user": str(current_user),
                            "user_dir": user_dir,
                            "date": today_dir,
                            "created_at": datetime.now().isoformat(timespec="seconds"),
                            "source": "upload_drop_combined_txt",
                            "paths": {
                                "job_root": str(job_root),
                                "combined_dir": str(combined_dir),
                                "logs_dir": str(logs_dir),
                            },
                            "saved_files": saved,
                        }
                        write_text(job_root / "job.json", json.dumps(job_json, ensure_ascii=False, indent=2))
                        append_log(log_path, "UPLOAD(COMBINED)->JOB DONE")

                    st.session_state[K_UP_JOB_ID] = job_id
                    st.session_state[K_UP_JOB_ROOT] = str(job_root)
                    st.success(f"✅ 新規ジョブを作成しました: {job_root}")

                finally:
                    st.session_state[K_UP_LOCKED] = False

        # 作成済み（または復元済み）なら、その job の combined を items に入れる
        if st.session_state.get(K_UP_JOB_ROOT):
            job_root = Path(st.session_state[K_UP_JOB_ROOT])
            meta = _read_job_json(job_root)
            created_at = meta.get("created_at")

            combined_dir = job_root / "transcript_speaker_separated_combined"
            combined_files = sorted(combined_dir.glob("*.txt"), key=lambda p: p.name.lower(), reverse=True)

            for p in combined_files:
                label = f"{job_root.parent.name} / {job_root.name} / {p.name} / created={_human_dt(created_at)}"
                items.append(
                    CombinedItem(
                        label=label,
                        path=p,
                        job_dir=job_root,
                        transcript_dir=combined_dir,
                        job_id=job_root.name,
                        date=job_root.parent.name,
                        created_at=created_at,
                    )
                )
    else:
        st.info("連結テキスト（.txt）をアップロードしてください。")

# ---- 共通：選択UI ----
selected: Optional[CombinedItem] = None
if items:
    labels = [it.label for it in items]
    picked = st.radio("処理対象（combined txt）",
        options=labels, index=0,
        label_visibility="collapsed",
        key=k("picked_combined"))
    selected = items[labels.index(picked)]
    st.caption(f"選択ファイル: {selected.path}")

    with st.expander("📌 選択ジョブ情報", expanded=False):
        base_stem, original_name = get_base_from_original(selected.job_dir)

        st.write(
            {
                "job_dir": str(selected.job_dir),
                "transcript_dir": str(selected.transcript_dir),
                "job_id": selected.job_id,
                "date": selected.date,
                "created_at": selected.created_at,
                "original": original_name,
                "base": base_stem,
            }
        )


# ============================================================
# 実行（非AI）
# ============================================================
st.subheader("② 重複箇所検出の実行")

run = st.button(
    "▶️ 重複箇所検出を実行する",
    type="primary",
    key=k("run_btn"),
    disabled=(selected is None),
)

if run:
    if selected is None:
        st.warning("処理対象（combined txt）を選択してください。")
        st.stop()

    it = selected  # NameError 防止

    # 読み込み
    try:
        text = it.path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = it.path.read_text(encoding="cp932", errors="replace")

    # 実行
    try:
        with st.spinner("重複箇所を検出しています…"):
            merged, logs = build_merged_text(text, int(min_match_size), bool(use_autojunk))
    except Exception as e:
        st.error(f"検出に失敗しました: {e}")
        st.stop()

    # 保存
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    marked_dir = it.job_dir / "transcript_marked"
    safe_mkdir(marked_dir)

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

    log_path = it.job_dir / "logs" / "process.log"
    append_log(log_path, "OVERLAP DETECT START")
    append_log(log_path, f"input={it.path.name}")
    append_log(log_path, f"output={out_txt.name}")
    append_log(log_path, f"log={out_log.name}")
    append_log(log_path, f"min_match_size={min_match_size} autojunk={use_autojunk}")
    append_log(log_path, "OVERLAP DETECT DONE")

    # 結果を session_state に保持（テンプレ準拠）
    st.session_state[K_LAST_RESULT] = {
        "selected": it.label,
        "saved_marked_txt": str(out_txt),
        "saved_log_json": str(out_log),
        "ts_tag": ts_tag,
        "log_count": len(logs),
    }

    #st.success("重複検出が完了しました（ストレージに保存しました）。")
    st.success("重複検出が完了しました（ファイルをパソコンにダウンロードして確認してください）。")

# ============================================================
# 結果表示（テンプレ準拠）
# ============================================================
st.divider()
st.subheader("③ 結果")

if st.session_state.get(K_LAST_RESULT) is None:
    st.info("まだ結果がありません。")
    st.stop()

last = st.session_state[K_LAST_RESULT]
st.caption(f"保存先: {last.get('saved_marked_txt')}")

# 直近実行で選んだものを復元表示したい場合（selectedが変わると中身が変わるので、ここは“直近の結果”として表示）
# ただし、詳細ログやプレビューは merged/logs が必要なので、直近実行時の結果を再生成せず、保存物を読む。

# 保存した marked テキストを読む
marked_path = Path(str(last.get("saved_marked_txt") or "")).expanduser()
log_path_json = Path(str(last.get("saved_log_json") or "")).expanduser()

merged_text = ""
logs = []
try:
    if marked_path.exists():
        try:
            merged_text = marked_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            merged_text = marked_path.read_text(encoding="cp932", errors="replace")
except Exception:
    merged_text = ""

try:
    if log_path_json.exists():
        logs_obj = json.loads(log_path_json.read_text(encoding="utf-8"))
        logs = logs_obj.get("logs", []) if isinstance(logs_obj, dict) else []
except Exception:
    logs = []

st.markdown("### 🔍 つなぎ目ごとの検出ログ（概要）")
if logs:
    st.table(
        [
            {
                "つなぎ目番号": item.get("つなぎ目番号"),
                "ファイル名": item.get("ファイル名"),
                "検出結果": item.get("検出結果"),
                "開始位置": item.get("開始位置"),
                "一致文字数": item.get("一致文字数"),
            }
            for item in logs
        ]
    )
else:
    st.info("ログがありません（つなぎ目マーカーが無い／未検出など）。")

with st.expander("🧩 head_phrase / matched_phrase 詳細", expanded=False):
    if not logs:
        st.info("ログがありません。")
    else:
        for item in logs:
            st.markdown(f"#### つなぎ目 {item.get('つなぎ目番号')} — {item.get('検出結果')}")
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

with st.expander("📘 結果プレビュー（marked）", expanded=False):
    st.text(merged_text)

# --- download file_name に original 名を入れる ---
job_dir_for_name = marked_path.parent.parent if marked_path else None  # .../job_xxx/transcript_marked/xxx.txt の想定
base_stem, original_name = get_base_from_original(job_dir_for_name) if job_dir_for_name else ("none", "none")

# ファイル名に使えるよう最低限安全化（スペース・スラッシュ等を潰す）
orig_safe = (original_name or "none").strip()
orig_safe = orig_safe.replace("\\", "_").replace("/", "_").replace(":", "_")
orig_safe = re.sub(r"\s+", "_", orig_safe)

ts_tag = str(last.get("ts_tag") or "")
download_name = f"{orig_safe}__{marked_path.stem}__{ts_tag}.txt" if marked_path else "merged_marked.txt"

st.markdown(
    """
<style>
div.stDownloadButton > button {
    background-color: #ff1744;
    color: white;
    border: 1px solid #d50000;
    border-radius: 8px;
    font-weight: 600;
}

div.stDownloadButton > button:hover {
    background-color: #d50000;
    color: white;
}
</style>
""",
    unsafe_allow_html=True,
)

st.download_button(
    "📥 検出済みのテキストをダウンロード",
    data=(merged_text or "").encode("utf-8"),
    file_name=download_name,
    mime="text/plain",
)

