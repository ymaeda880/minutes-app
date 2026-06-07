# -*- coding: utf-8 -*-
# minutes_app/lib/batch_processing/overlap_utils.py
# ============================================================
# 一括処理：重複検出ロジック
# ============================================================

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from .constants import (
    BEGIN_TAG,
    DEFAULT_USE_AUTOJUNK,
    HEAD_CHARS,
    HEAD_SENTENCES,
    HEAD_SHIFT_TRIES,
    MARKER_PATTERN,
    OVERLAP_CHARS,
    SPEAKER_PREFIX_PATTERN,
)


# ============================================================
# marker split
# ============================================================
def split_by_markers(text: str) -> tuple[list[str], list[dict[str, str]]]:
    segments: list[str] = []
    markers: list[dict[str, str]] = []
    prev_end = 0

    for m in MARKER_PATTERN.finditer(text):
        seg = text[prev_end : m.start()]
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
# speaker label cleanup
# ============================================================
def strip_leading_speaker_labels(text: str) -> str:
    if not text:
        return ""

    s = text.lstrip("\ufeff")

    for _ in range(5):
        m = SPEAKER_PREFIX_PATTERN.match(s)
        if not m:
            break
        s = s[m.end() :]

    return s


# ============================================================
# head phrase
# ============================================================
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
# normalize
# ============================================================
def normalize_text(s: str) -> str:
    s = s.replace("　", "")
    s = s.replace("\n", "")
    s = s.replace(" ", "")

    for ch in "、。！？,.!?":
        s = s.replace(ch, "")

    return s


# ============================================================
# raw / normalized index map
# ============================================================
def _build_index_map(raw: str, norm: str) -> list[tuple[int, int]]:
    mapping: list[tuple[int, int]] = []
    j = 0

    for i, ch in enumerate(raw):
        ch_norm = normalize_text(ch)
        if ch_norm == "":
            continue

        if j < len(norm):
            mapping.append((j, i))
            j += 1

    return mapping


def _mapped_index(mapping: list[tuple[int, int]], idx_norm: int) -> int | None:
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


# ============================================================
# phrase match
# ============================================================
def _match_with_phrase(
    prev_seg: str,
    phrase: str,
    min_match_size: int,
    use_autojunk: bool,
) -> tuple[int, int]:
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

    head_map = _build_index_map(phrase, norm_head)
    prev_map = _build_index_map(prev_tail, norm_prev)

    head_raw_start = _mapped_index(head_map, a)
    prev_raw_start = _mapped_index(prev_map, b)

    if head_raw_start is None or prev_raw_start is None:
        return -1, 0

    start_in_tail = max(0, int(prev_raw_start) - int(head_raw_start))
    global_prev_start = len(prev_seg) - len(prev_tail) + start_in_tail

    return global_prev_start, int(size)


# ============================================================
# overlap start
# ============================================================
def find_overlap_start(
    prev_seg: str,
    next_seg: str,
    min_match_size: int,
    use_autojunk: bool,
) -> tuple[int, int, str, list[str], str]:
    if not prev_seg or not next_seg:
        return -1, 0, "", [], ""

    raw_head = extract_head_phrase(next_seg)

    if not raw_head:
        return -1, 0, "", [], ""

    base_head = raw_head.strip().rstrip("。？！!？，、,.")

    if not base_head:
        return -1, 0, "", [], ""

    candidates: list[str] = []
    seen: set[str] = set()
    shifted_heads: list[str] = []

    def add_candidate(s: str, is_shifted: bool = False) -> None:
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
        start_pos, size = _match_with_phrase(
            prev_seg,
            phrase,
            min_match_size,
            use_autojunk,
        )

        if start_pos >= 0 and size > 0:
            matched_phrase = phrase
            return start_pos, size, base_head, shifted_heads, matched_phrase

    return -1, 0, base_head, shifted_heads, ""


# ============================================================
# merge with markers
# ============================================================
def build_merged_text(
    text: str,
    min_match_size: int,
    use_autojunk: bool = DEFAULT_USE_AUTOJUNK,
) -> tuple[str, list[dict[str, Any]]]:
    segments, markers = split_by_markers(text)

    if len(segments) <= 1 or not markers:
        return text, []

    merged: list[str] = []
    logs: list[dict[str, Any]] = []

    merged.append(segments[0])

    for idx, marker in enumerate(markers):
        prev_seg = segments[idx]
        next_seg = segments[idx + 1]

        start_pos, size, base_head, shifted_heads, matched_phrase = find_overlap_start(
            prev_seg,
            next_seg,
            min_match_size,
            use_autojunk,
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