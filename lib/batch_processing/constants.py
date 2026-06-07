# -*- coding: utf-8 -*-
# minutes_app/lib/batch_processing/constants.py
# ============================================================
# 一括処理の定数
# ============================================================

from __future__ import annotations

import re

# ============================================================
# 音声分割設定
# ============================================================
DEFAULT_CHUNK_MIN = 20
DEFAULT_OVERLAP_MIN = 1.0
DEFAULT_EXPORT_FMT = "mp3"
DEFAULT_TARGET_BITRATE = "160k"
DEFAULT_FADE_MS = 0
DEFAULT_ABSORB_TINY_TAIL = True

# ============================================================
# 文字起こし設定
# ============================================================
DEFAULT_TRANSCRIBE_MODEL = "whisper-1"
DEFAULT_TRANSCRIBE_RESPONSE_FORMAT = "json"
DEFAULT_LANGUAGE = "ja"
DEFAULT_TRANSCRIBE_PROMPT = ""
DEFAULT_STRIP_BRACKET_TAGS = True

# ============================================================
# 話者分離設定
# ============================================================
DEFAULT_SPEAKER_PROMPT_LEVEL = "標準（精度優先）"

# ============================================================
# 重複検出設定
# ============================================================
OVERLAP_CHARS = 700
HEAD_CHARS = 400
HEAD_SENTENCES = 3
HEAD_SHIFT_TRIES = 3
DEFAULT_MIN_MATCH_SIZE = 20
DEFAULT_USE_AUTOJUNK = True
BEGIN_TAG = "-----ここから重複-----"

# ============================================================
# 正規表現
# ============================================================
MARKER_PATTERN = re.compile(
    r"^-{3,}\s*ここがつなぎ目です（(.*?)）.*$",
    re.MULTILINE,
)

BRACKET_TAG_PATTERN = re.compile(r"【[^】]*】")

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

AUDIO_EXTS = {".mp3", ".wav", ".mp4", ".m4a", ".webm", ".ogg"}