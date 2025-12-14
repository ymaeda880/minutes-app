# config/config.py
import streamlit as st

# ============================================================
# OpenAI API
# ============================================================
OPENAI_TRANSCRIBE_URL = "https://api.openai.com/v1/audio/transcriptions"

def get_openai_api_key() -> str:
    return st.secrets.get("OPENAI_API_KEY", "")

# ============================================================
# Gemini API
# ============================================================
def get_gemini_api_key() -> str:
    return st.secrets.get("GEMINI_API_KEY", "")

def has_gemini_api_key() -> bool:
    return bool(get_gemini_api_key())

# ============================================================
# 価格（USD / 100万トークン）※テキスト生成・理解系
# ============================================================
MODEL_PRICES_USD = {
    # --- OpenAI ---
    "gpt-5":         {"in": 1.25,  "out": 10.00},
    "gpt-5-mini":    {"in": 0.25,  "out": 2.00},
    "gpt-5-nano":    {"in": 0.05,  "out": 0.40},

    "gpt-4.1":       {"in": 2.00,  "out": 8.00},
    "gpt-4.1-mini":  {"in": 0.40,  "out": 1.60},

    # ★ 追加（GPT-4o 系）
    "gpt-4o":        {"in": 1.00,  "out": 4.00},
    "gpt-4o-mini":   {"in": 0.15,  "out": 0.60},

    # --- Gemini ---
    # ※ 公式価格改定があり得るので「概算用」
    "gemini-2.0-flash": {"in": 0.30, "out": 2.50},
    "gemini-2.0-pro":   {"in": 1.25, "out": 10.00},
}

# ============================================================
# Whisper / Transcribe（USD / 分）
# ============================================================
WHISPER_PRICE_PER_MIN = 0.006  # Whisper

TRANSCRIBE_PRICES_USD_PER_MIN = {
    "gpt-4o-mini-transcribe": 0.0125,
    "gpt-4o-transcribe":      0.025,
    "whisper-1":              WHISPER_PRICE_PER_MIN,
    # Gemini は「分単価」ではないためここには入れない
}

# ============================================================
# 為替（USDJPY）
# ============================================================
DEFAULT_USDJPY = float(st.secrets.get("USDJPY", 150.0))

# ============================================================
# Gemini 用：トークン概算ユーティリティ
# ============================================================
def estimate_tokens_from_text(text: str) -> int:
    """
    Gemini / OpenAI 共通の簡易トークン推定。
    厳密ではないが費用目安には十分。
    目安：1 token ≒ 4 characters（日本語含む概算）
    """
    if not text:
        return 0
    return max(len(text) // 4, 1)

def estimate_gemini_cost_usd(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    """
    Gemini の概算費用（USD）を返す。
    未定義モデルの場合は None。
    """
    price = MODEL_PRICES_USD.get(model)
    if not price:
        return None

    usd = (
        (input_tokens * price["in"]) +
        (output_tokens * price["out"])
    ) / 1_000_000
    return usd
