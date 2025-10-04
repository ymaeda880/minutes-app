# lib/costs.py
from typing import Optional
from config.config import MODEL_PRICES_USD

def estimate_chat_cost_usd(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """Chat料金（USD）を概算。価格未設定モデルは None。modern（input/output）で統一。"""
    p = MODEL_PRICES_USD.get(model)
    if not p or p.get("in") is None or p.get("out") is None:
        return None
    # 単価は USD / 1,000,000 tokens を想定
    return round((input_tokens * p["in"] + output_tokens * p["out"]) / 1_000_000, 6)
