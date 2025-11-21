#lib/tokens.py
"""
tokens.py — modern API 専用のトークン抽出ヘルパー（シンプル版）

前提:
  - OpenAI の modern 系 usage スキーマのみを扱う。
  - usage = { input_tokens, output_tokens, total_tokens } を想定。
  - legacy 系（prompt_tokens / completion_tokens）は未対応（=0扱い）。

方針:
  - usage が無い／欠損している場合でも安全に 0 を返す。
  - total_tokens が無ければ input_tokens + output_tokens で補完する。
"""

from __future__ import annotations
from typing import Any, NamedTuple, Dict


class Tokens(NamedTuple):
    input: int
    output: int
    total: int


def _as_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _read_usage_modern(usage_obj: Any) -> Dict[str, int]:
    """
    modern スキーマの usage から input/output/total を読み取って int に積み替える。
    usage_obj は dict でも属性オブジェクトでもよい。
    欠損は 0 として扱う。
    """
    if usage_obj is None:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # 属性アクセス優先
    def _get(name: str):
        try:
            return getattr(usage_obj, name)
        except Exception:
            return None

    input_tok  = _get("input_tokens")
    output_tok = _get("output_tokens")
    total_tok  = _get("total_tokens")

    # dict の場合に補完
    if isinstance(usage_obj, dict):
        if input_tok  is None: input_tok  = usage_obj.get("input_tokens")
        if output_tok is None: output_tok = usage_obj.get("output_tokens")
        if total_tok  is None: total_tok  = usage_obj.get("total_tokens")

    return {
        "input_tokens":  _as_int(input_tok),
        "output_tokens": _as_int(output_tok),
        "total_tokens":  _as_int(total_tok),
    }


def extract_tokens_from_usage(usage_obj: Any) -> Tokens:
    f = _read_usage_modern(usage_obj)
    input_i  = f["input_tokens"]
    output_i = f["output_tokens"]
    total_i  = f["total_tokens"] or (input_i + output_i)

    # ---- modern usage フォールバック ----
    # modern API は input/output を 0 のまま返すケースがある（total_tokens のみ有効）。
    if total_i > 0 and input_i == 0 and output_i == 0:
        # もっとも単純で安全な下限見積り → 全部 input とみなす
        input_i = total_i
        output_i = 0

        # ※もし入力/出力を「半々」で扱いたければ以下を使う：
        # input_i = total_i // 2
        # output_i = total_i - input_i

    return Tokens(input_i, output_i, total_i)



def extract_tokens_from_response(resp: Any) -> Tokens:
    """
    レスポンス resp から resp.usage を取り出して extract_tokens_from_usage を適用。
    usage が無い場合は (0,0,0)。
    """
    if resp is None:
        return Tokens(0, 0, 0)
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    return extract_tokens_from_usage(usage)


def debug_usage_snapshot(usage_obj: Any) -> Dict[str, int]:
    """
    modern フィールドだけを整数化して返すスナップショット。
    （legacy フィールドは無視）
    """
    f = _read_usage_modern(usage_obj)
    return {
        "input_tokens":  f["input_tokens"],
        "output_tokens": f["output_tokens"],
        "total_tokens":  f["total_tokens"] or (f["input_tokens"] + f["output_tokens"]),
    }
