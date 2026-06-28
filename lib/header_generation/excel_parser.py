# -*- coding: utf-8 -*-
# minutes_app/lib/header_generation/excel_parser.py
# ============================================================
# 議事録ヘッダー用 Excel / CSV 解析
# ============================================================

from __future__ import annotations

import io
from typing import Any

import pandas as pd


def cell_to_text(value: Any) -> str:
    """Excel / CSV のセル値を Word 出力用の文字列に変換する。"""

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()

    # Excel由来の 1.0, 2.0 のような整数表記を 1, 2 に戻す
    if text.endswith(".0"):
        left = text[:-2]
        if left.isdigit():
            return left

    return text


def is_empty_row(values: list[str]) -> bool:
    """行全体が空かどうかを判定する。"""

    return all(not str(v).strip() for v in values)


def is_heading_row(values: list[str]) -> bool:
    """1セルだけ文字が入っている行を表ブロックの見出し行として判定する。"""

    non_empty = [
        str(v).strip()
        for v in values
        if str(v).strip()
    ]

    return len(non_empty) == 1


def parse_dataframe_blocks(df: pd.DataFrame) -> list[dict[str, Any]]:
    """DataFrame から縦並びの表ブロックを抽出する。"""

    rows: list[list[str]] = []

    for _, row in df.iterrows():
        rows.append([cell_to_text(v) for v in row.tolist()])

    blocks: list[dict[str, Any]] = []
    i = 0

    while i < len(rows):
        row = rows[i]

        if not is_heading_row(row):
            i += 1
            continue

        title = [v for v in row if v.strip()][0].strip()

        table_rows: list[list[str]] = []
        i += 1

        while i < len(rows):
            current = rows[i]

            if is_empty_row(current):
                i += 1
                break

            if is_heading_row(current):
                break

            table_rows.append(current)
            i += 1

        if not table_rows:
            continue

        max_cols = max(len(r) for r in table_rows)
        normalized_rows = [
            r + [""] * (max_cols - len(r))
            for r in table_rows
        ]

        while normalized_rows and is_empty_row(normalized_rows[-1]):
            normalized_rows.pop()

        if not normalized_rows:
            continue

        header = normalized_rows[0]
        data = normalized_rows[1:]

        blocks.append(
            {
                "title": title,
                "header": header,
                "data": data,
            }
        )

    return blocks


def parse_excel_blocks(file_bytes: bytes) -> list[dict[str, Any]]:
    """xlsx の先頭シートから表ブロックを抽出する。"""

    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    sheet_name = xls.sheet_names[0]

    df = pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=sheet_name,
        header=None,
        dtype=object,
    )

    return parse_dataframe_blocks(df)


def parse_csv_blocks(file_bytes: bytes) -> list[dict[str, Any]]:
    """CSV から表ブロックを抽出する。"""

    try:
        text = file_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = file_bytes.decode("cp932", errors="replace")

    df = pd.read_csv(
        io.StringIO(text),
        header=None,
        dtype=object,
    )

    return parse_dataframe_blocks(df)