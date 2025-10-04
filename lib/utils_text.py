# lib/utils_text.py
import re
from difflib import SequenceMatcher

# ========== 基本ユーティリティ ==========
def add_line_numbers(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(f"{i+1:>4}: {line}" for i, line in enumerate(lines))

def post_process(text: str, trim_spaces=True, remove_empty_lines=True) -> str:
    t = text
    if trim_spaces:
        t = "\n".join(line.strip() for line in t.splitlines())
        t = re.sub(r"[ \t]{2,}", " ", t)
        t = re.sub(r"([^\s])\s+([、。！？])", r"\1\2", t)
    if remove_empty_lines:
        t = re.sub(r"\n{3,}", "\n\n", t)
    return t

def escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# ========== 文分割 ==========
def sentence_split_by_period(text: str) -> str:
    """
    既存の句点・疑問・感嘆記号 + 閉じ括弧の直後で必ず改行。
    文単位の比較や後段処理用の素直な分割。
    """
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r'([。．\.？\?！!])([」』）】〉》]*)', r'\1\2\n', t)
    t = re.sub(r'\n{2,}', '\n', t).strip()
    return t

def sentence_split_with_inferred_periods(text: str) -> str:
    """
    句点が無い箇所を推測して補完しつつ1文1行化。
    談話マーカー（話題転換表現）の前でも積極的に改行。
    """
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t\u3000]+", " ", t)

    # 0) 既存の文末記号で改行
    t = re.sub(r'([。．\.？\?！!])([」』）】〉》]*)', r'\1\2\n', t)

    # 談話マーカー（ページ移動/話題転換）
    nav_cues = (
        r"(?:ちょっと飛んで(?:\d+)?ページ(?:ですかね|ですね)?|"
        r"では|それでは|じゃあ|次に|次|さて|まず|ここで|この辺|今この|続いて|以上です)"
    )
    # A) 直前に句点が無ければ句点＋改行を補う
    t = re.sub(rf'(?<![。．\.？\?！!])\s+(?={nav_cues})', '。\n', t)
    # B) 句点があっても談話マーカーの前は確実に改行
    t = re.sub(rf'\s+(?={nav_cues})', '\n', t)

    # 文頭キュー語の直前に句点補完
    cue = (
        r"(?:はい[、。]?|うん[、。]?|では|それでは|じゃあ|次に|次|さて|まず|あと|続いて|"
        r"以上です|お願いします|お願いいたします)"
    )
    t = re.sub(r'(?<![。．\.？\?！!])\s+(?=' + cue + r')', '。', t)

    # 行ごとに丁寧体などの終止で句点補完
    lines = [ln.strip() for ln in t.split("\n")]
    polite_end = r'(です|ます|でした|ですね|でしょう|であります|である|だ|と思います|お願いいたします|ください|下さい)$'

    fixed = []
    for ln in lines:
        if not ln:
            continue
        ln = re.sub(r'([。．\.？\?！!])([」』）】〉》]*)\s+', r'\1\2\n', ln).strip()
        parts = [p for p in ln.split("\n") if p.strip()]
        for p in parts:
            if not re.search(r'[。．\.？\?！!]$', p):
                if re.search(polite_end, p):
                    p = p + "。"
                elif len(p) >= 40 and "、" in p:
                    pos = p.rfind("、")
                    if pos >= 15:
                        p = p[:pos] + "。" + "\n" + p[pos+1:]
            # 「…と思います + [空白] + 談話マーカー」で強制改行
            p = re.sub(rf'(と思います)(\s+)(?={nav_cues})', r'\1。\n', p)
            for q in str(p).split("\n"):
                q = q.strip()
                if q:
                    fixed.append(q)

    # 仕上げ
    fixed2 = []
    for ln in fixed:
        if not re.search(r'[。．\.？\?！!]$', ln):
            if re.search(polite_end, ln):
                ln = ln + "。"
        fixed2.append(ln)

    out = "\n".join(fixed2)
    out = re.sub(r'\n{2,}', '\n', out).strip()
    return out

# ========== 【…】削除 ==========
def make_pattern(left, right, non_greedy, multiline_spanning):
    middle = r"[\s\S]*?" if multiline_spanning else r".*?"
    if not non_greedy:
        middle = r"[\s\S]*" if multiline_spanning else r".*"
    return re.compile(re.escape(left) + middle + re.escape(right))

def strip_bracketed(text, pairs, non_greedy=True, multiline_spanning=True, max_passes=3):
    counts = {f"{l}{r}": 0 for (l, r) in pairs}
    current = text
    for _ in range(max_passes):
        changed = False
        for (l, r) in pairs:
            pattern = make_pattern(l, r, non_greedy, multiline_spanning)
            current, n = pattern.subn("", current)
            if n > 0:
                counts[f"{l}{r}"] += n
                changed = True
        if not changed:
            break
    return current, counts

# ========== 差分 ==========
def highlight_inline(before, after):
    sm = SequenceMatcher(None, before, after)
    b_parts, a_parts = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        b_seg, a_seg = before[i1:i2], after[j1:j2]
        if op == "equal":
            b_parts.append(escape_html(b_seg))
            a_parts.append(escape_html(a_seg))
        elif op == "delete":
            b_parts.append(f"<span class='del'>{escape_html(b_seg)}</span>")
        elif op == "insert":
            a_parts.append(f"<span class='ins'>{escape_html(a_seg)}</span>")
        elif op == "replace":
            b_parts.append(f"<span class='del'>{escape_html(b_seg)}</span>")
            a_parts.append(f"<span class='ins'>{escape_html(a_seg)}</span>")
    return "".join(b_parts), "".join(a_parts)

def build_line_diff(before_text, after_text):
    b_lines, a_lines = before_text.splitlines(), after_text.splitlines()
    sm = SequenceMatcher(None, b_lines, a_lines)
    rows = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                rows.append(("=", i1+k+1, j1+k+1,
                             escape_html(b_lines[i1+k]),
                             escape_html(a_lines[j1+k])))
        elif tag == "replace":
            length = max(i2 - i1, j2 - j1)
            for k in range(length):
                b_line = b_lines[i1+k] if i1+k < i2 else ""
                a_line = a_lines[j1+k] if j1+k < j2 else ""
                b_html, a_html = highlight_inline(b_line, a_line)
                rows.append(("~", i1+k+1 if b_line else None,
                             j1+k+1 if a_line else None, b_html, a_html))
        elif tag == "delete":
            for k in range(i2 - i1):
                rows.append(("-", i1+k+1, None,
                             f"<span class='del'>{escape_html(b_lines[i1+k])}</span>", ""))
        elif tag == "insert":
            for k in range(j2 - j1):
                rows.append(("+", None, j1+k+1, "",
                             f"<span class='ins'>{escape_html(a_lines[j1+k])}</span>"))
    return rows

def build_sentence_diff(before_text, after_text):
    b_lines = [ln for ln in sentence_split_by_period(before_text).splitlines() if ln.strip()]
    a_lines = [ln for ln in sentence_split_by_period(after_text).splitlines() if ln.strip()]
    sm = SequenceMatcher(None, b_lines, a_lines)
    rows = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        elif tag == "replace":
            for k in range(max(i2 - i1, j2 - j1)):
                b_line = b_lines[i1+k] if i1+k < i2 else ""
                a_line = a_lines[j1+k] if j1+k < j2 else ""
                b_html, a_html = highlight_inline(b_line, a_line)
                rows.append(("~", b_html, a_html))
        elif tag == "delete":
            for k in range(i2 - i1):
                rows.append(("-", f"<span class='del'>{escape_html(b_lines[i1+k])}</span>", ""))
        elif tag == "insert":
            for k in range(j2 - j1):
                rows.append(("+", "", f"<span class='ins'>{escape_html(a_lines[j1+k])}</span>"))
    return rows

# ========== CSS ==========
LINE_DIFF_STYLE = """
<style>
.diffrow { display:grid; grid-template-columns:2rem 3rem 1fr 3rem 1fr; gap:.4rem;
           font-family:ui-monospace, Menlo, Consolas, monospace; }
.del { background:#ffd6d6; text-decoration:line-through; }
.ins { background:#d4fcbc; }
</style>
"""

SENT_DIFF_STYLE = """
<style>
.sdiffrow{display:grid;grid-template-columns:2rem 1fr 1fr;gap:.4rem;
          font-family:ui-monospace, Menlo, Consolas, monospace}
.del{background:#ffd6d6;text-decoration:line-through}
.ins{background:#d4fcbc}
</style>
"""
