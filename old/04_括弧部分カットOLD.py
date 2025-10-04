import re
from datetime import datetime
from difflib import SequenceMatcher
import streamlit as st

st.set_page_config(page_title="【 】削除ページ", page_icon="✂️", layout="wide")
st.title("✂️ 【 】削除ページ（文分割 → 削除 → 句点改行(任意) → 差分）")

# ===== サイドバー =====
with st.sidebar:
    st.header("前処理（削除前）")
    do_sentence_split = st.checkbox(
        "まず文ごとに改行する（推測して句点補完）",
        value=True,
        help="日本語の句点『。』『？』『！』や閉じ括弧を考慮し、句点が無い場合も推測して句点を補い、1文1行に整えます。"
    )

    st.header("削除の設定（【…】）")
    targets = {"【…】": ("【", "】")}
    enabled_pairs = []
    for label, pair in targets.items():
        if st.checkbox(label, value=True):
            enabled_pairs.append(pair)

    non_greedy = st.checkbox("最短一致で削除", value=True)
    multiline_spanning = st.checkbox("改行をまたぐ括弧も削除", value=True)

    st.header("後処理（削除後）")
    split_after_cleanup = st.checkbox(
        "削除後に句点ごとに改行する",
        value=True,
        help="削除が終わったテキストを、句点（。．.？?！!）の都度で必ず改行します。"
    )
    trim_spaces = st.checkbox("前後の空白を詰める（行strip等）", value=True)
    remove_empty_lines = st.checkbox("空行を詰める（3連→2連）", value=True)

    st.divider()
    st.header("差分表示の設定")
    diff_mode = st.radio(
        "差分ビューの単位",
        ["文単位（句点改行後・変更のみ）", "行単位（通常のBefore/After差分）"],
        index=0,
        help="『文単位』を選ぶと、句点で分割した後に変更された文だけを表示します。"
    )
    show_only_changed = st.checkbox("（行単位のみ）変更行のみ表示", value=False)

# ===== ユーティリティ =====
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

def post_process(text, trim_spaces, remove_empty_lines):
    t = text
    if trim_spaces:
        # 各行strip + 連続空白→1
        t = "\n".join(line.strip() for line in t.splitlines())
        t = re.sub(r"[ \t]{2,}", " ", t)
        # 句読点の直前の不要スペース除去
        t = re.sub(r"([^\s])\s+([、。！？])", r"\1\2", t)
    if remove_empty_lines:
        # 3つ以上の連続改行を2つに圧縮
        t = re.sub(r"\n{3,}", "\n\n", t)
    return t

def add_line_numbers(text):
    lines = text.splitlines()
    return "\n".join(f"{i+1:>4}: {line}" for i, line in enumerate(lines))

def escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

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
            length = max(i2-i1, j2-j1)
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

def render_line_diff(rows, only_changed: bool):
    st.markdown("""
<style>
.diffrow { display:grid; grid-template-columns:2rem 3rem 1fr 3rem 1fr; gap:.4rem;
           font-family:ui-monospace, Menlo, Consolas, monospace; }
.del { background:#ffd6d6; text-decoration:line-through; }
.ins { background:#d4fcbc; }
</style>
""", unsafe_allow_html=True)
    if only_changed:
        rows = [r for r in rows if r[0] != "="]
    st.write("凡例: =同一 ~変更 -削除 +追加")
    for status, b_no, a_no, b_html, a_html in rows:
        st.markdown(
            f"<div class='diffrow'>"
            f"<div>{status}</div><div>{b_no or ''}</div><div>{b_html}</div>"
            f"<div>{a_no or ''}</div><div>{a_html}</div></div>",
            unsafe_allow_html=True
        )

# --- 文分割（句点推測＋補完：削除前の前処理に使用） ---
def sentence_split_with_inferred_periods(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t\u3000]+", " ", t)
    # 既存の文末記号＋閉じ括弧の直後に改行
    t = re.sub(r'([。．\.？\?！!])([」』）】〉》]*)', r'\1\2\n', t)
    # 文頭キュー語の直前に句点補完
    cue = r'(?:はい[、。]?|うん[、。]?|では|それでは|じゃあ|次に|次|さて|まず|あと|続いて|以上です|お願いします|お願いいたします)'
    t = re.sub(r'(?<![。．\.？\?！!])\s+(?=' + cue + r')', '。', t)

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
            for q in str(p).split("\n"):
                q = q.strip()
                if q:
                    fixed.append(q)

    fixed2 = []
    for ln in fixed:
        if not re.search(r'[。．\.？\?！!]$', ln):
            if re.search(polite_end, ln):
                ln = ln + "。"
        fixed2.append(ln)

    out = "\n".join(fixed2)
    out = re.sub(r'\n{2,}', '\n', out).strip()
    return out

# --- 句点ごとに必ず改行（削除後オプション＆文単位差分で利用） ---
def sentence_split_by_period(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r'([。．\.？\?！!])([」』）】〉》]*)', r'\1\2\n', t)
    t = re.sub(r'\n{2,}', '\n', t).strip()
    return t

# ===== 文単位の差分（句点改行後・変更文のみ表示） =====
def build_sentence_diff(before_text, after_text):
    b_lines = [ln for ln in sentence_split_by_period(before_text).splitlines() if ln.strip()]
    a_lines = [ln for ln in sentence_split_by_period(after_text).splitlines() if ln.strip()]
    sm = SequenceMatcher(None, b_lines, a_lines)
    rows = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue  # ★同一文は非表示（＝変更のみ）
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

def render_sentence_diff(rows):
    st.markdown("""
<style>
.sdiffrow{display:grid;grid-template-columns:2rem 1fr 1fr;gap:.4rem;
          font-family:ui-monospace, Menlo, Consolas, monospace}
.del{background:#ffd6d6;text-decoration:line-through}
.ins{background:#d4fcbc}
</style>
""", unsafe_allow_html=True)
    st.write("凡例: ~変更 -削除 +追加（文単位・変更文のみ）")
    for status, b_html, a_html in rows:
        st.markdown(
            f"<div class='sdiffrow'><div>{status}</div><div>{b_html}</div><div>{a_html}</div></div>",
            unsafe_allow_html=True
        )

# ===== 入力 =====
st.subheader("1. 入力")
col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("テキストファイルをアップロード", type=["txt"])
    raw_text = uploaded.read().decode("utf-8", errors="replace") if uploaded else ""
with col2:
    pasted = st.text_area("ここにテキストを貼り付け", height=200)

original_source = raw_text if uploaded else pasted

# ===== パイプライン =====
if not original_source:
    st.info("テキストを入力してください。")
else:
    # 1) 削除前の前処理（任意）：1文1行化（句点推測＋補完）
    pre_source = sentence_split_with_inferred_periods(original_source) if do_sentence_split else original_source

    # プレビュー
    st.subheader("2. 文分割プレビュー（左：未加工 / 右：削除前の前処理後）")
    c0, c1 = st.columns(2)
    with c0:
        st.text_area("原文（未加工）", original_source[:20000], height=220)
    with c1:
        st.text_area("前処理（句点推測＋補完後）", pre_source[:20000], height=220)

    # 2) 【…】削除
    stripped, counts = strip_bracketed(
        pre_source,
        pairs=enabled_pairs if enabled_pairs else [("【", "】")],
        non_greedy=non_greedy,
        multiline_spanning=multiline_spanning
    )

    # 3) 削除後の後処理オプション：句点ごとに改行
    post_source = sentence_split_by_period(stripped) if split_after_cleanup else stripped

    # 4) 整形（空白・空行）
    final_text = post_process(post_source, trim_spaces, remove_empty_lines)

    # ===== 可視化 =====
    st.subheader("3. Before / After")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.text_area("Before（削除前の前処理テキスト）", add_line_numbers(pre_source), height=280, label_visibility="collapsed")
    with cc2:
        st.text_area("After（削除＋後処理テキスト）", add_line_numbers(final_text), height=280, label_visibility="collapsed")

    # 差分ビュー（選択制）
    if diff_mode.startswith("文単位"):
        st.subheader("4. 差分ビュー（文単位：句点改行後・変更文のみ）")
        rows = build_sentence_diff(pre_source, final_text)
        render_sentence_diff(rows)
    else:
        st.subheader("4. 差分ビュー（行単位で比較）")
        rows = build_line_diff(pre_source, final_text)
        render_line_diff(rows, show_only_changed)

    # 5) 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "💾 削除後テキストを保存",
        data=final_text.encode("utf-8"),
        file_name=f"cleaned_{ts}.txt",
        use_container_width=True
    )
