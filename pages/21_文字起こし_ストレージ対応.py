# pages/21_文字起こし_storage対応.py
# ============================================================
# 目的：
#   Storages/<user>/minutes_app/<YYYY-MM-DD>/<job_...>/split/ のチャンクを選び、
#   OpenAI Transcribe / Whisper / GPT-4o Transcribe または Gemini で連続文字起こし。
#   結果を job 配下の transcript/ に保存（個別＋combined）。
#
# UI 方針（康男さん指定）：
#   ①ジョブ選択 → メイン（中央）
#   ②チャンク選択・各種設定 → サイドバー
#   実行ボタン → メイン（使いやすさ優先）
#   結果表示 → メイン
# ============================================================

from __future__ import annotations

import io
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
import streamlit as st
import shutil

from config.config import (
    # keys / endpoints
    get_openai_api_key,
    get_gemini_api_key,
    has_gemini_api_key,
    OPENAI_TRANSCRIBE_URL,
    # prices
    WHISPER_PRICE_PER_MIN,
    TRANSCRIBE_PRICES_USD_PER_MIN,
    DEFAULT_USDJPY,
    # gemini cost helpers
    estimate_tokens_from_text,
    estimate_gemini_cost_usd,
)

from lib.audio import get_audio_duration_seconds
from lib.explanation import render_transcribe_continuous_expander


# ============================================================
# sys.path 調整 & ログイン判定（共通）
# ============================================================
import sys

_THIS = Path(__file__).resolve()
PROJECTS_ROOT = _THIS.parents[3]
if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

from common_lib.storage.external_ssd_root import resolve_storage_subdir_root
from common_lib.auth.auth_helpers import require_login

# ============================================================
# Storage root
# ============================================================
STORAGE_ROOT = resolve_storage_subdir_root(
    PROJECTS_ROOT,
    subdir="Storages",
)

# ============================================================
# ページ設定
# ============================================================
st.set_page_config(page_title="01 文字起こし — Storage Jobs", layout="wide")

sub = require_login(st)
if not sub:
    st.stop()
left, right = st.columns([2, 1])
with left:
    st.title("文字起こし（ストレージ対応")
with right:
    st.success(f"✅ ログイン中: **{sub}**")
current_user=sub

render_transcribe_continuous_expander()

# ============================================================
# ログイン
# ============================================================


# ============================================================
# ユーザー名のフォルダ安全化（pages/20 と合わせる）
# ============================================================
def _sanitize_username_for_path(username: str) -> str:
    u = (username or "").strip()
    if not u:
        return "anonymous"
    u = re.sub(r"[^0-9A-Za-z_-]+", "_", u).strip("_")
    return u or "anonymous"


USERNAME_DIR = _sanitize_username_for_path(str(current_user))
USER_ROOT = STORAGE_ROOT / USERNAME_DIR / "minutes_app"


# ============================================================
# OpenAI / Gemini キー
# ============================================================
OPENAI_API_KEY = get_openai_api_key()
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY が .streamlit/secrets.toml に設定されていません。")
    st.stop()

GEMINI_ENABLED = has_gemini_api_key()
GEMINI_API_KEY = get_gemini_api_key() if GEMINI_ENABLED else ""

# ============================================================
# セッション初期化
# ============================================================
st.session_state.setdefault("usd_jpy", float(DEFAULT_USDJPY))
st.session_state.setdefault("model_last_valid", "whisper-1")
st.session_state.setdefault("model_picker", "whisper-1")
st.session_state.setdefault("gemini_disabled_notice", False)

# ============================================================
# モデル・プロンプト等
# ============================================================
BRACKET_TAG_PATTERN = re.compile(r"【[^】]*】")


def strip_bracket_tags(text: str) -> str:
    if not text:
        return text
    return BRACKET_TAG_PATTERN.sub("", text)


PROMPT_OPTIONS = [
    "",
    "出力に話者名や【】などのラベルを入れない。音声に無い単語は書かない。",
    "人名やプロジェクト名は正確に出力してください。専門用語はカタカナで。",
    "句読点を正しく付与し、自然な文章にしてください。",
]

MODEL_OPTIONS = [
    "whisper-1",
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe",
    "gemini-2.0-flash",
]


def model_label(x: str) -> str:
    if x.startswith("gemini") and not GEMINI_ENABLED:
        return f"{x}（GEMINI_API_KEY 未設定）"
    return x


def on_change_model_picker():
    picked = st.session_state.get("model_picker", "whisper-1")
    if picked.startswith("gemini") and not GEMINI_ENABLED:
        st.session_state["gemini_disabled_notice"] = True
        st.session_state["model_picker"] = st.session_state.get("model_last_valid", "whisper-1")
    else:
        st.session_state["model_last_valid"] = picked
        st.session_state["gemini_disabled_notice"] = False


# ============================================================
# Job / split スキャン
# ============================================================
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".mp4", ".webm", ".ogg"}


@dataclass
class JobInfo:
    date_dir: str
    job_id: str
    job_dir: Path
    split_dir: Path
    transcript_dir: Path
    transcript_marked_dir: Path
    logs_dir: Path
    job_json: Optional[dict]

    @property
    def label(self) -> str:
        base = f"{self.date_dir}/{self.job_id}"
        try:
            if self.job_json:
                orig = self.job_json.get("paths", {}).get("original")
                if orig:
                    return f"{base}  （original: {Path(orig).name}）"
        except Exception:
            pass
        return base


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_log(log_path: Path, msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    safe_mkdir(log_path.parent)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")


def read_json_if_exists(p: Path) -> Optional[dict]:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def scan_jobs(user_root: Path) -> List[JobInfo]:
    jobs: List[JobInfo] = []
    if not user_root.exists():
        return jobs

    for date_dir in sorted([d for d in user_root.iterdir() if d.is_dir()], reverse=True):
        for job_dir in sorted([j for j in date_dir.iterdir() if j.is_dir() and j.name.startswith("job_")], reverse=True):
            split_dir = job_dir / "split"
            transcript_dir = job_dir / "transcript"
            transcript_marked_dir = job_dir / "transcript_marked"
            logs_dir = job_dir / "logs"
            job_json = read_json_if_exists(job_dir / "job.json")

            jobs.append(
                JobInfo(
                    date_dir=date_dir.name,
                    job_id=job_dir.name,
                    job_dir=job_dir,
                    split_dir=split_dir,
                    transcript_dir=transcript_dir,
                    transcript_marked_dir=transcript_marked_dir,
                    logs_dir=logs_dir,
                    job_json=job_json,
                )
            )
    return jobs


def list_split_audio_files(split_dir: Path) -> List[Path]:
    if not split_dir.exists():
        return []
    out = []
    for p in sorted(split_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            out.append(p)
    return out


# ============================================================
# ① ジョブ選択（メイン）
# ============================================================
jobs_all = scan_jobs(USER_ROOT)

#
# 古いジョブを物理的に消去
#
MAX_JOBS = 5
jobs_keep = jobs_all[:MAX_JOBS]
jobs_delete = jobs_all[MAX_JOBS:]

for j in jobs_delete:
    try:
        # job フォルダ削除
        shutil.rmtree(j.job_dir)

        # ★ 追加：親の日付フォルダが空なら削除
        date_dir = j.job_dir.parent
        if date_dir.exists() and not any(date_dir.iterdir()):
            date_dir.rmdir()

    except Exception as e:
        st.warning(f"削除失敗: {j.job_dir} ({e})")

jobs = jobs_keep


#
# ジョブ選択
#
st.subheader("ジョブ選択（storage）")
st.write("直近の５つのジョブより古いものは自動的に消去されます")

if not jobs:
    st.warning(f"ジョブが見つかりません: {USER_ROOT}\n先に「音声分割（storage）」で job を作成してください。")
    st.stop()

job_labels = [j.label for j in jobs]
picked_label = st.radio(
    "対象ジョブ",
    options=job_labels,
    index=0,
    help="音声分割で作成された job フォルダを選びます（split/ を参照）。",
)

picked_job = jobs[job_labels.index(picked_label)]

split_files = list_split_audio_files(picked_job.split_dir)
st.caption(f"選択中の job: {picked_job.job_dir}")

if not split_files:
    st.warning("この job には split/ の音声チャンクが見つかりません。")
    st.stop()

# ============================================================
# サイドバー：②チャンク選択〜③通貨換算（※実行ボタンはメイン）
# ============================================================
with st.sidebar:
    st.header("チャンク選択（split/）")

    options = [p.name for p in split_files]
    selected_names = st.multiselect(
        "処理するチャンク（選択順に連続文字起こし）",
        options=options,
        default=options,
    )

    st.divider()
    st.header("モデル")
    st.radio(
        "モデル",
        options=MODEL_OPTIONS,
        key="model_picker",
        format_func=model_label,
        on_change=on_change_model_picker,
    )

    if st.session_state.get("gemini_disabled_notice", False) and not GEMINI_ENABLED:
        st.warning("GEMINI_API_KEY が未設定のため、Gemini は選択できません。")

    model = st.session_state["model_picker"]

    st.divider()
    st.header("返却形式・言語・プロンプト")
    fmt = st.selectbox("返却形式（OpenAI response_format）", ["json", "text", "srt", "vtt"], index=0)
    language = st.text_input("言語コード（未指定なら自動判定）", value="ja")
    prompt_hint = st.selectbox("Transcribeプロンプト（省略可）", options=PROMPT_OPTIONS, index=0)
    do_strip_brackets = st.checkbox("書き起こし後に【…】を除去する", value=True)

    st.divider()
    st.header("③ 通貨換算（任意）")
    usd_jpy = st.number_input(
        "USD/JPY",
        min_value=50.0,
        max_value=500.0,
        value=float(st.session_state.get("usd_jpy", DEFAULT_USDJPY)),
        step=0.5,
    )
    st.session_state["usd_jpy"] = float(usd_jpy)

# ============================================================
# メイン：実行ボタン
# ============================================================
st.subheader("実行")
go = st.button("▶️ 文字起こしを実行（選択順）+ 保存", type="primary")

out_area = st.container()

def guess_mime(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".mp3":
        return "audio/mpeg"
    if suf == ".wav":
        return "audio/wav"
    if suf == ".m4a":
        return "audio/mp4"
    if suf == ".mp4":
        return "video/mp4"
    if suf == ".webm":
        return "audio/webm"
    if suf == ".ogg":
        return "audio/ogg"
    return "application/octet-stream"


def save_text(p: Path, text: str) -> None:
    safe_mkdir(p.parent)
    p.write_text(text, encoding="utf-8")


# ============================================================
# 実行部
# ============================================================
if go:
    if not selected_names:
        st.warning("チャンクを1つ以上選んでください。")
        st.stop()

    if model.startswith("gemini") and not GEMINI_ENABLED:
        st.error("GEMINI_API_KEY が未設定のため、Gemini は利用できません。")
        st.stop()

    safe_mkdir(picked_job.transcript_dir)
    safe_mkdir(picked_job.transcript_marked_dir)  # 次工程用（22）
    safe_mkdir(picked_job.logs_dir)

    job_log_path = picked_job.logs_dir / "process.log"
    append_log(job_log_path, "TRANSCRIBE START")
    append_log(job_log_path, f"job={picked_job.job_dir}")
    append_log(job_log_path, f"model={model} fmt={fmt} language={language!r}")
    append_log(job_log_path, f"selected={selected_names}")

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    sess = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset({"POST"}),
    )
    sess.mount("https://", HTTPAdapter(max_retries=retries))

    USE_GEMINI = model.startswith("gemini")
    if USE_GEMINI:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel(model)

    name_to_path = {p.name: p for p in split_files}
    targets = [name_to_path[n] for n in selected_names if n in name_to_path]

    progress = st.progress(0, text="準備中…")

    per_file_results: List[dict] = []
    combined_parts: List[str] = []
    total_elapsed = 0.0

    for idx, path in enumerate(targets, start=1):
        progress.progress((idx - 1) / len(targets), text=f"{idx}/{len(targets)} 処理中: {path.name}")

        file_bytes = path.read_bytes()
        mime = guess_mime(path)

        audio_sec = audio_min = None
        try:
            audio_sec = get_audio_duration_seconds(io.BytesIO(file_bytes))
            audio_min = (audio_sec / 60.0) if audio_sec else None
        except Exception:
            pass

        t0 = time.perf_counter()

        if USE_GEMINI:
            instr_parts = [
                "この音声を日本語で正確に文字起こししてください。",
                "日本語は分かち書きにしないでください（単語の間に不要な半角スペースを入れない）。",
                "句読点（、。）を適切に補い、自然な文章として出力してください。",
            ]
            if language and language.strip():
                instr_parts.append(f"言語コードは {language.strip()} を優先（不明なら自動判定）。")
            if prompt_hint and prompt_hint.strip():
                instr_parts.append(prompt_hint.strip())
            instruction = " ".join(instr_parts)

            with st.spinner(f"Gemini 文字起こし中…（{path.name}）"):
                response = gemini_model.generate_content(
                    [
                        instruction,
                        {"mime_type": mime, "data": file_bytes},
                    ]
                )
            text = getattr(response, "text", "") or ""
            req_id = "gemini"
        else:
            files = {"file": (path.name, file_bytes, mime)}
            data: dict = {"model": model, "response_format": fmt}
            if prompt_hint and prompt_hint.strip():
                data["prompt"] = prompt_hint.strip()
            if language and language.strip():
                data["language"] = language.strip()

            with st.spinner(f"Transcribe API に送信中…（{path.name}）"):
                resp = sess.post(
                    OPENAI_TRANSCRIBE_URL,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=600,
                )

            req_id = resp.headers.get("x-request-id")

            if not resp.ok:
                st.error(f"{path.name}: APIエラー: {resp.status_code}\n{resp.text}\nrequest-id: {req_id}")
                append_log(job_log_path, f"ERROR {path.name} status={resp.status_code} req_id={req_id}")
                continue

            if fmt == "json":
                try:
                    text = resp.json().get("text", "")
                except Exception:
                    text = resp.text
            else:
                text = resp.text

        elapsed = time.perf_counter() - t0
        total_elapsed += elapsed

        if do_strip_brackets and text:
            text = strip_bracket_tags(text)

        usd = jpy = None
        in_tok = out_tok = None

        if USE_GEMINI:
            out_tok = estimate_tokens_from_text(text)
            in_tok = out_tok
            usd_est = estimate_gemini_cost_usd(model=model, input_tokens=in_tok, output_tokens=out_tok)
            if usd_est is not None:
                usd = float(usd_est)
                jpy = usd * float(st.session_state["usd_jpy"])
        else:
            if audio_min is not None:
                price_per_min = TRANSCRIBE_PRICES_USD_PER_MIN.get(model, WHISPER_PRICE_PER_MIN)
                usd = float(audio_min) * float(price_per_min)
                jpy = usd * float(st.session_state["usd_jpy"])

        # 保存（個別）
        out_txt = picked_job.transcript_dir / f"{idx:03d}_{path.stem}.txt"
        save_text(out_txt, text or "")
        append_log(job_log_path, f"SAVED transcript {path.name} -> {out_txt.name}")

        with out_area:
            st.markdown(f"### 📁 {idx}. {path.name}")
            st.text_area("テキスト（個別）", value=text, height=220, key=f"ta_{idx}")

            cost_str = "—"
            if usd is not None and jpy is not None:
                cost_str = f"${usd:,.6f} / ¥{jpy:,.2f}"
            elif USE_GEMINI:
                cost_str = "—（Gemini：モデル単価未設定 or 推定不能）"

            metrics_data = {
                "処理時間": [f"{elapsed:.2f} 秒"],
                "音声長": [f"{audio_sec:.1f} 秒 / {audio_min:.2f} 分" if audio_sec else "—"],
                "概算 (USD/JPY)": [cost_str],
                "推定tokens(in/out)": [f"{in_tok}/{out_tok}" if USE_GEMINI and in_tok is not None else "—"],
                "request-id": [req_id or "—"],
                "モデル": [model],
                "保存先": [str(out_txt)],
            }
            st.table(pd.DataFrame(metrics_data))

        per_file_results.append(
            dict(
                name=path.name,
                text=text,
                sec=audio_sec,
                min=audio_min,
                usd=usd,
                jpy=jpy,
                elapsed=elapsed,
                req_id=req_id,
                in_tok=in_tok,
                out_tok=out_tok,
                out_txt=str(out_txt),
            )
        )

        combined_parts.append(text or "")
        if idx < len(targets):
            combined_parts.append(f"\n\n----- ここがつなぎ目です（{path.name} と次のファイルの間）-----\n\n")

    progress.progress(1.0, text="完了")

   
    # ============================================================
    # combined 保存（base_name + _combined を job_XX/transcript_combined/ に保存）
    # ============================================================

    combined_text = "".join(combined_parts)

    # base_name を split ファイル名から復元する
    # split ファイル名は pages/20 で
    #   {base_name}_part{i:03d}_{start}-{end}.{ext}
    # なので stem の "_part" より前が base_name
    if targets:
        first_stem = targets[0].stem  # 例: R4あり方検討会..._part000_000000-000300
        base_name = first_stem.split("_part", 1)[0] if "_part" in first_stem else first_stem
    else:
        base_name = "audio"  # 保険（通常は targets は必ずある）

    # 保存先フォルダ：job_XX 直下に transcript_combined を作って入れる
    combined_dir = picked_job.job_dir / "transcript_combined"
    safe_mkdir(combined_dir)

    combined_name = f"{base_name}_combined.txt"
    combined_txt_path = combined_dir / combined_name

    save_text(combined_txt_path, combined_text or "")
    append_log(job_log_path, f"SAVED combined -> {combined_txt_path}")
    st.success(f"✅ ストレージに保存しました: {combined_txt_path.name}")
    st.caption(f"保存先: {combined_txt_path}")




    with out_area:
        st.markdown("---")
        st.subheader("🔗 連結テキスト（全チャンク）")
        st.text_area("テキスト（連結済み）", value=combined_text, height=350)

        st.download_button(
            "🧩 連結テキスト（.txt）を（パソコンに）ダウンロード",
            data=(combined_text or "").encode("utf-8"),
            file_name="transcripts_combined.txt",
            mime="text/plain",
        )

        total_sec = sum([r["sec"] for r in per_file_results if r["sec"] is not None]) if per_file_results else None
        total_min = sum([r["min"] for r in per_file_results if r["min"] is not None]) if per_file_results else None
        total_usd = sum([r["usd"] for r in per_file_results if r["usd"] is not None]) if per_file_results else None
        total_jpy = sum([r["jpy"] for r in per_file_results if r["jpy"] is not None]) if per_file_results else None

        st.subheader("📊 料金の概要（合算）")
        df_total = pd.DataFrame(
            {
                "チャンク数": [len(per_file_results)],
                "合計処理時間": [f"{total_elapsed:.2f} 秒"],
                "合計音声長": [f"{total_sec:.1f} 秒 / {total_min:.2f} 分" if total_sec else "—"],
                "合計概算 (USD/JPY)": [
                    f"${total_usd:,.6f} / ¥{total_jpy:,.2f}" if total_usd is not None else "—"
                ],
                "モデル": [model],
                "備考": ["Gemini は tokens 推定による概算" if USE_GEMINI else "OpenAI は分単価による概算"],
                "保存先(combined)": [str(combined_txt_path)],
            }
        )
        st.table(df_total)

        if per_file_results:
            st.caption("チャンク別サマリー")
            df_each = pd.DataFrame(
                [
                    {
                        "チャンク": r["name"],
                        "処理時間(秒)": round(r["elapsed"], 2),
                        "音声長(分)": (round(r["min"], 2) if r["min"] is not None else None),
                        "推定tokens(in/out)": (f"{r['in_tok']}/{r['out_tok']}" if r["in_tok"] is not None else None),
                        "概算USD": (round(r["usd"], 6) if r["usd"] is not None else None),
                        "概算JPY": (round(r["jpy"], 2) if r["jpy"] is not None else None),
                        "request-id": r["req_id"] or "—",
                        "保存先": r["out_txt"],
                    }
                    for r in per_file_results
                ]
            )
            st.dataframe(df_each)

    # 次ページ引き継ぎ（必要なら）
    st.session_state["transcribed_texts"] = [r["text"] for r in per_file_results]
    st.session_state["transcribed_text"] = combined_text
    st.session_state["picked_job_dir"] = str(picked_job.job_dir)
    st.session_state["combined_txt_path"] = str(combined_txt_path)
