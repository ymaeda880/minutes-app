# lib/audio.py
# ============================================================
# get_audio_duration_seconds(uploaded_file)
# ------------------------------------------------------------
# アップロードされた音声ファイルの再生時間（秒）を推定して返す関数。
# 複数のライブラリを順番に試し、どれも失敗した場合は None を返す。
#
# 【処理の流れ】
# 1) mutagen
#    - mutagen.File() でファイルを解析し、メタ情報 (info.length) があれば返す。
#    - MP3, AAC, FLAC など多くの形式に対応。
#
# 2) wave（WAV専用）
#    - ファイル名が .wav の場合にのみ実行。
#    - wave.open() でフレーム数とサンプリングレートを取得し、
#      duration = frames / rate で秒数を計算。
#
# 3) audioread
#    - 一時ファイルに保存してから audioread.audio_open() で開き、
#      duration 属性があれば返す。
#    - 多くの形式（FFmpegバックエンドなど）に対応。
#
# 4) 全ての方法で失敗した場合
#    - None を返す。
#
# 【特徴】
# - 例外が発生しても握りつぶし、次の手段にフォールバックする安全設計。
# - 複数の形式に対応できる「汎用的な音声長取得ユーティリティ」。
# ============================================================


import io
import os
import tempfile

def get_audio_duration_seconds(uploaded_file) -> float | None:
    """
    mutagen → wave（WAV） → audioread の順で音声長（秒）を推定。
    どれも失敗なら None。
    """
    # 1) mutagen
    try:
        from mutagen import File as MutagenFile
        f = MutagenFile(io.BytesIO(uploaded_file.getbuffer()))
        if getattr(f, "info", None) and getattr(f.info, "length", None):
            return float(f.info.length)
    except Exception:
        pass

    # 2) wave（WAVのみ）
    try:
        import wave, contextlib
        if uploaded_file.name.lower().endswith(".wav"):
            with contextlib.closing(wave.open(io.BytesIO(uploaded_file.getbuffer()))) as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return frames / float(rate)
    except Exception:
        pass

    # 3) audioread
    try:
        import audioread
        suffix = os.path.splitext(uploaded_file.name)[1] or ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name
        try:
            with audioread.audio_open(tmp_path) as af:
                if getattr(af, "duration", None):
                    return float(af.duration)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    except Exception:
        pass

    return None
