# pages/100_ログインテスト.py
from __future__ import annotations

from pathlib import Path
import sys
import json
import datetime as dt

import streamlit as st

# ============================================================
# sys.path 調整（pages/13_ボット に倣う）
# ============================================================
_THIS = Path(__file__).resolve()
PROJECTS_ROOT = _THIS.parents[3]
if str(PROJECTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECTS_ROOT))

# ============================================================
# （pages/13 と同じ）ログイン判定
# ============================================================
from common_lib.auth.auth_helpers import get_current_user_from_session_or_cookie  # noqa: E402

# ============================================================
# 追加：config/jwt_utils を “直接” 参照して差分を見る（common_libは改変しない）
# ============================================================
auth_config_loaded = True
auth_config_error = None
try:
    import common_lib.auth.config as auth_config  # noqa: E402
    COOKIE_NAME = getattr(auth_config, "COOKIE_NAME", "prec_sso")
    JWT_SECRET = getattr(auth_config, "JWT_SECRET", None)
    JWT_AUD = getattr(auth_config, "JWT_AUD", None)
    JWT_ISS = getattr(auth_config, "JWT_ISS", None)
    JWT_ALGO = getattr(auth_config, "JWT_ALGO", "HS256")
    auth_config_file = getattr(auth_config, "__file__", None)
except Exception as e:
    auth_config_loaded = False
    auth_config_error = repr(e)
    COOKIE_NAME = "prec_sso"
    JWT_SECRET = None
    JWT_AUD = None
    JWT_ISS = None
    JWT_ALGO = "HS256"
    auth_config_file = None

jwt_utils_loaded = True
jwt_utils_error = None
jwt_utils_file = None
try:
    import common_lib.auth.jwt_utils as jwt_utils  # noqa: E402
    jwt_utils_file = getattr(jwt_utils, "__file__", None)
except Exception as e:
    jwt_utils_loaded = False
    jwt_utils_error = repr(e)

# ============================================================
# extra_streamlit_components の可否
# ============================================================
try:
    import extra_streamlit_components as stx  # type: ignore
    _stx_ok = True
except Exception:
    stx = None  # type: ignore
    _stx_ok = False

# ============================================================
# pyjwt（jwt）で “未検証デコード” と “検証デコード” をする
# ============================================================
pyjwt_ok = True
pyjwt_err = None
try:
    import jwt as pyjwt  # PyJWT
except Exception as e:
    pyjwt_ok = False
    pyjwt_err = repr(e)
    pyjwt = None  # type: ignore


def _get_cookie_token() -> tuple[bool, str | None, str | None]:
    """
    CookieManager 経由で COOKIE_NAME が読めるか確認。
    return: (present, preview, full_token)
    """
    if not _stx_ok or stx is None:
        return False, None, None
    try:
        cm = stx.CookieManager(key="cm_login_test")
        v = cm.get(COOKIE_NAME)
        if isinstance(v, str) and v:
            return True, (v[:12] + "..."), v
        return False, None, None
    except Exception:
        return False, None, None


def _jwt_unverified_payload(token: str) -> dict | None:
    """
    署名検証なしで payload を見る（aud/iss/exp/sub を確認するため）
    """
    if not pyjwt_ok or pyjwt is None:
        return None
    try:
        return pyjwt.decode(token, options={"verify_signature": False})
    except Exception:
        return None


def _jwt_verify_try(token: str) -> tuple[bool, str | None]:
    """
    検証あり decode を試し、失敗理由（例外名）を返す
    """
    if not pyjwt_ok or pyjwt is None:
        return False, "PyJWT import failed"
    if not isinstance(JWT_SECRET, str) or not JWT_SECRET:
        return False, "JWT_SECRET is not a non-empty string"

    try:
        _ = pyjwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGO] if isinstance(JWT_ALGO, str) and JWT_ALGO else ["HS256"],
            audience=JWT_AUD,
            issuer=JWT_ISS,
            options={"require": ["exp", "sub"]},
        )
        return True, None
    except Exception as e:
        # 例外クラス名が最重要（InvalidSignature/Expired/InvalidAudience/InvalidIssuer）
        return False, f"{e.__class__.__name__}: {e}"


def _safe_headers() -> dict:
    """
    st.context.headers は Streamlit のバージョン差があるので安全に取る
    """
    try:
        if hasattr(st, "context"):
            h = getattr(st.context, "headers", None)
            return h or {}
    except Exception:
        pass
    return {}


def _safe_base_url(headers: dict) -> str | None:
    """
    st.context.url は無い Streamlit があるので使わない。
    origin か host から最低限の base を作る。
    """
    try:
        if hasattr(st, "context"):
            u = getattr(st.context, "url", None)  # 無い版がある
            if isinstance(u, str) and u:
                return u
    except Exception:
        pass

    origin = headers.get("origin")
    if isinstance(origin, str) and origin:
        return origin

    host = headers.get("host")
    if isinstance(host, str) and host:
        return f"http://{host}"

    return None


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="ログインテスト", page_icon="🧪", layout="centered")
st.title("🧪 ログインテスト")
st.caption("pages/13_ボット と同じログイン判定＋JWT検証失敗理由まで表示します。")

# 1) まず pages/13 と完全に同じ
current_user, payload = get_current_user_from_session_or_cookie(st)

if current_user:
    st.success(f"✅ ログイン中: **{current_user}**")
else:
    st.warning("⚠️ 未ログイン（ポータルでログイン後に再読み込みしてください）")

st.divider()

# 2) Cookie / JWT の診断
cookie_present, cookie_preview, token_full = _get_cookie_token()
unverified = _jwt_unverified_payload(token_full) if token_full else None
verify_ok, verify_reason = _jwt_verify_try(token_full) if token_full else (False, "No token")

# exp を人間が読める形に（あれば）
exp_human = None
try:
    if isinstance(unverified, dict) and isinstance(unverified.get("exp"), (int, float)):
        exp_human = dt.datetime.fromtimestamp(int(unverified["exp"]), tz=dt.timezone.utc).isoformat()
except Exception:
    exp_human = None

headers = _safe_headers()
base_url = _safe_base_url(headers)

diag = {
    "THIS": str(_THIS),
    "PROJECTS_ROOT": str(PROJECTS_ROOT),

    "current_user": current_user,
    "payload_present": bool(payload),
    "session_current_user": st.session_state.get("current_user"),

    "COOKIE_NAME": COOKIE_NAME,
    "extra_streamlit_components_available": _stx_ok,
    "cookie_present": cookie_present,
    "cookie_preview": cookie_preview,

    "auth_config_loaded": auth_config_loaded,
    "auth_config_error": auth_config_error,
    "auth_config_file": auth_config_file,

    "jwt_utils_loaded": jwt_utils_loaded,
    "jwt_utils_error": jwt_utils_error,
    "jwt_utils_file": jwt_utils_file,

    "JWT_AUD": JWT_AUD,
    "JWT_ISS": JWT_ISS,
    "JWT_ALGO": JWT_ALGO,
    "JWT_SECRET_type": type(JWT_SECRET).__name__ if JWT_SECRET is not None else None,
    "JWT_SECRET_preview": (JWT_SECRET[:8] + "...") if isinstance(JWT_SECRET, str) and JWT_SECRET else None,

    "pyjwt_ok": pyjwt_ok,
    "pyjwt_err": pyjwt_err,

    # 署名検証なしで見た中身（＝auth_portal が発行した payload の事実）
    "jwt_unverified_payload": unverified,
    "jwt_unverified_exp_human_utc": exp_human,

    # 検証ありの結果（＝なぜ通らないかの理由）
    "jwt_verify_ok": verify_ok,
    "jwt_verify_reason": verify_reason,

    "headers_host": headers.get("host"),
    "headers_origin": headers.get("origin"),
    "base_url": base_url,
}

st.subheader("🔍 診断情報（切り分け用）")
st.code(json.dumps(diag, ensure_ascii=False, indent=2), language="json")

st.markdown(
    """
**【見るべき行（ここだけ見ればOK）】**

- `jwt_unverified_payload` の `aud` / `iss` / `exp` / `sub`
- `jwt_verify_reason`
  - `InvalidSignatureError` なら **JWT_SECRET 不一致**
  - `ExpiredSignatureError` なら **期限切れ**
  - `InvalidAudienceError` なら **JWT_AUD 不一致**
  - `InvalidIssuerError` なら **JWT_ISS 不一致**
"""
)
