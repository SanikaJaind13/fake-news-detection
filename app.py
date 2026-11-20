"""Streamlit frontend: Evidence-Based Fake News Detection using NLP + RAG."""
from __future__ import annotations
import sys
import os
print("Python Executable:", sys.executable)
print("PYTHONPATH:", os.environ.get("PYTHONPATH"))
print("Sys Path:", sys.path)
import bcrypt
print("Bcrypt location:", bcrypt.__file__)
import html
import json

import streamlit as st

from src import database
from src.auth import (User, Session, register_user, login_user, get_session,
                      get_user, logout_user, change_password, update_username, init_auth_db)
from src.config import load_settings
from src.errors import AuthError, ConfigError, FakeNewsError, InputError
from src.models import CONTRADICTED, INSUFFICIENT, SUPPORTED
from src.ocr import extract_text_from_image, get_ocr_language_options, OCRError
from src.pipeline import analyze

st.set_page_config(page_title="Evidence-Based Fake News Detection", page_icon="🔍", layout="wide")

# --------------------------------------------------------------------------- theme
COLORS = {SUPPORTED: "#1f7a4d", CONTRADICTED: "#b3261e", INSUFFICIENT: "#8a6d1a"}
NEUTRAL = "#6b7280"
EXAMPLES = {
    "Debunked claim": "Humans only use 10 percent of their brains.",
    "Verifiable claim": "India's Chandrayaan-3 landed near the Moon's south pole in August 2023.",
    "Multi-claim text": (
        "The World Health Organization declared COVID-19 a global pandemic on 11 March 2020. "
        "Officials also said that 5G towers spread the coronavirus across Europe. "
        "Vitamin C cures the common cold in 24 hours, according to a viral post."),
}

st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 2.2rem; padding-bottom: 3rem;}
#MainMenu, footer {visibility: hidden;}
h1 {font-weight: 650 !important; letter-spacing: -0.01em; margin-bottom: 0.2rem !important;}
.subtitle {color: #6b7280; font-size: 0.95rem; margin-bottom: 1.2rem;}
.section-label {font-size: 0.72rem; font-weight: 650; letter-spacing: 0.09em; text-transform: uppercase;
                color: #6b7280; margin: 1.1rem 0 0.4rem 0;}

.verdict-card {border: 1px solid rgba(128,128,128,0.28); border-left: 6px solid var(--c);
               border-radius: 8px; padding: 18px 22px; background: rgba(128,128,128,0.05);}
.verdict-kicker {font-size: 0.72rem; font-weight: 650; letter-spacing: 0.09em; text-transform: uppercase; color: #6b7280;}
.verdict-title {font-size: 1.7rem; font-weight: 650; color: var(--c); margin: 2px 0 8px 0;}
.verdict-summary {font-size: 0.95rem; opacity: 0.9;}

.bar {height: 8px; border-radius: 999px; background: rgba(128,128,128,0.22); overflow: hidden; margin-top: 6px;}
.bar > div {height: 100%; background: var(--c); border-radius: 999px;}
.conf-row {display: flex; justify-content: space-between; font-size: 0.8rem; color: #6b7280;}

.tile {border: 1px solid rgba(128,128,128,0.28); border-radius: 8px; padding: 12px 14px; text-align: center;}
.tile-v {font-size: 1.6rem; font-weight: 650; line-height: 1.2;}
.tile-l {font-size: 0.72rem; letter-spacing: 0.06em; text-transform: uppercase; color: #6b7280;}

.pill {display: inline-block; padding: 2px 10px; border-radius: 4px; color: #fff; font-size: 0.78rem;
       font-weight: 600; letter-spacing: 0.02em;}
.claim-text {font-size: 1.02rem; font-weight: 550; margin: 4px 0 10px 0;}
.meta-line {font-size: 0.82rem; color: #6b7280;}

.ev {border: 1px solid rgba(128,128,128,0.25); border-top: 3px solid var(--accent); border-radius: 6px;
     padding: 10px 14px; margin: 6px 0 12px 0; background: rgba(128,128,128,0.04);}
.ev-head {font-size: 0.9rem; font-weight: 600; margin-bottom: 2px;}
.ev-head a {text-decoration: none;}
.ev-head a:hover {text-decoration: underline;}
.eid {display: inline-block; font-size: 0.72rem; font-weight: 700; padding: 1px 7px; margin-right: 8px;
      border-radius: 4px; border: 1px solid var(--accent); color: var(--accent);}
.ev-meta {font-size: 0.76rem; color: #6b7280; margin-bottom: 6px;}
.ev-text {font-size: 0.88rem; line-height: 1.5;}
.hist-meta {font-size: 0.78rem; color: #6b7280;}

/* Image preview styling */
.img-preview {max-width: 100%; border: 1px solid rgba(128,128,128,0.28); border-radius: 8px; margin: 8px 0;}
.ocr-info {font-size: 0.82rem; color: #6b7280; margin: 4px 0;}

/* Auth form styling */
.auth-card {border: 1px solid rgba(128,128,128,0.28); border-radius: 12px; padding: 24px; background: rgba(128,128,128,0.03); max-width: 420px; margin: 2rem auto;}
.auth-title {font-size: 1.4rem; font-weight: 650; text-align: center; margin-bottom: 0.5rem;}
.auth-subtitle {color: #6b7280; text-align: center; margin-bottom: 1.5rem;}
.auth-divider {display: flex; align-items: center; margin: 1rem 0; color: #6b7280; font-size: 0.85rem;}
.auth-divider::before, .auth-divider::after {content: ""; flex: 1; border-bottom: 1px solid rgba(128,128,128,0.28);}
.auth-divider span {padding: 0 1rem;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------- html helpers
def esc(value) -> str:
    return html.escape(str(value), quote=True)


def pill(verdict: str) -> str:
    return f'<span class="pill" style="background:{COLORS.get(verdict, NEUTRAL)}">{esc(verdict)}</span>'


def conf_bar(value: float, color: str) -> str:
    pct = max(0, min(100, round(value * 100)))
    return (f'<div class="conf-row"><span>Confidence</span><span>{pct}%</span></div>'
            f'<div class="bar" style="--c:{color}"><div style="width:{pct}%"></div></div>')


def tile(label: str, value, color: str = "inherit") -> str:
    return (f'<div class="tile"><div class="tile-v" style="color:{color}">{esc(value)}</div>'
            f'<div class="tile-l">{esc(label)}</div></div>')


def evidence_html(p: dict, color: str) -> str:
    url = p["url"] if str(p["url"]).startswith(("http://", "https://")) else "#"
    return (f'<div class="ev" style="--accent:{color}">'
            f'<div class="ev-head"><span class="eid">{esc(p["eid"])}</span>'
            f'<a href="{esc(url)}" target="_blank" rel="noopener noreferrer">{esc(p["title"][:120])}</a></div>'
            f'<div class="ev-meta">{esc(p["domain"])} &nbsp;·&nbsp; {esc(p["credibility_label"])} '
            f'&nbsp;·&nbsp; Relevance {p["similarity"]:.2f}</div>'
            f'<div class="ev-text">{esc(p["text"])}</div></div>')


def evidence_block(title: str, passages: list, color: str) -> None:
    st.markdown(f'<div class="section-label">{esc(title)} ({len(passages)})</div>', unsafe_allow_html=True)
    for p in passages:
        st.markdown(evidence_html(p, color), unsafe_allow_html=True)


# ------------------------------------------------------------------------ results
def render_result(res: dict, key_prefix: str = "main") -> None:
    verdict = res["overall_verdict"]
    color = COLORS.get(verdict, NEUTRAL)
    results = res["claim_results"]

    st.markdown(
        f'<div class="verdict-card" style="--c:{color}">'
        f'<div class="verdict-kicker">Overall assessment</div>'
        f'<div class="verdict-title">{esc(verdict)}</div>'
        f'<div class="verdict-summary">{esc(res["summary"])}</div>'
        f'{conf_bar(res["overall_confidence"], color)}</div>', unsafe_allow_html=True)

    counts = {v: sum(r["verdict"] == v for r in results) for v in (SUPPORTED, CONTRADICTED, INSUFFICIENT)}
    st.write("")
    cols = st.columns(4)
    cols[0].markdown(tile("Claims checked", len(results)), unsafe_allow_html=True)
    cols[1].markdown(tile("Supported", counts[SUPPORTED], COLORS[SUPPORTED]), unsafe_allow_html=True)
    cols[2].markdown(tile("Contradicted", counts[CONTRADICTED], COLORS[CONTRADICTED]), unsafe_allow_html=True)
    cols[3].markdown(tile("Insufficient", counts[INSUFFICIENT], COLORS[INSUFFICIENT]), unsafe_allow_html=True)

    st.markdown('<div class="section-label">Claim-level analysis</div>', unsafe_allow_html=True)
    for cr in results:
        claim = cr["claim"]
        short = claim["text"] if len(claim["text"]) <= 85 else claim["text"][:84] + "…"
        with st.expander(f'{claim["id"]}  ·  {cr["verdict"]}  ·  {short}', expanded=len(results) <= 2):
            c = COLORS.get(cr["verdict"], NEUTRAL)
            left, right = st.columns([3, 1])
            with left:
                st.markdown(f'<div class="meta-line">Claim {esc(claim["id"])}</div>'
                            f'<div class="claim-text">{esc(claim["text"])}</div>'
                            f'{pill(cr["verdict"])}'
                            f'&nbsp;&nbsp;<span class="meta-line">{cr["n_documents"]} pages retrieved</span>',
                            unsafe_allow_html=True)
            with right:
                st.markdown(conf_bar(cr["confidence"], c), unsafe_allow_html=True)

            if cr.get("error"):
                st.warning(f'This claim could not be fully verified: {cr["error"]}')

            st.markdown('<div class="section-label">Reasoning</div>', unsafe_allow_html=True)
            st.write(cr["reasoning"])
            for point in cr["key_points"]:
                st.markdown(f"- {point}")

            sup, con = cr["supporting"], cr["contradicting"]
            if sup and con:
                a, b = st.columns(2)
                with a:
                    evidence_block("Supporting evidence", sup, COLORS[SUPPORTED])
                with b:
                    evidence_block("Contradicting evidence", con, COLORS[CONTRADICTED])
            elif sup:
                evidence_block("Supporting evidence", sup, COLORS[SUPPORTED])
            elif con:
                evidence_block("Contradicting evidence", con, COLORS[CONTRADICTED])
            if cr["other"]:
                with st.expander(f'Other retrieved passages ({len(cr["other"])})'):
                    for p in cr["other"]:
                        st.markdown(evidence_html(p, NEUTRAL), unsafe_allow_html=True)

    st.caption(f'Report generated {res["created_at"]}. Verdicts are produced from retrieved web evidence '
               f'and should be confirmed against the cited sources.')
    st.download_button("Download report (JSON)", json.dumps(res, indent=2, ensure_ascii=False),
                       file_name="fact_check_report.json", mime="application/json",
                       key=f"download_{key_prefix}")


# ------------------------------------------------------------------------ Authentication
def get_current_user(settings) -> User | None:
    """Get current logged-in user from session state."""
    token = st.session_state.get("auth_token")
    if not token:
        return None
    session = get_session(settings.db_path, token)
    if not session:
        st.session_state.pop("auth_token", None)
        st.session_state.pop("current_user", None)
        return None
    user = get_user(settings.db_path, session.user_id)
    if user:
        st.session_state["current_user"] = user
    return user


def render_login_form(settings) -> None:
    """Render login form."""
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">Welcome back</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-subtitle">Sign in to access your verification history</div>', unsafe_allow_html=True)

    with st.form("login_form"):
        email = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Please enter both email and password")
            else:
                try:
                    user, session = login_user(settings.db_path, email, password)
                    st.session_state["auth_token"] = session.token
                    st.session_state["current_user"] = user
                    st.success(f"Welcome back, {user.username}!")
                    st.rerun()
                except AuthError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Login failed: {exc}")

    st.markdown('<div class="auth-divider"><span>or</span></div>', unsafe_allow_html=True)

    if st.button("Create an account", use_container_width=True):
        st.session_state["auth_mode"] = "register"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def render_register_form(settings) -> None:
    """Render registration form."""
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.markdown('<div class="auth-title">Create an account</div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-subtitle">Save your verification history and access it anytime</div>', unsafe_allow_html=True)

    with st.form("register_form"):
        username = st.text_input("Username", placeholder="Choose a username (min 3 chars)")
        email = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password", placeholder="At least 8 characters")
        confirm = st.text_input("Confirm Password", type="password", placeholder="••••••••")
        submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)

        if submitted:
            if not username or not email or not password or not confirm:
                st.error("Please fill in all fields")
            elif password != confirm:
                st.error("Passwords do not match")
            elif len(password) < 8:
                st.error("Password must be at least 8 characters")
            elif len(username) < 3:
                st.error("Username must be at least 3 characters")
            elif "@" not in email:
                st.error("Please enter a valid email address")
            else:
                try:
                    user = register_user(settings.db_path, email, username, password)
                    st.success(f"Account created! Welcome, {user.username}!")
                    st.session_state["auth_mode"] = "login"
                    st.rerun()
                except AuthError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Registration failed: {exc}")

    st.markdown('<div class="auth-divider"><span>or</span></div>', unsafe_allow_html=True)

    if st.button("Back to login", use_container_width=True):
        st.session_state["auth_mode"] = "login"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


def render_user_menu(settings, user: User) -> None:
    """Render user menu in sidebar."""
    with st.sidebar:
        st.markdown("---")
        st.markdown(f"### 👤 {user.username}")
        st.caption(f"{user.email}")

        with st.expander("Account Settings"):
            # Change username
            new_username = st.text_input("Username", value=user.username, key="settings_username")
            if st.button("Update Username", key="update_username"):
                if new_username != user.username:
                    try:
                        update_username(settings.db_path, user.id, new_username)
                        user.username = new_username
                        st.session_state["current_user"] = user
                        st.success("Username updated!")
                        st.rerun()
                    except AuthError as exc:
                        st.error(str(exc))

            st.divider()

            # Change password
            with st.form("change_password_form"):
                old_pwd = st.text_input("Current Password", type="password")
                new_pwd = st.text_input("New Password", type="password")
                confirm_pwd = st.text_input("Confirm New Password", type="password")
                pwd_submitted = st.form_submit_button("Change Password")
                if pwd_submitted:
                    if not old_pwd or not new_pwd or not confirm_pwd:
                        st.error("Please fill all fields")
                    elif new_pwd != confirm_pwd:
                        st.error("New passwords do not match")
                    elif len(new_pwd) < 8:
                        st.error("Password must be at least 8 characters")
                    else:
                        try:
                            change_password(settings.db_path, user.id, old_pwd, new_pwd)
                            st.success("Password changed! Please log in again.")
                            st.session_state.pop("auth_token", None)
                            st.session_state.pop("current_user", None)
                            st.rerun()
                        except AuthError as exc:
                            st.error(str(exc))

            st.divider()

            # Danger zone
            with st.expander("⚠️ Danger Zone"):
                st.caption("This action is irreversible.")
                if st.button("Delete Account", type="secondary"):
                    st.session_state["confirm_delete"] = True

                if st.session_state.get("confirm_delete"):
                    st.warning("Type your password to confirm deletion:")
                    confirm_pwd = st.text_input("Password", type="password", key="delete_confirm_pwd")
                    col1, col2 = st.columns(2)
                    if col1.button("Confirm Delete", type="primary"):
                        try:
                            delete_user(settings.db_path, user.id, confirm_pwd)
                            st.session_state.pop("auth_token", None)
                            st.session_state.pop("current_user", None)
                            st.session_state.pop("confirm_delete", None)
                            st.success("Account deleted.")
                            st.rerun()
                        except AuthError as exc:
                            st.error(str(exc))
                    if col2.button("Cancel"):
                        st.session_state.pop("confirm_delete", None)
                        st.rerun()

        if st.button("Log Out", use_container_width=True):
            logout_user(settings.db_path, st.session_state["auth_token"])
            st.session_state.pop("auth_token", None)
            st.session_state.pop("current_user", None)
            st.rerun()


# ------------------------------------------------------------------------ sidebar (config)
def render_sidebar(settings):
    """Render configuration sidebar."""
    with st.sidebar:
        st.markdown("### Configuration")
        env = load_settings()
        with st.expander("API keys", expanded=not (env.groq_api_key and env.tavily_api_key)):
            st.caption("Keys are read from `.env`. Enter a value only to override it for this session.")
            groq_key = st.text_input("Groq API key", type="password",
                                     placeholder="Loaded from .env" if env.groq_api_key else "gsk_...")
            tavily_key = st.text_input("Tavily API key", type="password",
                                       placeholder="Loaded from .env" if env.tavily_api_key else "tvly-...")
        with st.expander("Analysis settings"):
            model = st.text_input("Groq model", value=env.groq_model)
            max_claims = st.slider("Claims to verify", 1, 5, env.max_claims)
            results_per_claim = st.slider("Web pages per claim", 2, 8, env.results_per_claim)
            passages = st.slider("Evidence passages per claim", 3, 10, env.passages_per_claim)

        settings = load_settings(groq_api_key=groq_key.strip(), tavily_api_key=tavily_key.strip(),
                                 groq_model=model.strip(), max_claims=max_claims,
                                 results_per_claim=results_per_claim, passages_per_claim=passages)
        st.markdown('<div class="section-label">Status</div>', unsafe_allow_html=True)
        for name, ok in (("Groq (LLM)", bool(settings.groq_api_key)),
                         ("Tavily (search)", bool(settings.tavily_api_key))):
            color = COLORS[SUPPORTED] if ok else COLORS[CONTRADICTED]
            st.markdown(f'<span style="color:{color}">●</span> {name}: '
                        f'{"Configured" if ok else "Missing key"}', unsafe_allow_html=True)
        st.divider()
        st.caption("For educational use. Results are AI-generated and may contain errors.")
    return settings


# --------------------------------------------------------------------------- main
def main():
    # Initialize session state
    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "login"

    st.title("Evidence-Based Fake News Detection")
    st.markdown('<div class="subtitle">Claim verification with retrieved evidence and cited reasoning</div>',
                unsafe_allow_html=True)

    # Load settings (needed for auth)
    env = load_settings()
    settings = load_settings()

    # Check authentication
    user = get_current_user(settings)

    if not user:
        # Show auth forms
        if st.session_state["auth_mode"] == "register":
            render_register_form(settings)
        else:
            render_login_form(settings)
        return

    # User is logged in - show main app
    render_user_menu(settings, user)
    settings = render_sidebar(settings)

    # Main tabs
    tab_check, tab_image, tab_history, tab_method = st.tabs(["Fact-check", "Image Verify", "History", "Methodology"])

    # ============================ TAB 1: FACT-CHECK (Text/URL) ============================
    with tab_check:
        def _set_example(text: str) -> None:
            st.session_state["input_text"] = text

        def _clear() -> None:
            st.session_state["input_text"] = ""
            st.session_state.pop("result", None)

        st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)
        user_text = st.text_area("Headline, claim, article text or URL", key="input_text", height=160,
                                 placeholder="Paste a headline, claim, article or link to verify…",
                                 label_visibility="collapsed")

        ex_cols = st.columns([1, 1, 1, 1.4])
        for col, (name, text) in zip(ex_cols, EXAMPLES.items()):
            col.button(name, on_click=_set_example, args=(text,), key=f"ex_{name}")

        act1, act2, _ = st.columns([1, 1, 6])
        run = act1.button("Verify", type="primary")
        act2.button("Clear", on_click=_clear)

        if run:
            bar, status = st.progress(0.0), st.empty()

            def on_progress(frac: float, msg: str) -> None:
                bar.progress(min(max(frac, 0.0), 1.0))
                status.caption(msg)

            try:
                result = analyze(user_text, settings, progress=on_progress, user_id=user.id)
                st.session_state["result"] = result.to_dict()
            except (InputError, ConfigError) as exc:
                st.session_state.pop("result", None)
                st.warning(str(exc))
            except FakeNewsError as exc:
                st.session_state.pop("result", None)
                st.error(f"The check could not be completed. {exc}")
            except Exception as exc:
                st.session_state.pop("result", None)
                st.error(f"Unexpected error: {exc}")
            finally:
                bar.empty()
                status.empty()

        if "result" in st.session_state:
            st.divider()
            render_result(st.session_state["result"])

    # ============================ TAB 2: IMAGE VERIFY ============================
    with tab_image:
        st.markdown('<div class="section-label">Upload a news image</div>', unsafe_allow_html=True)
        st.caption("Upload a screenshot, social media post, newspaper clipping, or news headline image. "
                   "Supported formats: PNG, JPG, JPEG. OCR languages: English, Hindi, Marathi (if installed).")

        # Language selection
        lang_options = get_ocr_language_options()
        available_langs = [opt["name"] for opt in lang_options if opt["available"]]
        unavailable_langs = [opt["name"] for opt in lang_options if not opt["available"]]

        if unavailable_langs:
            st.warning(f"⚠️ Language packs not installed: {', '.join(unavailable_langs)}. "
                       "Install `tesseract-ocr-hin` and `tesseract-ocr-mar` for Hindi/Marathi support.")

        selected_langs = st.multiselect(
            "OCR Languages",
            options=[opt["name"] for opt in lang_options if opt["available"]],
            default=available_langs,
            help="Select languages for OCR. English is always recommended as a base."
        )

        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=False,
            key="image_uploader"
        )

        # Session state for image processing
        if "image_ocr_text" not in st.session_state:
            st.session_state["image_ocr_text"] = ""
        if "image_ocr_result" not in st.session_state:
            st.session_state["image_ocr_result"] = None
        if "image_preview" not in st.session_state:
            st.session_state["image_preview"] = None

        if uploaded_file is not None:
            # Read image bytes
            image_bytes = uploaded_file.read()

            # Display image preview
            st.markdown('<div class="section-label">Image Preview</div>', unsafe_allow_html=True)
            st.image(image_bytes, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

            # Extract text button
            if st.button("Extract Text (OCR)", type="secondary", key="ocr_extract"):
                with st.spinner("Running OCR..."):
                    try:
                        ocr_result = extract_text_from_image(image_bytes, languages=selected_langs)
                        st.session_state["image_ocr_result"] = ocr_result
                        st.session_state["image_ocr_text"] = ocr_result.text
                        st.success(f"✅ OCR completed! Language: {ocr_result.language}, Confidence: {ocr_result.confidence}%")
                    except OCRError as exc:
                        st.session_state["image_ocr_result"] = None
                        st.session_state["image_ocr_text"] = ""
                        st.error(f"OCR failed: {exc}")
                    except Exception as exc:
                        st.session_state["image_ocr_result"] = None
                        st.session_state["image_ocr_text"] = ""
                        st.error(f"Unexpected OCR error: {exc}")

        # Show extracted text editor
        if st.session_state["image_ocr_text"]:
            st.markdown('<div class="section-label">Extracted Text (editable)</div>', unsafe_allow_html=True)

            edited_text = st.text_area(
                "Edit the extracted text before verification",
                value=st.session_state["image_ocr_text"],
                height=150,
                key="image_ocr_editor",
                label_visibility="collapsed"
            )
            st.session_state["image_ocr_text"] = edited_text

            # OCR info
            if st.session_state["image_ocr_result"]:
                ocr_res = st.session_state["image_ocr_result"]
                st.markdown(f'<div class="ocr-info">OCR Language: {ocr_res.language} · Average Confidence: {ocr_res.confidence}%</div>',
                            unsafe_allow_html=True)

            # Verify button
            verify_col, _ = st.columns([1, 5])
            if verify_col.button("Verify News", type="primary", key="image_verify"):
                if not edited_text.strip():
                    st.warning("Please enter or extract some text to verify.")
                else:
                    bar, status = st.progress(0.0), st.empty()

                    def on_progress(frac: float, msg: str) -> None:
                        bar.progress(min(max(frac, 0.0), 1.0))
                        status.caption(msg)

                    try:
                        result = analyze(edited_text, settings, progress=on_progress, user_id=user.id)
                        st.session_state["image_result"] = result.to_dict()
                    except (InputError, ConfigError) as exc:
                        st.session_state.pop("image_result", None)
                        st.warning(str(exc))
                    except FakeNewsError as exc:
                        st.session_state.pop("image_result", None)
                        st.error(f"The check could not be completed. {exc}")
                    except Exception as exc:
                        st.session_state.pop("image_result", None)
                        st.error(f"Unexpected error: {exc}")
                    finally:
                        bar.empty()
                        status.empty()

        # Show results for image verification
        if "image_result" in st.session_state:
            st.divider()
            render_result(st.session_state["image_result"], key_prefix="image")

    # ============================ TAB 3: HISTORY ============================
    with tab_history:
        try:
            rows = database.list_history(settings.db_path, user_id=user.id, limit=100)
        except Exception as exc:
            rows = []
            st.error(f"Could not read history: {exc}")

        if not rows:
            st.info("No saved checks yet. Completed analyses appear here automatically.")
        else:
            st.markdown(f'<div class="section-label">Saved analyses ({len(rows)})</div>', unsafe_allow_html=True)
            for r in rows:
                with st.container(border=True):
                    c1, c2, c3 = st.columns([7, 1, 1])
                    preview = r["input_text"][:110] + ("…" if len(r["input_text"]) > 110 else "")
                    c1.markdown(f'{pill(r["overall_verdict"])} &nbsp;<span class="meta-line">'
                                f'{r["overall_confidence"]:.0%} confidence</span><br>{esc(preview)}<br>'
                                f'<span class="hist-meta">#{r["id"]} · {esc(r["created_at"])}</span>',
                                unsafe_allow_html=True)
                    if c2.button("Open", key=f'open_{r["id"]}'):
                        st.session_state["history_view"] = database.get_analysis(settings.db_path, r["id"], user.id)
                    if c3.button("Delete", key=f'del_{r["id"]}'):
                        database.delete_analysis(settings.db_path, r["id"], user.id)
                        view = st.session_state.get("history_view")
                        if view and view.get("id") == r["id"]:
                            st.session_state.pop("history_view", None)
                        st.rerun()

            with st.expander("Danger zone"):
                if st.checkbox("I understand this permanently deletes all my saved analyses"):
                    if st.button("Clear all my history"):
                        database.clear_history(settings.db_path, user.id)
                        st.session_state.pop("history_view", None)
                        st.rerun()

            if st.session_state.get("history_view"):
                st.divider()
                render_result(st.session_state["history_view"], key_prefix="history")

    # ============================ TAB 4: METHODOLOGY ============================
    with tab_method:
        st.markdown("""
| Stage | Technique |
|---|---|
| 1. Claim extraction | Regex sentence segmentation with a check-worthiness score (numbers, dates, named entities, factual verbs; penalties for opinions and questions) |
| 2. Evidence retrieval | Tavily web search, followed by page download and text extraction (Requests, BeautifulSoup) |
| 3. Evidence ranking | Sentence-window passages ranked by TF-IDF cosine similarity combined with number and entity coverage |
| 4. Verdict generation | Retrieval-Augmented Generation: top passages are supplied to a Groq-hosted LLM that returns a cited JSON verdict |
| 5. Validation | Citations are checked against retrieved passages; verdicts without valid citations are downgraded to *Insufficient Evidence* |
| 6. Aggregation | Any contradicted claim → *Contradicted*; all claims supported → *Supported*; otherwise *Insufficient Evidence* |

**Image Verification Pipeline:**
1. User uploads an image (PNG/JPG/JPEG)
2. Tesseract OCR extracts text (supports English, Hindi, Marathi)
3. User reviews and edits the extracted text
4. Text enters the existing verification pipeline (steps 1-6 above)

**Limitations.** TF-IDF matches vocabulary rather than meaning, so heavily paraphrased evidence may be
ranked lower. Web sources may themselves contain misinformation, source reliability is a domain-based
heuristic, and language models can make errors. OCR accuracy depends on image quality, resolution, and
text clarity. Always confirm important findings against the cited sources.
""")


if __name__ == "__main__":
    main()