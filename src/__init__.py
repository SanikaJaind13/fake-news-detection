"""NEWSINTEL AI - Evidence-Based Fake News Detection Package."""

from .auth import (User, Session, register_user, login_user, get_session,
                   get_user, logout_user, logout_all_sessions, cleanup_expired_sessions,
                   change_password, update_username, delete_user, init_auth_db)
from .config import Settings, load_settings
from .database import (clear_history, delete_analysis, get_analysis, init_db,
                       list_history, save_analysis)
from .errors import (AuthError, ConfigError, FakeNewsError, InputError, LLMError,
                     RetrievalError)
from .models import (CONTRADICTED, INSUFFICIENT, SUPPORTED, AnalysisResult,
                     Claim, ClaimResult, Document, Passage)
from .ocr import OCRResult, extract_text_from_image, get_ocr_language_options
from .pipeline import analyze

__all__ = [
    # auth
    "User",
    "Session",
    "register_user",
    "login_user",
    "get_session",
    "get_user",
    "logout_user",
    "logout_all_sessions",
    "cleanup_expired_sessions",
    "change_password",
    "update_username",
    "delete_user",
    "init_auth_db",
    # config
    "Settings",
    "load_settings",
    # database
    "init_db",
    "save_analysis",
    "list_history",
    "get_analysis",
    "delete_analysis",
    "clear_history",
    # errors
    "AuthError",
    "FakeNewsError",
    "ConfigError",
    "InputError",
    "RetrievalError",
    "LLMError",
    # models
    "SUPPORTED",
    "CONTRADICTED",
    "INSUFFICIENT",
    "Claim",
    "Document",
    "Passage",
    "ClaimResult",
    "AnalysisResult",
    # ocr
    "OCRResult",
    "extract_text_from_image",
    "get_ocr_language_options",
    # pipeline
    "analyze",
]