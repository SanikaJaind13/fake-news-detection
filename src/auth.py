"""User authentication: registration, login, session management.

Uses bcrypt for password hashing, SQLite for user storage, and
Streamlit session state for login persistence.
"""
from __future__ import annotations

import hashlib
import secrets
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import bcrypt

from .config import Settings
from .database import _connect
from .errors import AuthError, ConfigError


@dataclass
class User:
    """User account."""
    id: int
    email: str
    username: str
    created_at: str
    last_login: str | None = None


@dataclass
class Session:
    """Login session."""
    token: str
    user_id: int
    created_at: str
    expires_at: str


# Session settings
SESSION_DAYS = 30
TOKEN_BYTES = 32


def _hash_password(password: str) -> str:
    """Hash password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def _verify_password(password: str, hashed: str) -> bool:
    """Verify password against bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def _generate_token() -> str:
    """Generate a secure random session token."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def init_auth_db(db_path: str) -> None:
    """Initialize auth tables in the database."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(db_path)) as conn, conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_login TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);
        CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);

        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions (user_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions (expires_at);
        """)


def register_user(db_path: str, email: str, username: str, password: str) -> User:
    """Register a new user.

    Args:
        db_path: Path to SQLite database
        email: User's email (unique)
        username: User's username (unique)
        password: Plain-text password (will be hashed)

    Returns:
        Created User object

    Raises:
        AuthError: If email/username already exists or validation fails
    """
    init_auth_db(db_path)

    email = email.strip().lower()
    username = username.strip()

    if not email or "@" not in email:
        raise AuthError("Invalid email address")
    if not username or len(username) < 3:
        raise AuthError("Username must be at least 3 characters")
    if not password or len(password) < 8:
        raise AuthError("Password must be at least 8 characters")

    password_hash = _hash_password(password)
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with closing(_connect(db_path)) as conn, conn:
        try:
            cur = conn.execute(
                "INSERT INTO users (email, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (email, username, password_hash, created_at)
            )
            user_id = int(cur.lastrowid)
        except sqlite3.IntegrityError as exc:
            if "email" in str(exc).lower():
                raise AuthError("Email already registered")
            elif "username" in str(exc).lower():
                raise AuthError("Username already taken")
            raise AuthError("Registration failed: " + str(exc))

    return User(
        id=user_id,
        email=email,
        username=username,
        created_at=created_at,
        last_login=None
    )


def login_user(db_path: str, email: str, password: str) -> tuple[User, Session]:
    """Authenticate user and create session.

    Args:
        db_path: Path to SQLite database
        email: User's email
        password: Plain-text password

    Returns:
        Tuple of (User, Session)

    Raises:
        AuthError: If credentials are invalid
    """
    init_auth_db(db_path)
    email = email.strip().lower()

    with closing(_connect(db_path)) as conn:
        row = conn.execute(
            "SELECT id, email, username, password_hash, created_at, last_login "
            "FROM users WHERE email = ?",
            (email,)
        ).fetchone()

    if not row:
        raise AuthError("Invalid email or password")

    if not _verify_password(password, row["password_hash"]):
        raise AuthError("Invalid email or password")

    # Create session
    token = _generate_token()
    now = datetime.now(timezone.utc)
    expires = now + timedelta(days=SESSION_DAYS)

    with closing(_connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, row["id"], now.strftime("%Y-%m-%d %H:%M:%S UTC"),
             expires.strftime("%Y-%m-%d %H:%M:%S UTC"))
        )
        # Update last_login
        conn.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (now.strftime("%Y-%m-%d %H:%M:%S UTC"), row["id"])
        )

    user = User(
        id=row["id"],
        email=row["email"],
        username=row["username"],
        created_at=row["created_at"],
        last_login=now.strftime("%Y-%m-%d %H:%M:%S UTC")
    )
    session = Session(
        token=token,
        user_id=row["id"],
        created_at=now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        expires_at=expires.strftime("%Y-%m-%d %H:%M:%S UTC")
    )
    return user, session


def get_session(db_path: str, token: str) -> Optional[Session]:
    """Get valid session by token.

    Returns None if session doesn't exist or is expired.
    """
    init_auth_db(db_path)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    with closing(_connect(db_path)) as conn:
        row = conn.execute(
            "SELECT token, user_id, created_at, expires_at "
            "FROM sessions WHERE token = ? AND expires_at > ?",
            (token, now)
        ).fetchone()

    if not row:
        return None

    return Session(
        token=row["token"],
        user_id=row["user_id"],
        created_at=row["created_at"],
        expires_at=row["expires_at"]
    )


def get_user(db_path: str, user_id: int) -> Optional[User]:
    """Get user by ID."""
    init_auth_db(db_path)
    with closing(_connect(db_path)) as conn:
        row = conn.execute(
            "SELECT id, email, username, created_at, last_login FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()

    if not row:
        return None

    return User(
        id=row["id"],
        email=row["email"],
        username=row["username"],
        created_at=row["created_at"],
        last_login=row["last_login"]
    )


def logout_user(db_path: str, token: str) -> bool:
    """Delete a session (logout)."""
    init_auth_db(db_path)
    with closing(_connect(db_path)) as conn, conn:
        return conn.execute("DELETE FROM sessions WHERE token = ?", (token,)).rowcount > 0


def logout_all_sessions(db_path: str, user_id: int) -> int:
    """Delete all sessions for a user (logout everywhere)."""
    init_auth_db(db_path)
    with closing(_connect(db_path)) as conn, conn:
        cur = conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        return cur.rowcount


def cleanup_expired_sessions(db_path: str) -> int:
    """Remove expired sessions. Returns count deleted."""
    init_auth_db(db_path)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    with closing(_connect(db_path)) as conn, conn:
        cur = conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
        return cur.rowcount


def change_password(db_path: str, user_id: int, old_password: str, new_password: str) -> bool:
    """Change user's password after verifying old password."""
    init_auth_db(db_path)

    if not new_password or len(new_password) < 8:
        raise AuthError("New password must be at least 8 characters")

    with closing(_connect(db_path)) as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,)).fetchone()

    if not row or not _verify_password(old_password, row["password_hash"]):
        raise AuthError("Current password is incorrect")

    new_hash = _hash_password(new_password)
    with closing(_connect(db_path)) as conn, conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id))

    # Invalidate all existing sessions (force re-login)
    logout_all_sessions(db_path, user_id)
    return True


def update_username(db_path: str, user_id: int, new_username: str) -> bool:
    """Update username if not taken."""
    init_auth_db(db_path)
    new_username = new_username.strip()

    if not new_username or len(new_username) < 3:
        raise AuthError("Username must be at least 3 characters")

    with closing(_connect(db_path)) as conn, conn:
        try:
            conn.execute("UPDATE users SET username = ? WHERE id = ?", (new_username, user_id))
        except sqlite3.IntegrityError:
            raise AuthError("Username already taken")
    return True


def delete_user(db_path: str, user_id: int, password: str) -> bool:
    """Delete user account after verifying password."""
    init_auth_db(db_path)

    with closing(_connect(db_path)) as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,)).fetchone()

    if not row or not _verify_password(password, row["password_hash"]):
        raise AuthError("Password is incorrect")

    with closing(_connect(db_path)) as conn, conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    return True