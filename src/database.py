"""SQLite storage for search history (standard-library sqlite3, no ORM)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    input_text TEXT NOT NULL,
    overall_verdict TEXT NOT NULL,
    overall_confidence REAL NOT NULL,
    summary TEXT NOT NULL,
    result_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_user_created ON history (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_history_user ON history (user_id);
"""


def _connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    with closing(_connect(db_path)) as conn, conn:
        conn.executescript(SCHEMA)
        # Migration: add user_id column if it doesn't exist (for existing databases)
        try:
            conn.execute("SELECT user_id FROM history LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE history ADD COLUMN user_id INTEGER DEFAULT 1")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_user_created ON history (user_id, created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_user ON history (user_id)")


def save_analysis(db_path: str, result: dict, user_id: int = 1) -> int:
    init_db(db_path)
    with closing(_connect(db_path)) as conn, conn:
        cur = conn.execute(
            "INSERT INTO history (user_id, created_at, input_text, overall_verdict, "
            "overall_confidence, summary, result_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, result["created_at"], result["input_text"], result["overall_verdict"],
             float(result["overall_confidence"]), result["summary"],
             json.dumps(result, ensure_ascii=False)))
        return int(cur.lastrowid)


def list_history(db_path: str, user_id: int, limit: int = 50) -> list[dict]:
    init_db(db_path)
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT id, created_at, input_text, overall_verdict, overall_confidence "
            "FROM history WHERE user_id = ? ORDER BY id DESC LIMIT ?", (user_id, limit)).fetchall()
    return [dict(r) for r in rows]


def get_analysis(db_path: str, analysis_id: int, user_id: int) -> dict | None:
    init_db(db_path)
    with closing(_connect(db_path)) as conn:
        row = conn.execute("SELECT result_json FROM history WHERE id = ? AND user_id = ?",
                           (analysis_id, user_id)).fetchone()
    if not row:
        return None
    data = json.loads(row["result_json"])
    data["id"] = analysis_id
    return data


def delete_analysis(db_path: str, analysis_id: int, user_id: int) -> bool:
    init_db(db_path)
    with closing(_connect(db_path)) as conn, conn:
        return conn.execute("DELETE FROM history WHERE id = ? AND user_id = ?", (analysis_id, user_id)).rowcount > 0


def clear_history(db_path: str, user_id: int) -> int:
    init_db(db_path)
    with closing(_connect(db_path)) as conn, conn:
        return conn.execute("DELETE FROM history WHERE user_id = ?", (user_id,)).rowcount
