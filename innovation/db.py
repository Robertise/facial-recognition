"""
db.py – innovation tables, added beside the attendance schema in the same SQLite file.

The existing tables are not changed except for one added column,
attendance.method ('quick' for face-only check-ins, 'secure' for verified ones).
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Optional

import numpy as np

from backend.database import get_conn

SCHEMA = """
CREATE TABLE IF NOT EXISTS voice_embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id TEXT NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    embedding   BLOB NOT NULL,
    source      TEXT NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS challenges (
    id           TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    digits       TEXT NOT NULL,
    issued_at    REAL NOT NULL,
    expires_at   REAL NOT NULL,
    used         INTEGER NOT NULL DEFAULT 0,
    mode         TEXT NOT NULL DEFAULT 'attendance'
);

-- No foreign key on candidate_id: research records outlive deleted people.
CREATE TABLE IF NOT EXISTS verification_attempts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    challenge_id TEXT,
    candidate_id TEXT,
    decision     TEXT NOT NULL,
    reason_codes TEXT NOT NULL DEFAULT '[]',
    s_face       REAL,
    p_live       REAL,
    s_voice      REAL,
    asr_text     TEXT,
    asr_dist     INTEGER,
    s_av         REAL,
    t_response   REAL,
    latency_ms   REAL,
    attack_label TEXT,
    session_tag  TEXT,
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(row["name"] == column for row in conn.execute(f"PRAGMA table_info({table})"))


def init_innovation_db() -> None:
    """Create the innovation tables and add attendance.method. Safe to run on every start."""
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        if not _has_column(conn, "attendance", "method"):
            conn.execute("ALTER TABLE attendance ADD COLUMN method TEXT NOT NULL DEFAULT 'quick'")


def list_attempts(limit: int = 100, decision: Optional[str] = None,
                  attack_label: Optional[str] = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM verification_attempts"
    where, args = [], []
    if decision:
        where.append("decision = ?"); args.append(decision)
    if attack_label:
        where.append("attack_label = ?"); args.append(attack_label)
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    with get_conn() as conn:
        rows = conn.execute(sql, args).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["reason_codes"] = json.loads(d.get("reason_codes") or "[]")
        out.append(d)
    return out


# ── Voice embeddings ────────────────────────────────────────────────────────

def save_voice_embedding(employee_id: str, embedding: np.ndarray, source: str) -> None:
    with get_conn() as conn:
        conn.execute("INSERT INTO voice_embeddings (employee_id, embedding, source) VALUES (?,?,?)",
                     (employee_id, np.asarray(embedding, np.float32).tobytes(), source))


def get_voice_embeddings(employee_id: str) -> list[np.ndarray]:
    with get_conn() as conn:
        rows = conn.execute("SELECT embedding FROM voice_embeddings WHERE employee_id=? ORDER BY id",
                            (employee_id,)).fetchall()
    return [np.frombuffer(r["embedding"], dtype=np.float32).copy() for r in rows]


def voice_counts() -> dict[str, int]:
    """Number of enrolled voice samples per employee (only employees that have any)."""
    with get_conn() as conn:
        rows = conn.execute("SELECT employee_id, COUNT(*) AS n FROM voice_embeddings GROUP BY employee_id").fetchall()
    return {r["employee_id"]: r["n"] for r in rows}


def delete_voice(employee_id: str) -> int:
    with get_conn() as conn:
        return conn.execute("DELETE FROM voice_embeddings WHERE employee_id=?", (employee_id,)).rowcount


def employee_exists(employee_id: str) -> bool:
    with get_conn() as conn:
        return conn.execute("SELECT 1 FROM employees WHERE id=?", (employee_id,)).fetchone() is not None


# ── Challenges ──────────────────────────────────────────────────────────────

def create_challenge(challenge_id: str, candidate_id: str, digits: str, window_s: float,
                     mode: str = "attendance", now: Optional[float] = None) -> dict[str, Any]:
    issued = time.time() if now is None else now
    row = {"id": challenge_id, "candidate_id": candidate_id, "digits": digits,
           "issued_at": issued, "expires_at": issued + window_s, "used": 0, "mode": mode}
    with get_conn() as conn:
        conn.execute("INSERT INTO challenges (id, candidate_id, digits, issued_at, expires_at, used, mode) "
                     "VALUES (:id, :candidate_id, :digits, :issued_at, :expires_at, :used, :mode)", row)
    return row


def get_challenge(challenge_id: str) -> Optional[dict[str, Any]]:
    with get_conn() as conn:
        r = conn.execute("SELECT * FROM challenges WHERE id=?", (challenge_id,)).fetchone()
    return dict(r) if r else None


def consume_challenge(challenge_id: str) -> bool:
    """Mark a challenge used. False when it was already used (so each one counts once)."""
    with get_conn() as conn:
        return conn.execute("UPDATE challenges SET used=1 WHERE id=? AND used=0", (challenge_id,)).rowcount == 1
