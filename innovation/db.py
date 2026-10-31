"""
db.py – innovation tables, added beside the attendance schema in the same SQLite file.

The existing tables are not changed except for one added column,
attendance.method ('quick' for face-only check-ins, 'secure' for verified ones).
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Optional

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
