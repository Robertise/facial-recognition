"""
database.py – SQLite database operations
"""
import sqlite3
import numpy as np
from datetime import date
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).parent.parent / "attendance.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id TEXT NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
    angle       TEXT NOT NULL,
    embedding   BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id    TEXT NOT NULL REFERENCES employees(id),
    timestamp      DATETIME DEFAULT CURRENT_TIMESTAMP,
    emotion        TEXT,
    liveness_score REAL,
    similarity     REAL
);
"""

@contextmanager
def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


# ── Employees ─────────────────────────────────────────────────────────────────

def create_employee(emp_id: str, name: str) -> bool:
    try:
        with get_conn() as conn:
            conn.execute("INSERT INTO employees (id, name) VALUES (?,?)", (emp_id, name))
        return True
    except sqlite3.IntegrityError:
        return False


def get_all_employees():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT e.id, e.name, e.created_at, COUNT(em.id) as angle_count "
            "FROM employees e LEFT JOIN embeddings em ON e.id=em.employee_id "
            "GROUP BY e.id ORDER BY e.created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_employee(emp_id: str):
    with get_conn() as conn:
        # Attendance rows reference employees without ON DELETE CASCADE.
        # Remove dependent history in the same transaction before the employee.
        conn.execute("DELETE FROM attendance WHERE employee_id=?", (emp_id,))
        conn.execute("DELETE FROM employees WHERE id=?", (emp_id,))


# ── Embeddings ─────────────────────────────────────────────────────────────────

def save_embedding(employee_id: str, angle: str, embedding: np.ndarray):
    blob = embedding.astype(np.float32).tobytes()
    with get_conn() as conn:
        # Remove old embedding for same angle
        conn.execute(
            "DELETE FROM embeddings WHERE employee_id=? AND angle=?",
            (employee_id, angle)
        )
        conn.execute(
            "INSERT INTO embeddings (employee_id, angle, embedding) VALUES (?,?,?)",
            (employee_id, angle, blob)
        )


def get_all_embeddings() -> list[dict]:
    """Return all embeddings as list of {employee_id, name, angle, embedding (np.ndarray)}."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT em.employee_id, e.name, em.angle, em.embedding "
            "FROM embeddings em JOIN employees e ON em.employee_id=e.id"
        ).fetchall()
    result = []
    for r in rows:
        emb = np.frombuffer(r["embedding"], dtype=np.float32).copy()
        result.append({
            "employee_id": r["employee_id"],
            "name": r["name"],
            "angle": r["angle"],
            "embedding": emb,
        })
    return result


# ── Attendance ─────────────────────────────────────────────────────────────────

def log_attendance(employee_id: str, emotion: str, liveness_score: float, similarity: float):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO attendance (employee_id, emotion, liveness_score, similarity) "
            "VALUES (?,?,?,?)",
            (employee_id, emotion, liveness_score, similarity)
        )


def get_today_attendance():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT a.id, a.employee_id, e.name, a.timestamp, a.emotion, "
            "a.liveness_score, a.similarity "
            "FROM attendance a JOIN employees e ON a.employee_id=e.id "
            "WHERE DATE(a.timestamp)=DATE('now','localtime') "
            "ORDER BY a.timestamp DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_attendance_history(limit: int = 100):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT a.id, a.employee_id, e.name, a.timestamp, a.emotion, "
            "a.liveness_score, a.similarity "
            "FROM attendance a JOIN employees e ON a.employee_id=e.id "
            "ORDER BY a.timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def already_checked_in_today(employee_id: str) -> bool:
    """Returns True if employee already has attendance record today."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT 1 FROM attendance WHERE employee_id=? AND DATE(timestamp)=DATE('now','localtime')",
            (employee_id,)
        ).fetchone()
    return row is not None
