"""
database.py
------------
Handles all SQLite persistence for InterviewIQ: creating tables and
storing / retrieving interview session history.

Why SQLite?
A single-file, zero-configuration relational database is ideal for a
B.Tech project running locally in VS Code -- no server setup required,
and it still lets us demonstrate real relational storage + SQL queries.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
DB_PATH = os.path.join(DB_DIR, "interviewiq.db")


def ensure_data_dir():
    os.makedirs(DB_DIR, exist_ok=True)


@contextmanager
def get_connection():
    ensure_data_dir()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they do not already exist. Safe to call every run."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                candidate_name TEXT,
                interview_type TEXT,
                difficulty TEXT,
                num_questions INTEGER,
                questions_answered INTEGER,
                overall_score REAL,
                technical_score REAL,
                communication_score REAL,
                problem_solving_score REAL,
                confidence_score REAL,
                resume_alignment_score REAL,
                strengths TEXT,
                weaknesses TEXT,
                skill_gaps_json TEXT,
                qa_history_json TEXT,
                resume_summary_json TEXT,
                jd_summary_json TEXT,
                learning_plan_json TEXT
            )
            """
        )


def save_session(session_data: dict) -> int:
    """Persist a completed (or in-progress) interview session. Returns row id."""
    init_db()
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO sessions (
                created_at, candidate_name, interview_type, difficulty,
                num_questions, questions_answered, overall_score, technical_score,
                communication_score, problem_solving_score, confidence_score,
                resume_alignment_score, strengths, weaknesses, skill_gaps_json,
                qa_history_json, resume_summary_json, jd_summary_json, learning_plan_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().isoformat(timespec="seconds"),
                session_data.get("candidate_name", "Candidate"),
                session_data.get("interview_type", ""),
                session_data.get("difficulty", ""),
                session_data.get("num_questions", 0),
                session_data.get("questions_answered", 0),
                session_data.get("overall_score", 0),
                session_data.get("technical_score", 0),
                session_data.get("communication_score", 0),
                session_data.get("problem_solving_score", 0),
                session_data.get("confidence_score", 0),
                session_data.get("resume_alignment_score", 0),
                json.dumps(session_data.get("strengths", [])),
                json.dumps(session_data.get("weaknesses", [])),
                json.dumps(session_data.get("skill_gaps", {})),
                json.dumps(session_data.get("qa_history", [])),
                json.dumps(session_data.get("resume_summary", {})),
                json.dumps(session_data.get("jd_summary", {})),
                json.dumps(session_data.get("learning_plan", [])),
            ),
        )
        return cur.lastrowid


def get_all_sessions() -> list:
    init_db()
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM sessions ORDER BY id DESC").fetchall()
        return [dict(r) for r in rows]


def get_session(session_id: int) -> dict:
    init_db()
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return dict(row) if row else {}


def delete_session(session_id: int):
    init_db()
    with get_connection() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


def session_count() -> int:
    init_db()
    with get_connection() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()
        return row["c"] if row else 0
