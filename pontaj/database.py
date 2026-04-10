"""
database.py — SQLite database layer.

Covers:
  - Schema creation & admin seeding (init_db)
  - User CRUD
  - Time-entry CRUD
  - Monthly statistics
"""

import calendar
import sqlite3
from datetime import date, datetime

import bcrypt

from .constants import DB_PATH, DEFAULT_ADMIN_PASS, DEFAULT_ADMIN_USER, ENTRY_TYPES


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create tables if they don't exist and seed the default admin account."""
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                full_name     TEXT    NOT NULL,
                role          TEXT    NOT NULL DEFAULT 'employee'
                              CHECK(role IN ('admin', 'employee')),
                is_active     INTEGER NOT NULL DEFAULT 1,
                created_at    TEXT    NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS time_entries (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                date         TEXT    NOT NULL,
                entry_type   TEXT    NOT NULL
                             CHECK(entry_type IN ('lucrat','liber','concediu','bolnav','absent')),
                start_time   TEXT,
                end_time     TEXT,
                hours_worked REAL    NOT NULL DEFAULT 0.0,
                notes        TEXT,
                created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
                updated_at   TEXT    NOT NULL DEFAULT (datetime('now')),
                UNIQUE (user_id, date)
            );
        """)
        pw_hash = bcrypt.hashpw(DEFAULT_ADMIN_PASS.encode(), bcrypt.gensalt()).decode()
        conn.execute(
            "INSERT OR IGNORE INTO users (username, password_hash, full_name, role) VALUES (?,?,?,?)",
            (DEFAULT_ADMIN_USER, pw_hash, "Administrator", "admin"),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------

def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), hashed.encode())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# User queries
# ---------------------------------------------------------------------------

def get_user_by_username(username: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def has_any_employees() -> bool:
    """True if at least one employee-role user exists (used for first-run detection)."""
    with get_conn() as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM users WHERE role='employee'"
        ).fetchone()[0]
        return count > 0


def get_all_employees() -> list[dict]:
    """Return all active employee-role users, ordered by name."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM users WHERE role='employee' ORDER BY full_name"
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_users() -> list[dict]:
    """Return all users (any role), ordered by role then name."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM users ORDER BY role DESC, full_name"
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# User mutations
# ---------------------------------------------------------------------------

def create_user(username: str, password: str, full_name: str, role: str) -> bool:
    """Returns False if username is already taken."""
    try:
        with get_conn() as conn:
            pw_hash = hash_password(password)
            conn.execute(
                "INSERT INTO users (username, password_hash, full_name, role) VALUES (?,?,?,?)",
                (username, pw_hash, full_name, role),
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def update_user_password(user_id: int, new_pw: str) -> None:
    pw_hash = hash_password(new_pw)
    with get_conn() as conn:
        conn.execute("UPDATE users SET password_hash=? WHERE id=?", (pw_hash, user_id))
        conn.commit()


def toggle_user_active(user_id: int) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE users SET is_active = CASE WHEN is_active=1 THEN 0 ELSE 1 END WHERE id=?",
            (user_id,),
        )
        conn.commit()


def delete_user(user_id: int) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit()


# ---------------------------------------------------------------------------
# Time-entry helpers
# ---------------------------------------------------------------------------

def _compute_hours(start_str: str | None, end_str: str | None) -> float:
    """Return the decimal hours between two 'HH:MM' strings, or 0.0."""
    if not start_str or not end_str:
        return 0.0
    try:
        sh, sm = map(int, start_str.split(":"))
        eh, em = map(int, end_str.split(":"))
        diff = (eh * 60 + em) - (sh * 60 + sm)
        return max(0.0, diff / 60.0)
    except Exception:
        return 0.0


def upsert_time_entry(
    user_id: int,
    date_str: str,
    entry_type: str,
    start_time: str | None,
    end_time: str | None,
    notes: str | None,
) -> None:
    """Insert or update the time entry for a given user and date."""
    hours = _compute_hours(start_time, end_time)
    now = datetime.now().isoformat()
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM time_entries WHERE user_id=? AND date=?", (user_id, date_str)
        ).fetchone()
        if existing:
            conn.execute(
                """UPDATE time_entries
                   SET entry_type=?, start_time=?, end_time=?, hours_worked=?, notes=?, updated_at=?
                   WHERE user_id=? AND date=?""",
                (entry_type, start_time, end_time, hours, notes, now, user_id, date_str),
            )
        else:
            conn.execute(
                """INSERT INTO time_entries
                   (user_id, date, entry_type, start_time, end_time, hours_worked, notes, updated_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (user_id, date_str, entry_type, start_time, end_time, hours, notes, now),
            )
        conn.commit()


def delete_entry(user_id: int, date_str: str) -> None:
    with get_conn() as conn:
        conn.execute(
            "DELETE FROM time_entries WHERE user_id=? AND date=?", (user_id, date_str)
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Time-entry queries
# ---------------------------------------------------------------------------

def _month_range(year: int, month: int) -> tuple[str, str]:
    last_day = calendar.monthrange(year, month)[1]
    return (f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last_day:02d}")


def get_entries_for_user_month(user_id: int, year: int, month: int) -> list[dict]:
    start, end = _month_range(year, month)
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM time_entries WHERE user_id=? AND date BETWEEN ? AND ? ORDER BY date",
            (user_id, start, end),
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_entries_month(year: int, month: int) -> list[dict]:
    """All entries for the month, joined with user full_name."""
    start, end = _month_range(year, month)
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT te.*, u.full_name, u.username
               FROM time_entries te JOIN users u ON te.user_id = u.id
               WHERE te.date BETWEEN ? AND ?
               ORDER BY u.full_name, te.date""",
            (start, end),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_monthly_stats(user_id: int, year: int, month: int) -> dict:
    """
    Returns a dict with keys:
      days_worked, total_hours, liber, concediu, bolnav, absent,
      missing, working_days_in_month
    """
    entries = get_entries_for_user_month(user_id, year, month)
    last_day = calendar.monthrange(year, month)[1]
    working_days_in_month = sum(
        1 for d in range(1, last_day + 1)
        if date(year, month, d).weekday() < 5
    )
    counts = {t: 0 for t in ENTRY_TYPES}
    total_hours = 0.0
    for e in entries:
        t = e["entry_type"]
        if t in counts:
            counts[t] += 1
        total_hours += e["hours_worked"]
    missing = max(0, working_days_in_month - len(entries))
    return {
        "days_worked": counts["lucrat"],
        "total_hours": round(total_hours, 1),
        "liber": counts["liber"],
        "concediu": counts["concediu"],
        "bolnav": counts["bolnav"],
        "absent": counts["absent"],
        "missing": missing,
        "working_days_in_month": working_days_in_month,
    }
