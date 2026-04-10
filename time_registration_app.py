"""
Pontaj - Time Registration App for Romanian Companies
======================================================
A Streamlit-based time registration system with two roles:
  - admin:    overview of all employees, stats, charts, employee management
  - employee: self-service monthly time registration (worked / free / vacation / sick)

Default admin login:  admin / Admin1234!
Run with:  streamlit run time_registration_app.py
"""

# ==============================================================================
# SECTION 1 — IMPORTS & CONSTANTS
# ==============================================================================

import calendar
import io
import os
import sqlite3
from datetime import date, datetime, time, timedelta

import bcrypt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "time_registration.db")
DEFAULT_ADMIN_USER = "admin"
DEFAULT_ADMIN_PASS = "Admin1234!"

ENTRY_TYPES = {
    "lucrat":   "Lucrat",
    "liber":    "Liber",
    "concediu": "Concediu",
    "bolnav":   "Bolnav",
    "absent":   "Absent",
}

ENTRY_COLORS = {
    "lucrat":   "#4CAF50",
    "liber":    "#2196F3",
    "concediu": "#FF9800",
    "bolnav":   "#F44336",
    "absent":   "#9E9E9E",
}

ENTRY_EMOJIS = {
    "lucrat":   "✅",
    "liber":    "🏖️",
    "concediu": "🌴",
    "bolnav":   "🤒",
    "absent":   "❌",
}

RO_DAYS = ["Lu", "Ma", "Mi", "Jo", "Vi", "Sâ", "Du"]
RO_DAYS_FULL = ["Luni", "Marți", "Miercuri", "Joi", "Vineri", "Sâmbătă", "Duminică"]
RO_MONTHS = [
    "", "Ianuarie", "Februarie", "Martie", "Aprilie", "Mai", "Iunie",
    "Iulie", "August", "Septembrie", "Octombrie", "Noiembrie", "Decembrie",
]


# ==============================================================================
# SECTION 2 — DATABASE LAYER
# ==============================================================================

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
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
        # Seed default admin
        pw_hash = bcrypt.hashpw(DEFAULT_ADMIN_PASS.encode(), bcrypt.gensalt()).decode()
        conn.execute(
            "INSERT OR IGNORE INTO users (username, password_hash, full_name, role) VALUES (?,?,?,?)",
            (DEFAULT_ADMIN_USER, pw_hash, "Administrator", "admin"),
        )
        conn.commit()


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), hashed.encode())
    except Exception:
        return False


def get_user_by_username(username: str):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def get_all_employees():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM users WHERE role='employee' ORDER BY full_name"
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_users():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY role DESC, full_name").fetchall()
        return [dict(r) for r in rows]


def create_user(username: str, password: str, full_name: str, role: str) -> bool:
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
        conn.execute(
            "UPDATE users SET password_hash=? WHERE id=?", (pw_hash, user_id)
        )
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


def _compute_hours(start_str: str | None, end_str: str | None) -> float:
    if not start_str or not end_str:
        return 0.0
    try:
        sh, sm = map(int, start_str.split(":"))
        eh, em = map(int, end_str.split(":"))
        start_mins = sh * 60 + sm
        end_mins = eh * 60 + em
        diff = end_mins - start_mins
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


def get_entries_for_user_month(user_id: int, year: int, month: int) -> list[dict]:
    month_start = f"{year:04d}-{month:02d}-01"
    last_day = calendar.monthrange(year, month)[1]
    month_end = f"{year:04d}-{month:02d}-{last_day:02d}"
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM time_entries WHERE user_id=? AND date BETWEEN ? AND ? ORDER BY date",
            (user_id, month_start, month_end),
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_entries_month(year: int, month: int) -> list[dict]:
    month_start = f"{year:04d}-{month:02d}-01"
    last_day = calendar.monthrange(year, month)[1]
    month_end = f"{year:04d}-{month:02d}-{last_day:02d}"
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT te.*, u.full_name, u.username
               FROM time_entries te JOIN users u ON te.user_id = u.id
               WHERE te.date BETWEEN ? AND ?
               ORDER BY u.full_name, te.date""",
            (month_start, month_end),
        ).fetchall()
        return [dict(r) for r in rows]


def compute_monthly_stats(user_id: int, year: int, month: int) -> dict:
    entries = get_entries_for_user_month(user_id, year, month)
    # Count working days (Mon-Fri) in the month
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
    days_with_entry = len(entries)
    missing = max(0, working_days_in_month - days_with_entry)
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


# ==============================================================================
# SECTION 3 — AUTH HELPERS
# ==============================================================================

def login(username: str, password: str):
    user = get_user_by_username(username)
    if user and user["is_active"] and verify_password(password, user["password_hash"]):
        return user
    return None


def logout() -> None:
    for key in ["user", "selected_date", "admin_selected_employee", "show_entry_form"]:
        st.session_state[key] = None if key != "show_entry_form" else False
    st.rerun()


def init_session_state() -> None:
    today = date.today()
    defaults = {
        "user": None,
        "emp_year": today.year,
        "emp_month": today.month,
        "adm_year": today.year,
        "adm_month": today.month,
        "selected_date": None,
        "admin_selected_employee": None,
        "show_entry_form": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ==============================================================================
# SECTION 4 — SHARED UI COMPONENTS
# ==============================================================================

def render_month_navigator(prefix: str) -> tuple[int, int]:
    year_key = f"{prefix}_year"
    month_key = f"{prefix}_month"
    year = st.session_state[year_key]
    month = st.session_state[month_key]

    col_prev, col_title, col_next = st.columns([1, 4, 1])
    with col_prev:
        if st.button("◀", key=f"{prefix}_prev"):
            if month == 1:
                st.session_state[year_key] = year - 1
                st.session_state[month_key] = 12
            else:
                st.session_state[month_key] = month - 1
            st.rerun()
    with col_title:
        st.markdown(
            f"<h2 style='text-align:center; margin:0'>{RO_MONTHS[month]} {year}</h2>",
            unsafe_allow_html=True,
        )
    with col_next:
        if st.button("▶", key=f"{prefix}_next"):
            if month == 12:
                st.session_state[year_key] = year + 1
                st.session_state[month_key] = 1
            else:
                st.session_state[month_key] = month + 1
            st.rerun()

    return st.session_state[year_key], st.session_state[month_key]


def _inject_calendar_css() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stHorizontalBlock"] > div { padding: 1px !important; }
        div.cal-day button {
            width: 100% !important;
            height: 64px !important;
            font-size: 12px !important;
            padding: 2px !important;
            border-radius: 6px !important;
            white-space: pre-wrap !important;
        }
        div.cal-header { text-align: center; font-weight: bold; padding: 4px 0; font-size: 13px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_calendar_grid(
    entries_map: dict,   # date_str -> entry dict
    year: int,
    month: int,
    clickable: bool = True,
    selected_date: str | None = None,
    key_prefix: str = "cal",
) -> None:
    _inject_calendar_css()

    # Weekday header row
    header_cols = st.columns(7)
    for i, day_name in enumerate(RO_DAYS):
        with header_cols[i]:
            color = "#c00" if i >= 5 else "#333"
            st.markdown(
                f"<div class='cal-header' style='color:{color}'>{day_name}</div>",
                unsafe_allow_html=True,
            )

    first_weekday, last_day = calendar.monthrange(year, month)
    # first_weekday: 0=Monday … 6=Sunday

    day_num = 1
    cells = []  # list of (day_number_or_None)
    for _ in range(first_weekday):
        cells.append(None)
    while day_num <= last_day:
        cells.append(day_num)
        day_num += 1
    # Pad to full weeks
    while len(cells) % 7 != 0:
        cells.append(None)

    today_str = date.today().isoformat()

    for week_start in range(0, len(cells), 7):
        week_cells = cells[week_start: week_start + 7]
        cols = st.columns(7)
        for col_idx, d in enumerate(week_cells):
            with cols[col_idx]:
                if d is None:
                    st.markdown("<div style='height:64px'></div>", unsafe_allow_html=True)
                    continue

                date_str = f"{year:04d}-{month:02d}-{d:02d}"
                entry = entries_map.get(date_str)
                etype = entry["entry_type"] if entry else None
                color = ENTRY_COLORS.get(etype, "#EEEEEE")
                is_weekend = col_idx >= 5
                is_today = date_str == today_str
                is_selected = date_str == selected_date

                if is_weekend:
                    bg = "#f5dde0" if not etype else color
                elif etype:
                    bg = color
                else:
                    bg = "#f9f9f9"

                border = "3px solid #1a73e8" if is_selected else ("2px solid #aaa" if is_today else "1px solid #ddd")
                text_color = "#fff" if etype and not is_weekend else "#333"

                if etype == "lucrat" and entry:
                    label = f"{d}\n{entry.get('start_time','?')}-{entry.get('end_time','?')}\n{entry.get('hours_worked',0):.1f}h"
                elif etype:
                    label = f"{d}\n{ENTRY_EMOJIS.get(etype,'')}\n{ENTRY_TYPES[etype]}"
                else:
                    label = f"{d}"

                if clickable and not is_weekend:
                    st.markdown(f"<div class='cal-day'>", unsafe_allow_html=True)
                    clicked = st.button(
                        label,
                        key=f"{key_prefix}_{date_str}",
                        help=f"Click to register {date_str}",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                    # Apply background color via JS-free CSS trick (button background)
                    st.markdown(
                        f"""<style>
                        div[data-testid="stButton"] > button[kind="secondary"][title="Click to register {date_str}"] {{
                            background-color: {bg} !important;
                            border: {border} !important;
                            color: {text_color} !important;
                        }}
                        </style>""",
                        unsafe_allow_html=True,
                    )
                    if clicked:
                        st.session_state["selected_date"] = date_str
                        st.session_state["show_entry_form"] = True
                        st.rerun()
                else:
                    # Read-only cell
                    st.markdown(
                        f"""<div style='
                            background:{bg};border:{border};border-radius:6px;
                            min-height:64px;padding:4px;text-align:center;
                            font-size:12px;color:{text_color};
                            display:flex;flex-direction:column;
                            align-items:center;justify-content:center;
                            white-space:pre-wrap;
                        '>{label}</div>""",
                        unsafe_allow_html=True,
                    )


def render_entry_form(user_id: int, date_str: str, existing: dict | None) -> None:
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    day_name = RO_DAYS_FULL[d.weekday()]
    month_name = RO_MONTHS[d.month]
    st.markdown(f"### {day_name}, {d.day} {month_name} {d.year}")

    current_type = existing["entry_type"] if existing else "lucrat"
    type_labels = list(ENTRY_TYPES.values())
    type_keys = list(ENTRY_TYPES.keys())
    current_idx = type_keys.index(current_type) if current_type in type_keys else 0

    entry_type = st.selectbox(
        "Tip intrare",
        options=type_keys,
        format_func=lambda k: f"{ENTRY_EMOJIS[k]} {ENTRY_TYPES[k]}",
        index=current_idx,
        key=f"form_type_{date_str}",
    )

    start_t = end_t = None
    if entry_type == "lucrat":
        col1, col2 = st.columns(2)
        with col1:
            default_start = time(9, 0)
            if existing and existing.get("start_time"):
                h, m = map(int, existing["start_time"].split(":"))
                default_start = time(h, m)
            start_t = st.time_input("Ora start", value=default_start, key=f"form_start_{date_str}")
        with col2:
            default_end = time(17, 0)
            if existing and existing.get("end_time"):
                h, m = map(int, existing["end_time"].split(":"))
                default_end = time(h, m)
            end_t = st.time_input("Ora sfarsit", value=default_end, key=f"form_end_{date_str}")

        if start_t and end_t and end_t <= start_t:
            st.error("Ora de sfarsit trebuie sa fie dupa ora de start.")
            return

    notes = st.text_area(
        "Note (optional)",
        value=existing.get("notes", "") if existing else "",
        key=f"form_notes_{date_str}",
        height=68,
    )

    col_save, col_del, col_cancel = st.columns([2, 1, 1])
    with col_save:
        if st.button("💾 Salveaza", key=f"save_{date_str}", type="primary"):
            start_str = start_t.strftime("%H:%M") if start_t else None
            end_str = end_t.strftime("%H:%M") if end_t else None
            upsert_time_entry(user_id, date_str, entry_type, start_str, end_str, notes or None)
            st.success("Inregistrat!")
            st.session_state["selected_date"] = None
            st.session_state["show_entry_form"] = False
            st.rerun()
    with col_del:
        if existing and st.button("🗑️ Sterge", key=f"del_{date_str}"):
            delete_entry(user_id, date_str)
            st.session_state["selected_date"] = None
            st.session_state["show_entry_form"] = False
            st.rerun()
    with col_cancel:
        if st.button("✖ Anuleaza", key=f"cancel_{date_str}"):
            st.session_state["selected_date"] = None
            st.session_state["show_entry_form"] = False
            st.rerun()


def render_summary_cards(stats: dict) -> None:
    cols = st.columns(6)
    cards = [
        ("🕐 Ore lucrate",  f"{stats['total_hours']:.1f}h",  None),
        ("✅ Zile lucrate", stats["days_worked"],              None),
        ("🏖️ Liber",        stats["liber"],                   None),
        ("🌴 Concediu",     stats["concediu"],                None),
        ("🤒 Bolnav",       stats["bolnav"],                  None),
        ("⚠️ Nelucrate",    stats["missing"],                 "off"),
    ]
    for col, (label, val, delta_color) in zip(cols, cards):
        with col:
            st.metric(label, val)


# ==============================================================================
# SECTION 5 — EMPLOYEE VIEW
# ==============================================================================

def render_employee_view(user: dict) -> None:
    with st.sidebar:
        st.markdown(f"### 👤 {user['full_name']}")
        st.caption(f"@{user['username']}")
        st.divider()
        if st.button("🚪 Deconectare", use_container_width=True):
            logout()

    st.title("📋 Pontaj Personal")

    year, month = render_month_navigator("emp")
    st.divider()

    entries = get_entries_for_user_month(user["id"], year, month)
    entries_map = {e["date"]: e for e in entries}
    stats = compute_monthly_stats(user["id"], year, month)

    render_summary_cards(stats)
    st.divider()

    selected = st.session_state.get("selected_date")

    render_calendar_grid(
        entries_map=entries_map,
        year=year,
        month=month,
        clickable=True,
        selected_date=selected,
        key_prefix="emp_cal",
    )

    if selected and st.session_state.get("show_entry_form"):
        st.divider()
        existing = entries_map.get(selected)
        with st.container():
            render_entry_form(user["id"], selected, existing)


# ==============================================================================
# SECTION 6 — ADMIN VIEW
# ==============================================================================

def render_admin_view(user: dict) -> None:
    with st.sidebar:
        st.markdown(f"### 🔧 {user['full_name']}")
        st.caption("Administrator")
        st.divider()
        if st.button("🚪 Deconectare", use_container_width=True):
            logout()

    st.title("📊 Panou Administrare Pontaj")

    tab1, tab2, tab3 = st.tabs([
        "📊 Prezentare Generala",
        "📅 Calendar Angajat",
        "👥 Gestionare Angajati",
    ])

    # ── TAB 1: OVERVIEW ──────────────────────────────────────────────────────
    with tab1:
        year, month = render_month_navigator("adm")
        st.divider()

        employees = get_all_employees()
        if not employees:
            st.info("Nu exista angajati inregistrati. Adaugati angajati in tab-ul 'Gestionare Angajati'.")
        else:
            all_entries = get_all_entries_month(year, month)

            # Build stats per employee
            rows = []
            for emp in employees:
                stats = compute_monthly_stats(emp["id"], year, month)
                rows.append({
                    "Angajat": emp["full_name"],
                    "Zile Lucrate": stats["days_worked"],
                    "Ore Totale": stats["total_hours"],
                    "Liber": stats["liber"],
                    "Concediu": stats["concediu"],
                    "Bolnav": stats["bolnav"],
                    "Absent": stats["absent"],
                    "Nelucrate": stats["missing"],
                })
            df = pd.DataFrame(rows)

            st.subheader(f"Raport {RO_MONTHS[month]} {year}")

            # Styled table
            def highlight_row(val, col):
                if col == "Nelucrate" and val > 0:
                    return "background-color: #fff3cd"
                if col == "Ore Totale" and val > 0:
                    return "background-color: #d4edda"
                return ""

            st.dataframe(
                df.style.format({"Ore Totale": "{:.1f}"}),
                use_container_width=True,
                hide_index=True,
            )

            # CSV export
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False)
            st.download_button(
                "⬇️ Export CSV",
                data=csv_buf.getvalue(),
                file_name=f"pontaj_{year}_{month:02d}.csv",
                mime="text/csv",
            )

            st.divider()

            # Bar chart: hours per employee
            if not df.empty and df["Ore Totale"].sum() > 0:
                fig_bar = px.bar(
                    df,
                    x="Angajat",
                    y="Ore Totale",
                    color="Ore Totale",
                    color_continuous_scale="Greens",
                    title=f"Ore lucrate per angajat — {RO_MONTHS[month]} {year}",
                    text="Ore Totale",
                )
                fig_bar.update_traces(texttemplate="%{text:.1f}h", textposition="outside")
                fig_bar.update_layout(showlegend=False, coloraxis_showscale=False, height=400)
                st.plotly_chart(fig_bar, use_container_width=True)

            # Heatmap: employee × day type breakdown
            if not df.empty:
                melt_cols = ["Angajat", "Zile Lucrate", "Liber", "Concediu", "Bolnav", "Absent"]
                df_melt = df[melt_cols].melt(id_vars="Angajat", var_name="Tip", value_name="Zile")
                fig_heat = px.bar(
                    df_melt,
                    x="Angajat",
                    y="Zile",
                    color="Tip",
                    barmode="stack",
                    title="Distributia zilelor per angajat",
                    color_discrete_map={
                        "Zile Lucrate": "#4CAF50",
                        "Liber": "#2196F3",
                        "Concediu": "#FF9800",
                        "Bolnav": "#F44336",
                        "Absent": "#9E9E9E",
                    },
                    height=400,
                )
                st.plotly_chart(fig_heat, use_container_width=True)

    # ── TAB 2: EMPLOYEE CALENDAR ──────────────────────────────────────────────
    with tab2:
        employees = get_all_employees()
        if not employees:
            st.info("Nu exista angajati inregistrati.")
        else:
            emp_options = {e["full_name"]: e for e in employees}
            sel_name = st.selectbox(
                "Selecteaza angajat",
                options=list(emp_options.keys()),
                key="admin_cal_emp_select",
            )
            sel_emp = emp_options[sel_name]

            adm_year = st.session_state["adm_year"]
            adm_month = st.session_state["adm_month"]

            entries = get_entries_for_user_month(sel_emp["id"], adm_year, adm_month)
            entries_map = {e["date"]: e for e in entries}
            stats = compute_monthly_stats(sel_emp["id"], adm_year, adm_month)

            st.markdown(f"#### {sel_name} — {RO_MONTHS[adm_month]} {adm_year}")
            render_summary_cards(stats)
            st.divider()
            render_calendar_grid(
                entries_map=entries_map,
                year=adm_year,
                month=adm_month,
                clickable=False,
                key_prefix=f"admin_cal_{sel_emp['id']}",
            )

    # ── TAB 3: MANAGE EMPLOYEES ───────────────────────────────────────────────
    with tab3:
        st.subheader("Angajati inregistrati")
        all_users = get_all_users()

        for u in all_users:
            if u["username"] == DEFAULT_ADMIN_USER and u["role"] == "admin":
                continue  # Don't show the default admin in this list
            col_name, col_role, col_status, col_toggle, col_del = st.columns([3, 2, 2, 2, 1])
            with col_name:
                st.write(f"**{u['full_name']}** (@{u['username']})")
            with col_role:
                role_label = "🔧 Admin" if u["role"] == "admin" else "👤 Angajat"
                st.write(role_label)
            with col_status:
                status_label = "🟢 Activ" if u["is_active"] else "🔴 Inactiv"
                st.write(status_label)
            with col_toggle:
                toggle_label = "Dezactiveaza" if u["is_active"] else "Activeaza"
                if st.button(toggle_label, key=f"toggle_{u['id']}"):
                    toggle_user_active(u["id"])
                    st.rerun()
            with col_del:
                if st.button("🗑️", key=f"del_user_{u['id']}", help="Sterge utilizator"):
                    delete_user(u["id"])
                    st.rerun()

        st.divider()

        col_add, col_pw = st.columns(2)

        with col_add:
            st.subheader("➕ Adauga utilizator")
            with st.form("add_user_form", clear_on_submit=True):
                new_username = st.text_input("Utilizator")
                new_full_name = st.text_input("Nume complet")
                new_password = st.text_input("Parola", type="password")
                new_role = st.selectbox("Rol", ["employee", "admin"], format_func=lambda r: "Angajat" if r == "employee" else "Admin")
                submitted = st.form_submit_button("Adauga", type="primary")
                if submitted:
                    if not new_username or not new_full_name or not new_password:
                        st.error("Toate campurile sunt obligatorii.")
                    elif len(new_password) < 6:
                        st.error("Parola trebuie sa aiba cel putin 6 caractere.")
                    else:
                        ok = create_user(new_username, new_password, new_full_name, new_role)
                        if ok:
                            st.success(f"Utilizatorul '{new_username}' a fost adaugat.")
                            st.rerun()
                        else:
                            st.error(f"Utilizatorul '{new_username}' exista deja.")

        with col_pw:
            st.subheader("🔑 Reseteaza parola")
            all_users_for_pw = [u for u in all_users if u["username"] != DEFAULT_ADMIN_USER or u["role"] != "admin"]
            if all_users_for_pw:
                with st.form("reset_pw_form", clear_on_submit=True):
                    pw_user_options = {f"{u['full_name']} (@{u['username']})": u["id"] for u in all_users_for_pw}
                    sel_pw_label = st.selectbox("Selecteaza utilizator", list(pw_user_options.keys()))
                    new_pw = st.text_input("Parola noua", type="password")
                    pw_submitted = st.form_submit_button("Reseteaza", type="primary")
                    if pw_submitted:
                        if len(new_pw) < 6:
                            st.error("Parola trebuie sa aiba cel putin 6 caractere.")
                        else:
                            update_user_password(pw_user_options[sel_pw_label], new_pw)
                            st.success("Parola a fost resetata.")
            else:
                st.info("Nu exista utilizatori de gestionat.")


# ==============================================================================
# SECTION 7 — LOGIN PAGE
# ==============================================================================

def render_login() -> None:
    st.markdown(
        """
        <div style='text-align:center; padding: 60px 0 20px 0;'>
            <h1>🕐 Pontaj</h1>
            <p style='color:#666; font-size:18px;'>Sistem de Inregistrare Timp</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        with st.form("login_form"):
            st.markdown("#### Autentificare")
            username = st.text_input("Utilizator", placeholder="username")
            password = st.text_input("Parola", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("🔑 Conectare", use_container_width=True, type="primary")

            if submitted:
                if not username or not password:
                    st.error("Introduceti utilizatorul si parola.")
                else:
                    user = login(username, password)
                    if user:
                        st.session_state["user"] = user
                        st.rerun()
                    else:
                        st.error("Utilizator sau parola incorecte.")


# ==============================================================================
# SECTION 8 — MAIN ENTRY POINT
# ==============================================================================

def main() -> None:
    st.set_page_config(
        page_title="Pontaj",
        layout="wide",
        page_icon="🕐",
        initial_sidebar_state="expanded",
    )

    # Global style tweaks
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { min-width: 200px !important; max-width: 260px !important; }
        .stMetric { background: #f8f9fa; border-radius: 8px; padding: 12px; }
        div[data-testid="metric-container"] { border: 1px solid #e0e0e0; border-radius: 8px; padding: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_db()
    init_session_state()

    user = st.session_state.get("user")

    if user is None:
        render_login()
    elif user["role"] == "admin":
        render_admin_view(user)
    else:
        render_employee_view(user)


if __name__ == "__main__":
    main()
