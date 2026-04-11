"""
components.py — Reusable Streamlit UI components.

  render_month_navigator(prefix)  → (year, month)
  render_calendar_grid(...)
  render_entry_form(user_id, date_str, existing)
  render_summary_cards(stats)
"""

import calendar
from datetime import date, datetime, time

import streamlit as st

from .constants import (
    ENTRY_COLORS,
    ENTRY_EMOJIS,
    ENTRY_TYPES,
    RO_DAYS,
    RO_DAYS_FULL,
    RO_MONTHS,
)
from .database import delete_entry, upsert_time_entry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_hours(decimal_hours: float) -> str:
    """Convert decimal hours to a readable string.
    Examples: 8.0 → '8h', 6.75 → '6h 45min', 14.8 → '14h 48min'
    """
    total_minutes = round(decimal_hours * 60)
    h = total_minutes // 60
    m = total_minutes % 60
    if m == 0:
        return f"{h}h"
    return f"{h}h {m}min"


# ---------------------------------------------------------------------------
# Month navigator
# ---------------------------------------------------------------------------

def render_month_navigator(prefix: str) -> tuple[int, int]:
    """
    Render prev/title/next controls.  Reads and writes
    session_state[f'{prefix}_year'] and session_state[f'{prefix}_month'].
    Returns (year, month).
    """
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


# ---------------------------------------------------------------------------
# Calendar grid
# ---------------------------------------------------------------------------

def _inject_calendar_css() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stHorizontalBlock"] > div { padding: 1px !important; }
        div.cal-day button {
            width: 100% !important; height: 64px !important;
            font-size: 12px !important; padding: 2px !important;
            border-radius: 6px !important; white-space: pre-wrap !important;
        }
        div.cal-header { text-align:center; font-weight:bold; padding:4px 0; font-size:13px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_calendar_grid(
    entries_map: dict,          # date_str → entry dict
    year: int,
    month: int,
    clickable: bool = True,
    selected_date: str | None = None,
    key_prefix: str = "cal",
) -> None:
    """
    Render a monthly calendar grid.

    When clickable=True (employee view) each weekday cell is a button;
    clicking stores the date in session_state['selected_date'] and sets
    show_entry_form=True.  When clickable=False (admin read-only view)
    cells are plain styled divs.
    """
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

    # Build flat list of cells (None = empty offset)
    cells: list[int | None] = [None] * first_weekday
    cells += list(range(1, last_day + 1))
    while len(cells) % 7:
        cells.append(None)

    today_str = date.today().isoformat()

    for week_start in range(0, len(cells), 7):
        cols = st.columns(7)
        for col_idx, d in enumerate(cells[week_start: week_start + 7]):
            with cols[col_idx]:
                if d is None:
                    st.markdown("<div style='height:64px'></div>", unsafe_allow_html=True)
                    continue

                date_str = f"{year:04d}-{month:02d}-{d:02d}"
                entry = entries_map.get(date_str)
                etype = entry["entry_type"] if entry else None
                is_weekend = col_idx >= 5
                is_today = date_str == today_str
                is_selected = date_str == selected_date

                bg = (
                    "#f5dde0" if is_weekend and not etype
                    else ENTRY_COLORS.get(etype, "#f9f9f9")
                )
                border = (
                    "3px solid #1a73e8" if is_selected
                    else "2px solid #aaa" if is_today
                    else "1px solid #ddd"
                )
                text_color = "#fff" if etype else "#333"

                if etype == "lucrat" and entry:
                    label = (
                        f"{d}\n"
                        f"{entry.get('start_time','?')}-{entry.get('end_time','?')}\n"
                        f"{_fmt_hours(entry.get('hours_worked', 0))}"
                    )
                elif etype:
                    label = f"{d}\n{ENTRY_EMOJIS.get(etype,'')}\n{ENTRY_TYPES[etype]}"
                else:
                    label = str(d)

                if clickable:
                    st.markdown("<div class='cal-day'>", unsafe_allow_html=True)
                    clicked = st.button(
                        label,
                        key=f"{key_prefix}_{date_str}",
                        help=f"Click to register {date_str}",
                    )
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown(
                        f"""<style>
                        div[data-testid="stButton"] > button[kind="secondary"]
                            [title="Click to register {date_str}"] {{
                            background-color:{bg} !important;
                            border:{border} !important;
                            color:{text_color} !important;
                        }}
                        </style>""",
                        unsafe_allow_html=True,
                    )
                    if clicked:
                        st.session_state["selected_date"] = date_str
                        st.session_state["show_entry_form"] = True
                        st.rerun()
                else:
                    st.markdown(
                        f"""<div style='background:{bg};border:{border};border-radius:6px;
                            min-height:64px;padding:4px;text-align:center;font-size:12px;
                            color:{text_color};display:flex;flex-direction:column;
                            align-items:center;justify-content:center;white-space:pre-wrap;
                        '>{label}</div>""",
                        unsafe_allow_html=True,
                    )


# ---------------------------------------------------------------------------
# Entry form
# ---------------------------------------------------------------------------

def render_entry_form(user_id: int, date_str: str, existing: dict | None) -> None:
    """Inline form to add, edit, or delete a single day's entry."""
    d = datetime.strptime(date_str, "%Y-%m-%d").date()
    st.markdown(
        f"### {RO_DAYS_FULL[d.weekday()]}, {d.day} {RO_MONTHS[d.month]} {d.year}"
    )

    type_keys = list(ENTRY_TYPES.keys())
    current_type = (existing["entry_type"] if existing else "lucrat")
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
            upsert_time_entry(
                user_id, date_str, entry_type,
                start_t.strftime("%H:%M") if start_t else None,
                end_t.strftime("%H:%M") if end_t else None,
                notes or None,
            )
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


# ---------------------------------------------------------------------------
# Summary metric cards
# ---------------------------------------------------------------------------

def render_summary_cards(stats: dict) -> None:
    """Render six st.metric cards from a compute_monthly_stats() result."""
    cols = st.columns(6)
    cards = [
        ("🕐 Ore lucrate",  _fmt_hours(stats['total_hours'])),
        ("✅ Zile lucrate", stats["days_worked"]),
        ("🏖️ Liber",        stats["liber"]),
        ("🌴 Concediu",     stats["concediu"]),
        ("🤒 Bolnav",       stats["bolnav"]),
        ("⚠️ Nelucrate",    stats["missing"]),
    ]
    for col, (label, val) in zip(cols, cards):
        with col:
            st.metric(label, val)
