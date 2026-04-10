"""
employee_view.py — Streamlit dashboard for the employee role.
"""

import streamlit as st

from .auth import logout
from .components import render_calendar_grid, render_entry_form, render_month_navigator, render_summary_cards
from .database import compute_monthly_stats, get_entries_for_user_month


def render_employee_view(user: dict) -> None:
    """Full employee dashboard: month navigator, summary cards, calendar, entry form."""
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
        with st.container():
            render_entry_form(user["id"], selected, entries_map.get(selected))
