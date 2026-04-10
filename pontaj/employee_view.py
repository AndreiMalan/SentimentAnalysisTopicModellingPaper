"""
employee_view.py — Streamlit dashboard for the employee role.
"""

import streamlit as st

from .auth import logout
from .components import render_calendar_grid, render_entry_form, render_month_navigator, render_summary_cards
from .database import compute_monthly_stats, get_entries_for_user_month, update_user_password, verify_password


def render_employee_view(user: dict) -> None:
    """Full employee dashboard: month navigator, summary cards, calendar, entry form."""
    with st.sidebar:
        st.markdown(f"### 👤 {user['full_name']}")
        st.caption(f"@{user['username']}")
        st.divider()

        # ── Change my password ────────────────────────────────────────────────
        with st.expander("🔑 Schimba parola"):
            with st.form("emp_change_pw_form", clear_on_submit=True):
                current_pw = st.text_input("Parola curenta", type="password")
                new_pw     = st.text_input("Parola noua", type="password")
                confirm_pw = st.text_input("Confirma parola noua", type="password")
                if st.form_submit_button("Salveaza parola", use_container_width=True):
                    if not current_pw or not new_pw or not confirm_pw:
                        st.error("Completeaza toate campurile.")
                    elif not verify_password(current_pw, user["password_hash"]):
                        st.error("Parola curenta este incorecta.")
                    elif len(new_pw) < 6:
                        st.error("Parola noua trebuie sa aiba cel putin 6 caractere.")
                    elif new_pw != confirm_pw:
                        st.error("Parolele noi nu coincid.")
                    else:
                        update_user_password(user["id"], new_pw)
                        st.success("Parola a fost schimbata!")

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
