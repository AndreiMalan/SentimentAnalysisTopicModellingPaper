"""
Pontaj - Time Registration App for Romanian Companies
======================================================
Entry point — only page config, CSS, and routing live here.
Business logic is split across the pontaj/ package:

  pontaj/constants.py      — entry types, colours, Romanian labels
  pontaj/database.py       — SQLite schema, user & time-entry CRUD, stats
  pontaj/auth.py           — login/logout, session state, login page UI
  pontaj/components.py     — month navigator, calendar grid, entry form, summary cards
  pontaj/employee_view.py  — employee self-service dashboard
  pontaj/admin_view.py     — admin overview, charts, user management

Default admin login:  admin / Admin1234!
Run with:  streamlit run time_registration_app.py
"""

import streamlit as st

from pontaj.admin_view import render_admin_view
from pontaj.auth import init_session_state, render_login
from pontaj.database import init_db
from pontaj.employee_view import render_employee_view


def main() -> None:
    st.set_page_config(
        page_title="Pontaj",
        layout="wide",
        page_icon="🕐",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { min-width: 200px !important; max-width: 260px !important; }
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
