"""
auth.py — Authentication logic and session-state management.

Exports:
  login(username, password) -> user dict | None
  logout()
  init_session_state()
  render_login()            ← Streamlit login page UI
"""

from datetime import date

import streamlit as st

from .database import get_user_by_username, verify_password


# ---------------------------------------------------------------------------
# Core auth functions
# ---------------------------------------------------------------------------

def login(username: str, password: str) -> dict | None:
    """Return the user dict on success, None on failure or inactive account."""
    user = get_user_by_username(username)
    if user and user["is_active"] and verify_password(password, user["password_hash"]):
        return user
    return None


def logout() -> None:
    """Clear auth-related session state and trigger a rerun."""
    for key in ["user", "selected_date", "admin_selected_employee"]:
        st.session_state[key] = None
    st.session_state["show_entry_form"] = False
    st.rerun()


def init_session_state() -> None:
    """Set default values for every session-state key the app uses (runs once)."""
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


# ---------------------------------------------------------------------------
# Login page UI
# ---------------------------------------------------------------------------

def render_login() -> None:
    """Render the centred login form. Sets session_state['user'] on success."""
    st.markdown(
        """
        <div style='text-align:center; padding: 60px 0 20px 0;'>
            <h1>🕐 Pontaj</h1>
            <p style='color:#666; font-size:18px;'>Sistem de Inregistrare Timp</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, col_center, _ = st.columns([1, 2, 1])
    with col_center:
        with st.form("login_form"):
            st.markdown("#### Autentificare")
            username = st.text_input("Utilizator", placeholder="username")
            password = st.text_input("Parola", type="password", placeholder="••••••••")
            submitted = st.form_submit_button(
                "🔑 Conectare", use_container_width=True, type="primary"
            )

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
