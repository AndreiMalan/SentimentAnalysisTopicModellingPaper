"""
admin_view.py — Streamlit dashboard for the admin role.

Three tabs:
  1. Prezentare Generala  — per-employee stats table + charts + CSV export
  2. Calendar Angajat     — read-only calendar for a selected employee
  3. Gestionare Angajati  — add users, toggle active, reset passwords, delete
"""

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from .auth import logout
from .components import render_calendar_grid, render_month_navigator, render_summary_cards
from .constants import DEFAULT_ADMIN_USER, RO_MONTHS
from .database import (
    compute_monthly_stats,
    create_user,
    delete_user,
    get_all_employees,
    get_all_entries_month,
    get_all_users,
    get_entries_for_user_month,
    toggle_user_active,
    update_user_password,
    verify_password,
)


def render_admin_view(user: dict) -> None:
    """Full admin dashboard with three tabs."""
    with st.sidebar:
        st.markdown(f"### 🔧 {user['full_name']}")
        st.caption("Administrator")
        st.divider()

        # ── Change my password ────────────────────────────────────────────────
        with st.expander("🔑 Schimba parola"):
            with st.form("adm_change_pw_form", clear_on_submit=True):
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

    st.title("📊 Panou Administrare Pontaj")

    tab1, tab2, tab3 = st.tabs([
        "📊 Prezentare Generala",
        "📅 Calendar Angajat",
        "👥 Gestionare Angajati",
    ])

    with tab1:
        _render_overview_tab()

    with tab2:
        _render_employee_calendar_tab()

    with tab3:
        _render_manage_users_tab()


# ---------------------------------------------------------------------------
# Tab 1 — Overview
# ---------------------------------------------------------------------------

def _render_overview_tab() -> None:
    year, month = render_month_navigator("adm")
    st.divider()

    employees = get_all_employees()
    if not employees:
        st.info("Nu exista angajati inregistrati. Adaugati angajati in tab-ul 'Gestionare Angajati'.")
        return

    # Build per-employee stats DataFrame
    rows = []
    for emp in employees:
        s = compute_monthly_stats(emp["id"], year, month)
        rows.append({
            "Angajat":      emp["full_name"],
            "Zile Lucrate": s["days_worked"],
            "Ore Totale":   s["total_hours"],
            "Liber":        s["liber"],
            "Concediu":     s["concediu"],
            "Bolnav":       s["bolnav"],
            "Absent":       s["absent"],
            "Nelucrate":    s["missing"],
        })
    df = pd.DataFrame(rows)

    st.subheader(f"Raport {RO_MONTHS[month]} {year}")
    st.dataframe(df.style.format({"Ore Totale": "{:.1f}"}), use_container_width=True, hide_index=True)

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

    # Bar chart — total hours per employee
    if df["Ore Totale"].sum() > 0:
        fig_bar = px.bar(
            df, x="Angajat", y="Ore Totale",
            color="Ore Totale", color_continuous_scale="Greens",
            title=f"Ore lucrate per angajat — {RO_MONTHS[month]} {year}",
            text="Ore Totale",
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}h", textposition="outside")
        fig_bar.update_layout(showlegend=False, coloraxis_showscale=False, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Stacked bar — day-type distribution
    melt_cols = ["Angajat", "Zile Lucrate", "Liber", "Concediu", "Bolnav", "Absent"]
    df_melt = df[melt_cols].melt(id_vars="Angajat", var_name="Tip", value_name="Zile")
    fig_dist = px.bar(
        df_melt, x="Angajat", y="Zile", color="Tip", barmode="stack",
        title="Distributia zilelor per angajat",
        color_discrete_map={
            "Zile Lucrate": "#4CAF50",
            "Liber":        "#2196F3",
            "Concediu":     "#FF9800",
            "Bolnav":       "#F44336",
            "Absent":       "#9E9E9E",
        },
        height=400,
    )
    st.plotly_chart(fig_dist, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2 — Employee calendar (read-only)
# ---------------------------------------------------------------------------

def _render_employee_calendar_tab() -> None:
    employees = get_all_employees()
    if not employees:
        st.info("Nu exista angajati inregistrati.")
        return

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


# ---------------------------------------------------------------------------
# Tab 3 — Manage users
# ---------------------------------------------------------------------------

def _render_manage_users_tab() -> None:
    st.subheader("Utilizatori inregistrati")
    all_users = get_all_users()

    if not all_users:
        st.info("Nu exista utilizatori.")
    else:
        for u in all_users:
            col_name, col_role, col_status, col_toggle, col_del = st.columns([3, 2, 2, 2, 1])
            with col_name:
                st.write(f"**{u['full_name']}** (@{u['username']})")
            with col_role:
                st.write("🔧 Admin" if u["role"] == "admin" else "👤 Angajat")
            with col_status:
                st.write("🟢 Activ" if u["is_active"] else "🔴 Inactiv")
            with col_toggle:
                # Prevent deactivating the default admin
                if u["username"] == DEFAULT_ADMIN_USER and u["role"] == "admin":
                    st.caption("—")
                else:
                    label = "Dezactiveaza" if u["is_active"] else "Activeaza"
                    if st.button(label, key=f"toggle_{u['id']}"):
                        toggle_user_active(u["id"])
                        st.rerun()
            with col_del:
                # Prevent deleting the default admin
                if u["username"] == DEFAULT_ADMIN_USER and u["role"] == "admin":
                    st.caption("—")
                else:
                    if st.button("🗑️", key=f"del_user_{u['id']}", help="Sterge utilizator"):
                        delete_user(u["id"])
                        st.rerun()

    st.divider()
    col_add, col_pw = st.columns(2)

    with col_add:
        st.subheader("➕ Adauga utilizator")
        with st.form("add_user_form", clear_on_submit=True):
            new_username  = st.text_input("Utilizator")
            new_full_name = st.text_input("Nume complet")
            new_password  = st.text_input("Parola", type="password")
            new_role = st.selectbox(
                "Rol", ["employee", "admin"],
                format_func=lambda r: "Angajat" if r == "employee" else "Admin",
            )
            if st.form_submit_button("Adauga", type="primary"):
                if not new_username or not new_full_name or not new_password:
                    st.error("Toate campurile sunt obligatorii.")
                elif len(new_password) < 6:
                    st.error("Parola trebuie sa aiba cel putin 6 caractere.")
                elif create_user(new_username, new_password, new_full_name, new_role):
                    st.success(f"Utilizatorul '{new_username}' a fost adaugat.")
                    st.rerun()
                else:
                    st.error(f"Utilizatorul '{new_username}' exista deja.")

    with col_pw:
        st.subheader("🔑 Reseteaza parola (admin)")
        st.caption("Folositi aceasta sectiune doar pentru a reseta parola altui utilizator. "
                   "Pentru propria parola folositi bara laterala.")
        if all_users:
            with st.form("reset_pw_form", clear_on_submit=True):
                pw_options = {f"{u['full_name']} (@{u['username']})": u["id"] for u in all_users}
                sel_label = st.selectbox("Selecteaza utilizator", list(pw_options.keys()))
                new_pw = st.text_input("Parola noua", type="password")
                if st.form_submit_button("Reseteaza", type="primary"):
                    if len(new_pw) < 6:
                        st.error("Parola trebuie sa aiba cel putin 6 caractere.")
                    else:
                        update_user_password(pw_options[sel_label], new_pw)
                        st.success("Parola a fost resetata.")
        else:
            st.info("Nu exista utilizatori de gestionat.")
