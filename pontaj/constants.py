"""
constants.py — App-wide constants: entry types, colours, Romanian labels, DB path.
"""

import os

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "time_registration.db",
)

DEFAULT_ADMIN_USER = "admin"
DEFAULT_ADMIN_PASS = "Admin1234!"

# ---------------------------------------------------------------------------
# Entry-type metadata
# ---------------------------------------------------------------------------

ENTRY_TYPES: dict[str, str] = {
    "lucrat":   "Lucrat",
    "liber":    "Liber",
    "concediu": "Concediu",
    "bolnav":   "Bolnav",
    "absent":   "Absent",
}

ENTRY_COLORS: dict[str, str] = {
    "lucrat":   "#4CAF50",
    "liber":    "#2196F3",
    "concediu": "#FF9800",
    "bolnav":   "#F44336",
    "absent":   "#9E9E9E",
}

ENTRY_EMOJIS: dict[str, str] = {
    "lucrat":   "✅",
    "liber":    "🏖️",
    "concediu": "🌴",
    "bolnav":   "🤒",
    "absent":   "❌",
}

# ---------------------------------------------------------------------------
# Romanian locale strings
# ---------------------------------------------------------------------------

RO_DAYS: list[str] = ["Lu", "Ma", "Mi", "Jo", "Vi", "Sâ", "Du"]

RO_DAYS_FULL: list[str] = [
    "Luni", "Marți", "Miercuri", "Joi", "Vineri", "Sâmbătă", "Duminică",
]

RO_MONTHS: list[str] = [
    "",
    "Ianuarie", "Februarie", "Martie", "Aprilie", "Mai", "Iunie",
    "Iulie", "August", "Septembrie", "Octombrie", "Noiembrie", "Decembrie",
]
