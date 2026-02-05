"""Database configuration and helpers for Freight Budget Management."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DEFAULT_DB_FILENAME = "freight_budget_management.sqlite3"
DEFAULT_PDF_DIRNAME = "freight_budget_pdfs"


def get_data_root() -> Path:
    """Return the base directory for local data storage."""
    configured_root = os.getenv("FREIGHT_BUDGET_DATA_ROOT")
    if configured_root:
        return Path(configured_root).expanduser()
    return Path.cwd() / "data"


def get_db_path() -> Path:
    """Return the SQLite database file path."""
    return get_data_root() / DEFAULT_DB_FILENAME


def get_pdf_output_dir() -> Path:
    """Return the directory where PDFs are stored."""
    return get_data_root() / DEFAULT_PDF_DIRNAME


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with required settings."""
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db() -> None:
    """Create database schema if it does not exist."""
    get_pdf_output_dir().mkdir(parents=True, exist_ok=True)
    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS quotations (
                quotation_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                valid_from TEXT NOT NULL,
                valid_to TEXT NOT NULL,
                currency TEXT NOT NULL,
                total_amount REAL NOT NULL,
                budget_allocation_id TEXT,
                customer_name TEXT,
                PRIMARY KEY (quotation_id, version)
            );

            CREATE TABLE IF NOT EXISTS quotation_line_items (
                line_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                quotation_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                description TEXT NOT NULL,
                quantity REAL NOT NULL,
                unit_price REAL NOT NULL,
                amount REAL NOT NULL,
                FOREIGN KEY (quotation_id, version)
                    REFERENCES quotations (quotation_id, version)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS budget_allocations (
                budget_allocation_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                amount_available REAL NOT NULL,
                currency TEXT NOT NULL,
                valid_from TEXT NOT NULL,
                valid_to TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_events (
                audit_event_id TEXT PRIMARY KEY,
                quotation_id TEXT,
                version INTEGER,
                command_name TEXT NOT NULL,
                actor TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                outcome TEXT NOT NULL,
                reason TEXT,
                payload_json TEXT
            );

            CREATE TABLE IF NOT EXISTS edit_locks (
                quotation_id TEXT PRIMARY KEY,
                locked_by TEXT NOT NULL,
                locked_at TEXT NOT NULL,
                lock_status TEXT NOT NULL
            );
            """
        )

        _ensure_column(connection, "quotations", "customer_name", "TEXT")


def _ensure_column(connection: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
    existing = connection.execute(f"PRAGMA table_info({table})").fetchall()
    if any(row["name"] == column for row in existing):
        return
    connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
