"""
database/db.py
---------------
SQLite storage — transactions and anomalies.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.getenv("DB_PATH", "smartbudget.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS transactions (
            id               TEXT PRIMARY KEY,
            date             TEXT NOT NULL,
            description      TEXT NOT NULL,
            amount           REAL NOT NULL,
            category         TEXT,
            categorized_by   TEXT,
            created_at       TEXT
        );

        CREATE TABLE IF NOT EXISTS anomalies (
            id               TEXT PRIMARY KEY,
            transaction_id   TEXT,
            date             TEXT,
            description      TEXT,
            amount           REAL,
            category         TEXT,
            reason           TEXT,
            ratio            REAL,
            explanation      TEXT,
            created_at       TEXT
        );

        CREATE TABLE IF NOT EXISTS labeled_transactions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            date             TEXT NOT NULL,
            raw_description  TEXT NOT NULL,
            amount           REAL NOT NULL,
            channel          TEXT,
            category         TEXT NOT NULL,
            created_at       TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_txn_date     ON transactions(date);
        CREATE INDEX IF NOT EXISTS idx_txn_category ON transactions(category);
        CREATE INDEX IF NOT EXISTS idx_labeled_category ON labeled_transactions(category);
    """)
    conn.commit()
    conn.close()


def save_transactions(transactions: list[dict]):
    conn = get_connection()
    now  = datetime.now().isoformat()
    try:
        conn.execute("DELETE FROM transactions")
        conn.executemany(
            """INSERT OR REPLACE INTO transactions
               (id, date, description, amount, category, categorized_by, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [(t["id"], t["date"], t["description"], t["amount"],
              t.get("category"), t.get("categorized_by"), now)
             for t in transactions],
        )
        conn.commit()
    finally:
        conn.close()


def save_anomalies(anomalies: list[dict]):
    conn = get_connection()
    now  = datetime.now().isoformat()
    try:
        conn.execute("DELETE FROM anomalies")
        conn.executemany(
            """INSERT OR REPLACE INTO anomalies
               (id, transaction_id, date, description, amount, category,
                reason, ratio, explanation, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [(a["id"], a.get("transaction_id"), a["date"], a["description"],
              a["amount"], a.get("category"), a.get("reason"),
              a.get("ratio"), a.get("explanation"), now)
             for a in anomalies],
        )
        conn.commit()
    finally:
        conn.close()


def save_labeled_transaction(transaction: dict, category: str, channel: str):
    conn = get_connection()
    now = datetime.now().isoformat()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS labeled_transactions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                date             TEXT NOT NULL,
                raw_description  TEXT NOT NULL,
                amount           REAL NOT NULL,
                channel          TEXT,
                category         TEXT NOT NULL,
                created_at       TEXT
            )
        """)
        conn.execute(
            """INSERT INTO labeled_transactions
               (date, raw_description, amount, channel, category, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                transaction["date"],
                transaction["description"],
                abs(float(transaction["amount"])),
                channel,
                category,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_all_transactions() -> list[dict]:
    conn = get_connection()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM transactions ORDER BY date DESC"
        ).fetchall()]
    finally:
        conn.close()


def get_transactions_by_month(month: str) -> list[dict]:
    conn = get_connection()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM transactions WHERE date LIKE ? ORDER BY date DESC",
            (f"{month}%",),
        ).fetchall()]
    finally:
        conn.close()


def get_anomalies() -> list[dict]:
    conn = get_connection()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM anomalies ORDER BY amount DESC"
        ).fetchall()]
    finally:
        conn.close()
