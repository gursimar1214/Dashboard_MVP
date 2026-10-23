"""
build_db.py

Rebuilds sqlite_db/ecommerce.db from the cleaned CSVs in clean_data/.
Run this any time the .db file is missing, corrupted, or (as happened
here) stuck as a OneDrive cloud-placeholder stub instead of the real
file.

Usage (from anywhere -- paths are resolved relative to this file, same
convention as dashboard/app.py and dashboard/data_access.py):

    python src/build_db.py

Expects, one level up from this file:
    clean_data/clean_customers.csv
    clean_data/clean_sessions.csv
    clean_data/clean_transactions.csv
    clean_data/clean_geo_data.csv
    clean_data/clean_marketing_campaigns.csv

Writes:
    sqlite_db/ecommerce.db   (overwritten if it already exists)
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DATA_DIR = PROJECT_ROOT / "clean_data"
DB_PATH = PROJECT_ROOT / "sqlite_db" / "ecommerce.db"

# table_name -> (csv filename, columns that are boolean flags, columns
# that hold a "$1,234" style money string needing to be cleaned to a
# plain float)
TABLES = {
    "customers": ("clean_customers.csv", [], []),
    "sessions": ("clean_sessions.csv", ["bounce_flag"], []),
    "transactions": ("clean_transactions.csv", ["discount_applied"], []),
    "geo_data": ("clean_geo_data.csv", [], []),
    "marketing_campaigns": ("clean_marketing_campaigns.csv", [], ["campaign_budget"]),
}


def _clean_money(series: pd.Series) -> pd.Series:
    """'$46,187' -> 46187.0"""
    return (
        series.astype(str)
        .str.replace(r"[$,]", "", regex=True)
        .astype(float)
    )


def build() -> None:
    if not CLEAN_DATA_DIR.exists():
        sys.exit(f"Can't find {CLEAN_DATA_DIR} -- expected the clean_data/ folder next to sqlite_db/ and src/.")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        print(f"Removing existing {DB_PATH} (it will be rebuilt from scratch)...")
        DB_PATH.unlink()

    conn = sqlite3.connect(DB_PATH)
    try:
        for table, (csv_name, bool_cols, money_cols) in TABLES.items():
            csv_path = CLEAN_DATA_DIR / csv_name
            if not csv_path.exists():
                sys.exit(f"Missing {csv_path}")

            print(f"Loading {csv_name} -> {table} ...")
            df = pd.read_csv(csv_path)

            for col in bool_cols:
                # sqlite has no native boolean; store as 0/1 integers so
                # the existing SQL in data_access.py (e.g.
                # "WHEN t.discount_applied THEN 1.0 ELSE 0.0") works
                # unchanged.
                df[col] = df[col].astype(bool).astype(int)

            for col in money_cols:
                df[col] = _clean_money(df[col])

            df.to_sql(table, conn, if_exists="replace", index=False)
            print(f"  {len(df):,} rows written.")

        conn.commit()

        # Same indexes data_access.py creates on first run -- doing it
        # here too means the very first dashboard query after a rebuild
        # is already fast, not just every query after that.
        print("Creating indexes...")
        index_statements = [
            "CREATE INDEX IF NOT EXISTS idx_sessions_customer_id ON sessions(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_timestamp ON sessions(session_timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_transactions_customer_id ON transactions(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(transaction_timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_customers_country_code ON customers(country_code)",
            "CREATE INDEX IF NOT EXISTS idx_customers_customer_tier ON customers(customer_tier)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_traffic_source ON sessions(traffic_source)",
        ]
        for stmt in index_statements:
            conn.execute(stmt)
        conn.commit()

    finally:
        conn.close()

    print(f"\nDone. {DB_PATH} is {DB_PATH.stat().st_size / (1024 * 1024):.1f} MB.")


if __name__ == "__main__":
    build()