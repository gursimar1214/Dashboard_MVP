"""
Sprint 2 - S2-01: Create the SQLite database.

Loads every cleaned CSV from clean_data/ into its own table in
sqlite_db/ecommerce.db. One table per file. Nothing is joined here -
that happens later, at query time, using SQL views (Sprint 2 continued).

Run from the project root:
    python src/build_db.py
"""

import sqlite3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "clean_data"
DB_PATH = ROOT / "sqlite_db" / "ecommerce.db"

# table_name -> source csv file
TABLES = {
    "customers": "clean_customers.csv",
    "sessions": "clean_sessions.csv",
    "transactions": "clean_transactions.csv",
    "geo_data": "clean_geo_data.csv",
    "marketing_campaigns": "clean_marketing_campaigns.csv",
}

DB_PATH.parent.mkdir(exist_ok=True)
conn = sqlite3.connect(DB_PATH)

print(f"Building {DB_PATH} ...\n")

for table_name, csv_file in TABLES.items():
    df = pd.read_csv(CLEAN_DIR / csv_file)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"  {table_name:<22} <- {csv_file:<28} {len(df):>10,} rows, {len(df.columns)} columns")

conn.close()
print("\nDone. Database saved to sqlite_db/ecommerce.db")
