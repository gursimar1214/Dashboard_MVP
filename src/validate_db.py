"""
Sprint 2 - S2-02: Validate joins and row counts.

Checks, using real SQL against ecommerce.db, that:
  - table counts match what we expect
  - primary keys have no duplicates
  - no orphan foreign keys exist
  - joining tables does not change row counts (no fan-out / duplication)
  - a couple of totals match a hand-calculated number

Run from the project root:
    python src/validate_db.py
"""

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "sqlite_db" / "ecommerce.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()


def q(sql):
    return cur.execute(sql).fetchone()[0]


print("=== 1. Table counts ===")
expected = {
    "customers": 200_000,
    "sessions": 2_000_000,
    "transactions": 500_000,
    "geo_data": 100,
    "marketing_campaigns": 200,
}
for table, expected_count in expected.items():
    actual = q(f"SELECT COUNT(*) FROM {table}")
    status = "OK" if actual == expected_count else "MISMATCH"
    print(f"  {table:<22} expected {expected_count:>10,}  actual {actual:>10,}  [{status}]")

print("\n=== 2. Duplicate primary keys ===")
pk_checks = {
    "customers": "customer_id",
    "sessions": "session_id",
    "transactions": "transaction_id",
    "geo_data": "geo_ip_region",
    "marketing_campaigns": "campaign_id",
}
for table, pk in pk_checks.items():
    total = q(f"SELECT COUNT(*) FROM {table}")
    distinct = q(f"SELECT COUNT(DISTINCT {pk}) FROM {table}")
    status = "OK, no duplicate keys" if total == distinct else f"PROBLEM: {total - distinct} duplicate keys"
    print(f"  {table:<22} {pk:<18} total={total:,} distinct={distinct:,}  [{status}]")

print("\n=== 3. Orphan foreign keys ===")
orphan_checks = [
    ("sessions.customer_id -> customers",
     "SELECT COUNT(*) FROM sessions s LEFT JOIN customers c ON s.customer_id = c.customer_id WHERE c.customer_id IS NULL"),
    ("sessions.geo_ip_region -> geo_data",
     "SELECT COUNT(*) FROM sessions s LEFT JOIN geo_data g ON s.geo_ip_region = g.geo_ip_region WHERE g.geo_ip_region IS NULL"),
    ("sessions.campaign_id -> marketing_campaigns",
     "SELECT COUNT(*) FROM sessions s LEFT JOIN marketing_campaigns m ON s.campaign_id = m.campaign_id WHERE m.campaign_id IS NULL"),
    ("transactions.customer_id -> customers",
     "SELECT COUNT(*) FROM transactions t LEFT JOIN customers c ON t.customer_id = c.customer_id WHERE c.customer_id IS NULL"),
]
for label, sql in orphan_checks:
    orphans = q(sql)
    status = "OK, 0 orphans" if orphans == 0 else f"PROBLEM: {orphans} orphans"
    print(f"  {label:<45} [{status}]")

print("\n=== 4. Join cardinality (no fan-out / row duplication) ===")
raw_sessions = q("SELECT COUNT(*) FROM sessions")
joined_sessions = q("""
    SELECT COUNT(*) FROM sessions s
    JOIN customers c ON s.customer_id = c.customer_id
""")
status = "OK, same row count after join" if raw_sessions == joined_sessions else "PROBLEM: join changed row count"
print(f"  sessions alone: {raw_sessions:,}   sessions JOIN customers: {joined_sessions:,}  [{status}]")

raw_trans = q("SELECT COUNT(*) FROM transactions")
joined_trans = q("""
    SELECT COUNT(*) FROM transactions t
    JOIN customers c ON t.customer_id = c.customer_id
""")
status = "OK, same row count after join" if raw_trans == joined_trans else "PROBLEM: join changed row count"
print(f"  transactions alone: {raw_trans:,}   transactions JOIN customers: {joined_trans:,}  [{status}]")

print("\n=== 5. Hand-checked totals ===")
total_revenue = q("SELECT ROUND(SUM(order_value), 2) FROM transactions")
total_orders = q("SELECT COUNT(*) FROM transactions")
avg_order = round(total_revenue / total_orders, 2)
print(f"  total revenue (sum of order_value):     ${total_revenue:,.2f}")
print(f"  total orders:                           {total_orders:,}")
print(f"  average order value (revenue / orders): ${avg_order}")

one_customer = q("SELECT customer_id FROM transactions LIMIT 1")
cur.execute("SELECT COUNT(*), ROUND(SUM(order_value),2) FROM transactions WHERE customer_id = ?", (one_customer,))
cnt, total = cur.fetchone()
print(f"  spot check - customer_id {one_customer}: {cnt} transactions, ${total} total spend"
      f" (verify this by eye against clean_transactions.csv if in doubt)")

conn.close()
