"""
data_access.py  --  Sprint 3, ticket S3-03

The only file in this app that writes SQL. Every page calls these
functions instead of writing its own queries, so:

  - the dashboard and the Sprint 2 metrics glossary can never disagree
    about what a metric means (every formula here matches
    docs/dashboard_metrics_glossary.md)
  - a filter combination with no matching rows returns an empty
    DataFrame / zeroed metrics, not an exception
  - a real database problem (missing file, locked file, bad SQL) raises
    one predictable DataAccessError that app.py turns into a clean
    message instead of a traceback

Schema (see docs/data_dictionary.md for the full description):
  customers            (customer_id, age, country_code, region, signup_date,
                         loyalty_score, email_open_rate, discount_usage_rate,
                         avg_review_score, referral_code, customer_tier, credit_limit)
  sessions             (session_id, customer_id, session_timestamp, session_duration,
                         pages_viewed, cart_additions, bounce_flag, traffic_source,
                         device_type, campaign_id, geo_ip_region)
  transactions         (transaction_id, customer_id, transaction_timestamp, order_value,
                         items_count, payment_method, discount_applied, shipping_speed,
                         high_value_flag)
  geo_data             (geo_ip_region, average_income, urban_ratio, internet_penetration, region_tier)
  marketing_campaigns  (campaign_id, campaign_type, campaign_budget, region_target, start_date, end_date)

Note on filters: "channel" (traffic_source) and "device" only exist on
sessions -- transactions has no channel column and cannot be attributed
to a session in this schema, so the channel filter narrows session-based
numbers only. Revenue and order metrics narrow by country, tier, customer
ID, and date, but not by channel. See README.md for more on this.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional
import sqlite3

import pandas as pd
import streamlit as st

DB_PATH = Path(__file__).resolve().parent.parent / "sqlite_db" / "ecommerce.db"

# Rows are capped on the browsable tables (Customers/Sessions/Transactions
# pages) because the underlying tables have 200k-2,000,000 rows -- these
# pages show the top/most-recent slice, not a full export.
TABLE_ROW_LIMIT = 500


class DataAccessError(Exception):
    """Raised for anything that stops a query from returning normally."""


@dataclass
class Filters:
    """The one shared filter shape every page and helper uses."""
    country: str = "All"
    tier: str = "All"
    channel: str = "All"
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    customer_id: Optional[int] = None


def _connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise DataAccessError(
            f"No database found at '{DB_PATH}'. Run src/build_db.py first."
        )
    try:
        conn = sqlite3.connect(DB_PATH)
        _ensure_indexes(conn)
        return conn
    except sqlite3.OperationalError as exc:
        raise DataAccessError(f"Could not open '{DB_PATH}': {exc}") from exc


_INDEXES_READY = False


def _ensure_indexes(conn: sqlite3.Connection) -> None:
    """Create read-speed indexes the first time this process touches the
    database. Safe to run repeatedly (IF NOT EXISTS); does not change any
    data, only adds lookup structures. Sessions/transactions are large
    enough (2,000,000 / 500,000 rows) that filtered queries are much
    faster with these in place."""
    global _INDEXES_READY
    if _INDEXES_READY:
        return
    conn.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_customers_country ON customers(country_code);
        CREATE INDEX IF NOT EXISTS idx_customers_tier ON customers(customer_tier);
        CREATE INDEX IF NOT EXISTS idx_customers_id ON customers(customer_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_customer ON sessions(customer_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_ts ON sessions(session_timestamp);
        CREATE INDEX IF NOT EXISTS idx_sessions_channel ON sessions(traffic_source);
        CREATE INDEX IF NOT EXISTS idx_transactions_customer ON transactions(customer_id);
        CREATE INDEX IF NOT EXISTS idx_transactions_ts ON transactions(transaction_timestamp);
        """
    )
    conn.commit()
    _INDEXES_READY = True


def _run_query(query: str, params: tuple = ()) -> pd.DataFrame:
    try:
        conn = _connect()
        try:
            return pd.read_sql_query(query, conn, params=params)
        finally:
            conn.close()
    except DataAccessError:
        raise
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        raise DataAccessError(f"That query could not be run: {exc}") from exc


def _customer_where(filters: Filters, alias: str = "c", include_tier: bool = True):
    """Country / tier / customer-id conditions shared by most queries."""
    clauses, params = [], []
    if filters.country and filters.country != "All":
        clauses.append(f"{alias}.country_code = ?")
        params.append(filters.country)
    if include_tier and filters.tier and filters.tier != "All":
        clauses.append(f"{alias}.customer_tier = ?")
        params.append(filters.tier)
    if filters.customer_id is not None:
        clauses.append(f"{alias}.customer_id = ?")
        params.append(filters.customer_id)
    return clauses, params


def _needs_customer_join(filters: Filters, include_tier: bool = True) -> bool:
    """True only if a customer-level filter is actually set. Sessions and
    transactions queries that don't need country/tier/customer_id should
    skip the join to `customers` entirely -- on a 2,000,000-row sessions
    table, joining 2M rows to `customers` just to apply zero conditions
    took 7.4s in testing; the same query with no join took 0.13s."""
    if filters.country and filters.country != "All":
        return True
    if include_tier and filters.tier and filters.tier != "All":
        return True
    if filters.customer_id is not None:
        return True
    return False


def _add_date_clause(clauses: list, params: list, date_col: str, filters: Filters):
    if filters.date_from:
        clauses.append(f"{date_col} >= ?")
        params.append(filters.date_from.isoformat())
    if filters.date_to:
        clauses.append(f"{date_col} <= ?")
        params.append(filters.date_to.isoformat())


def _where(clauses: list) -> str:
    return f"WHERE {' AND '.join(clauses)}" if clauses else ""


@st.cache_data(ttl=300, show_spinner=False)
def get_filter_options() -> dict:
    countries = _run_query("SELECT DISTINCT country_code FROM customers ORDER BY 1")
    tiers = _run_query("SELECT DISTINCT customer_tier FROM customers ORDER BY 1")
    channels = _run_query("SELECT DISTINCT traffic_source FROM sessions ORDER BY 1")
    bounds = _run_query(
        "SELECT MIN(transaction_timestamp) AS min_d, MAX(transaction_timestamp) AS max_d "
        "FROM transactions"
    )
    min_date = pd.to_datetime(bounds["min_d"].iloc[0]).date() if not bounds.empty and bounds["min_d"].iloc[0] else None
    max_date = pd.to_datetime(bounds["max_d"].iloc[0]).date() if not bounds.empty and bounds["max_d"].iloc[0] else None
    return {
        "countries": countries["country_code"].tolist() if not countries.empty else [],
        "tiers": tiers["customer_tier"].tolist() if not tiers.empty else [],
        "channels": channels["traffic_source"].dropna().tolist() if not channels.empty else [],
        "min_date": min_date,
        "max_date": max_date,
    }


@st.cache_data(ttl=300, show_spinner=False)
def get_overview_metrics(filters: Filters) -> dict:
    """Every number here follows docs/dashboard_metrics_glossary.md exactly."""

    # Customer count: distinct customers matching country/tier/customer-id (no date -- customers aren't dated events)
    cust_clauses, cust_params = _customer_where(filters, alias="c")
    customer_count = _run_query(
        f"SELECT COUNT(DISTINCT c.customer_id) AS n FROM customers c {_where(cust_clauses)}",
        tuple(cust_params),
    )["n"].iloc[0]

    # Orders / revenue / AOV / discount rate / high-value share: only join to
    # customers when a country/tier/customer filter actually needs it.
    if _needs_customer_join(filters):
        tx_clauses, tx_params = _customer_where(filters, alias="c")
        _add_date_clause(tx_clauses, tx_params, "t.transaction_timestamp", filters)
        tx_from = "FROM transactions t JOIN customers c ON c.customer_id = t.customer_id"
    else:
        tx_clauses, tx_params = [], []
        _add_date_clause(tx_clauses, tx_params, "t.transaction_timestamp", filters)
        tx_from = "FROM transactions t"
    tx = _run_query(
        f"""
        SELECT COUNT(*) AS orders,
               COALESCE(SUM(t.order_value), 0) AS revenue,
               COALESCE(AVG(CASE WHEN t.discount_applied THEN 1.0 ELSE 0.0 END), 0) AS discount_rate,
               COALESCE(AVG(CASE WHEN t.high_value_flag = 'Yes' THEN 1.0 ELSE 0.0 END), 0) AS high_value_share
        {tx_from}
        {_where(tx_clauses)}
        """,
        tuple(tx_params),
    ).iloc[0]
    orders = int(tx["orders"])
    revenue = float(tx["revenue"])
    aov = round(revenue / orders, 2) if orders else 0.0

    # Sessions / bounce rate: same rule -- only join to customers if a
    # country/tier/customer filter is set. Channel (traffic_source) never
    # needs the join since it's a sessions column already.
    if _needs_customer_join(filters):
        sess_clauses, sess_params = _customer_where(filters, alias="c")
        sess_from = "FROM sessions s JOIN customers c ON c.customer_id = s.customer_id"
    else:
        sess_clauses, sess_params = [], []
        sess_from = "FROM sessions s"
    if filters.channel and filters.channel != "All":
        sess_clauses.append("s.traffic_source = ?")
        sess_params.append(filters.channel)
    _add_date_clause(sess_clauses, sess_params, "s.session_timestamp", filters)
    sess = _run_query(
        f"""
        SELECT COUNT(*) AS sessions,
               COALESCE(AVG(CASE WHEN s.bounce_flag THEN 1.0 ELSE 0.0 END), 0) AS bounce_rate
        {sess_from}
        {_where(sess_clauses)}
        """,
        tuple(sess_params),
    ).iloc[0]

    return {
        "customer_count": int(customer_count),
        "order_count": orders,
        "total_revenue": revenue,
        "avg_order_value": aov,
        "session_count": int(sess["sessions"]),
        "bounce_rate": float(sess["bounce_rate"]),
        "high_value_share": float(tx["high_value_share"]),
        "discount_rate": float(tx["discount_rate"]),
    }


@st.cache_data(ttl=300, show_spinner=False)
def get_tier_breakdown(filters: Filters) -> pd.DataFrame:
    """Customer count per tier -- does not apply the tier filter to itself."""
    clauses, params = _customer_where(filters, alias="c", include_tier=False)
    return _run_query(
        f"""
        SELECT c.customer_tier, COUNT(*) AS customers
        FROM customers c
        {_where(clauses)}
        GROUP BY c.customer_tier
        ORDER BY customers DESC
        """,
        tuple(params),
    )


@st.cache_data(ttl=300, show_spinner=False)
def get_revenue_trend(filters: Filters) -> pd.DataFrame:
    """Monthly revenue and order count, for the Overview trend chart."""
    if _needs_customer_join(filters):
        clauses, params = _customer_where(filters, alias="c")
        from_sql = "FROM transactions t JOIN customers c ON c.customer_id = t.customer_id"
    else:
        clauses, params = [], []
        from_sql = "FROM transactions t"
    _add_date_clause(clauses, params, "t.transaction_timestamp", filters)
    return _run_query(
        f"""
        SELECT substr(t.transaction_timestamp, 1, 7) AS month,
               COUNT(*) AS orders,
               ROUND(SUM(t.order_value), 2) AS revenue
        {from_sql}
        {_where(clauses)}
        GROUP BY month
        ORDER BY month
        """,
        tuple(params),
    )


@st.cache_data(ttl=300, show_spinner=False)
def get_customers(filters: Filters) -> pd.DataFrame:
    """Customers matching the filters, with lifetime orders/spend.
    Capped at TABLE_ROW_LIMIT rows, sorted by lifetime spend, unless a
    specific customer_id is given -- that always returns its one row."""
    clauses, params = _customer_where(filters, alias="c")
    limit_clause = "" if filters.customer_id is not None else f"LIMIT {TABLE_ROW_LIMIT}"
    return _run_query(
        f"""
        SELECT c.customer_id, c.country_code, c.region, c.customer_tier,
               c.signup_date, c.loyalty_score, c.credit_limit,
               COUNT(t.transaction_id) AS lifetime_orders,
               COALESCE(SUM(t.order_value), 0) AS lifetime_spend
        FROM customers c
        LEFT JOIN transactions t ON t.customer_id = c.customer_id
        {_where(clauses)}
        GROUP BY c.customer_id
        ORDER BY lifetime_spend DESC
        {limit_clause}
        """,
        tuple(params),
    )


@st.cache_data(ttl=300, show_spinner=False)
def get_sessions(filters: Filters) -> pd.DataFrame:
    """Most recent sessions matching the filters. Capped at TABLE_ROW_LIMIT rows."""
    clauses, params = _customer_where(filters, alias="c")
    if filters.channel and filters.channel != "All":
        clauses.append("s.traffic_source = ?")
        params.append(filters.channel)
    _add_date_clause(clauses, params, "s.session_timestamp", filters)
    limit_clause = "" if filters.customer_id is not None else f"LIMIT {TABLE_ROW_LIMIT}"
    return _run_query(
        f"""
        SELECT s.session_id, s.session_timestamp, s.customer_id, c.country_code,
               c.customer_tier, s.traffic_source, s.device_type, s.session_duration,
               s.pages_viewed, s.cart_additions, s.bounce_flag
        FROM sessions s
        JOIN customers c ON c.customer_id = s.customer_id
        {_where(clauses)}
        ORDER BY s.session_timestamp DESC
        {limit_clause}
        """,
        tuple(params),
    )


@st.cache_data(ttl=300, show_spinner=False)
def get_sessions_by_channel(filters: Filters) -> pd.DataFrame:
    """Session count and bounce rate per traffic source, for the Sessions page chart."""
    if _needs_customer_join(filters):
        clauses, params = _customer_where(filters, alias="c")
        from_sql = "FROM sessions s JOIN customers c ON c.customer_id = s.customer_id"
    else:
        clauses, params = [], []
        from_sql = "FROM sessions s"
    _add_date_clause(clauses, params, "s.session_timestamp", filters)
    return _run_query(
        f"""
        SELECT s.traffic_source,
               COUNT(*) AS sessions,
               ROUND(AVG(CASE WHEN s.bounce_flag THEN 1.0 ELSE 0.0 END), 4) AS bounce_rate
        {from_sql}
        {_where(clauses)}
        GROUP BY s.traffic_source
        ORDER BY sessions DESC
        """,
        tuple(params),
    )


@st.cache_data(ttl=300, show_spinner=False)
def get_transactions(filters: Filters) -> pd.DataFrame:
    """Most recent transactions matching the filters. Capped at TABLE_ROW_LIMIT rows."""
    clauses, params = _customer_where(filters, alias="c")
    _add_date_clause(clauses, params, "t.transaction_timestamp", filters)
    limit_clause = "" if filters.customer_id is not None else f"LIMIT {TABLE_ROW_LIMIT}"
    return _run_query(
        f"""
        SELECT t.transaction_id, t.transaction_timestamp, t.customer_id, c.country_code,
               c.customer_tier, t.order_value, t.items_count, t.payment_method,
               t.discount_applied, t.shipping_speed, t.high_value_flag
        FROM transactions t
        JOIN customers c ON c.customer_id = t.customer_id
        {_where(clauses)}
        ORDER BY t.transaction_timestamp DESC
        {limit_clause}
        """,
        tuple(params),
    )


@st.cache_data(ttl=300, show_spinner=False)
def get_transactions_by_payment_method(filters: Filters) -> pd.DataFrame:
    """Order count and revenue per payment method, for the Transactions page chart."""
    if _needs_customer_join(filters):
        clauses, params = _customer_where(filters, alias="c")
        from_sql = "FROM transactions t JOIN customers c ON c.customer_id = t.customer_id"
    else:
        clauses, params = [], []
        from_sql = "FROM transactions t"
    _add_date_clause(clauses, params, "t.transaction_timestamp", filters)
    return _run_query(
        f"""
        SELECT t.payment_method,
               COUNT(*) AS orders,
               ROUND(SUM(t.order_value), 2) AS revenue
        {from_sql}
        {_where(clauses)}
        GROUP BY t.payment_method
        ORDER BY revenue DESC
        """,
        tuple(params),
    )