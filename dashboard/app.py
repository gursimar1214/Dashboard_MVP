"""
app.py  --  Sprint 3, tickets S3-01 and S3-04

Run with (from the dashboard/ folder, or anywhere -- the path to
sqlite_db/ecommerce.db is resolved relative to this file):

    streamlit run app.py

One command to start, sidebar navigation, a title on every page, a
loading state while data is fetched, and a friendly message instead of
a traceback when a query fails or a filter combination has no data.
"""

import pandas as pd
import streamlit as st

from data_access import (
    DataAccessError,
    TABLE_ROW_LIMIT,
    get_overview_metrics,
    get_tier_breakdown,
    get_revenue_trend,
    get_customers,
    get_sessions,
    get_sessions_by_channel,
    get_transactions,
    get_transactions_by_payment_method,
)
from filters import render_filters

st.set_page_config(page_title="Global Ecommerce Dashboard", page_icon="🌍", layout="wide")

PAGES = ["Overview", "Customers", "Sessions", "Transactions"]


def main() -> None:
    st.sidebar.title("Global Ecommerce Dashboard")
    page = st.sidebar.radio("Go to", PAGES, key="nav_page")
    filters = render_filters()

    st.title(page)

    try:
        if page == "Overview":
            show_overview(filters)
        elif page == "Customers":
            show_customers(filters)
        elif page == "Sessions":
            show_sessions(filters)
        elif page == "Transactions":
            show_transactions(filters)
    except DataAccessError as exc:
        st.error(f"Couldn't load this page's data. {exc}")
    except Exception:
        st.error(
            "Something went wrong loading this page. Try a different filter "
            "combination, or check that sqlite_db/ecommerce.db exists (run "
            "src/build_db.py if it doesn't)."
        )


def show_overview(filters) -> None:
    with st.spinner("Loading overview..."):
        m = get_overview_metrics(filters)
        tiers = get_tier_breakdown(filters)
        trend = get_revenue_trend(filters)

    if m["customer_count"] == 0:
        st.info("No data matches these filters. Try widening the date range or clearing a filter.")
        return

    row1 = st.columns(4)
    row1[0].metric("Customers", f"{m['customer_count']:,}")
    row1[1].metric("Orders", f"{m['order_count']:,}")
    row1[2].metric("Revenue", f"${m['total_revenue']:,.2f}")
    row1[3].metric("Avg order value", f"${m['avg_order_value']:,.2f}")

    row2 = st.columns(4)
    row2[0].metric("Sessions", f"{m['session_count']:,}")
    row2[1].metric("Bounce rate", f"{m['bounce_rate']:.1%}")
    row2[2].metric("High-value order share", f"{m['high_value_share']:.1%}")
    row2[3].metric("Discount usage rate", f"{m['discount_rate']:.1%}")

    st.caption(
        "Sessions and bounce rate also apply the Channel filter. Revenue and order "
        "metrics do not -- transactions aren't linked to a session/channel in this schema."
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.subheader("Revenue by month")
        if trend.empty:
            st.info("No transactions in this range.")
        else:
            st.line_chart(trend.set_index("month")["revenue"])
    with col_b:
        st.subheader("Customers by tier")
        if tiers.empty:
            st.info("No customers match these filters.")
        else:
            st.bar_chart(tiers.set_index("customer_tier")["customers"])


def show_customers(filters) -> None:
    with st.spinner("Loading customers..."):
        df = get_customers(filters)

    if df.empty:
        st.info("No customers match these filters.")
        return

    if filters.customer_id is not None:
        st.caption(f"Customer {filters.customer_id}")
    else:
        st.caption(f"Top {len(df):,} customers by lifetime spend (capped at {TABLE_ROW_LIMIT:,} rows -- narrow the filters to see fewer, more specific customers).")
    st.dataframe(df, use_container_width=True, hide_index=True)


def show_sessions(filters) -> None:
    with st.spinner("Loading sessions..."):
        df = get_sessions(filters)
        by_channel = get_sessions_by_channel(filters)

    if df.empty:
        st.info("No sessions match these filters.")
        return

    if not by_channel.empty:
        st.subheader("Sessions and bounce rate by channel")
        c1, c2 = st.columns(2)
        c1.bar_chart(by_channel.set_index("traffic_source")["sessions"])
        c2.bar_chart(by_channel.set_index("traffic_source")["bounce_rate"])

    if filters.customer_id is not None:
        st.caption(f"Sessions for customer {filters.customer_id}")
    else:
        st.caption(f"Most recent {len(df):,} sessions (capped at {TABLE_ROW_LIMIT:,} rows).")
    st.dataframe(df, use_container_width=True, hide_index=True)


def show_transactions(filters) -> None:
    with st.spinner("Loading transactions..."):
        df = get_transactions(filters)
        by_payment = get_transactions_by_payment_method(filters)

    if df.empty:
        st.info("No transactions match these filters.")
        return

    if not by_payment.empty:
        st.subheader("Revenue by payment method")
        st.bar_chart(by_payment.set_index("payment_method")["revenue"])

    if filters.customer_id is not None:
        st.caption(f"Transactions for customer {filters.customer_id}")
    else:
        st.caption(f"Most recent {len(df):,} transactions (capped at {TABLE_ROW_LIMIT:,} rows).")
    st.dataframe(df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
