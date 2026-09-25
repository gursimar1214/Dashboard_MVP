"""
filters.py  --  Sprint 3, ticket S3-02

Shared filters rendered once in the sidebar and reused by every page:
country, tier, channel, date range, and an exact customer ID lookup.

Note: "channel" (traffic_source) only exists on sessions in this schema.
Revenue/order metrics ignore it; the Overview page says so, and the
Sessions page is where it actually narrows anything.
"""

import streamlit as st

from data_access import Filters, get_filter_options


def render_filters() -> Filters:
    options = get_filter_options()

    st.sidebar.markdown("### Filters")

    country = st.sidebar.selectbox("Country", ["All"] + options["countries"], key="filter_country")
    tier = st.sidebar.selectbox("Tier", ["All"] + options["tiers"], key="filter_tier")
    channel = st.sidebar.selectbox(
        "Channel (sessions only)", ["All"] + options["channels"], key="filter_channel"
    )

    min_date, max_date = options["min_date"], options["max_date"]
    if min_date and max_date:
        date_range = st.sidebar.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="filter_date_range",
        )
    else:
        date_range = (None, None)
    date_from, date_to = (date_range if isinstance(date_range, tuple) else (date_range, date_range))

    customer_id_text = st.sidebar.text_input(
        "Look up one customer ID (optional)", value="", key="filter_customer_id"
    ).strip()

    customer_id = None
    if customer_id_text:
        if customer_id_text.isdigit():
            customer_id = int(customer_id_text)
        else:
            st.sidebar.warning("Customer ID must be a number -- ignoring it for now.")

    return Filters(
        country=country,
        tier=tier,
        channel=channel,
        date_from=date_from,
        date_to=date_to,
        customer_id=customer_id,
    )