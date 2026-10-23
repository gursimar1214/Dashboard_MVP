"""
schema_glossary.py

The single source of truth the LLM is grounded on for stages 3 and 4,
and that the validator in stage 5 checks generated SQL against.

IMPORTANT: the formulas in METRICS_GLOSSARY for revenue, orders, AOV,
discount usage, high-value share, sessions and bounce rate are copied from
the exact SQL in dashboard/data_access.py's get_overview_metrics(). If a
formula changes there, mirror the change here. The remaining entries
(customer, geo and campaign metrics) are not on the dashboard overview; their
formulas are plain AVG/SUM over the column and were checked against the data.

Facts stated below that were verified against ecommerce.db (re-check them if
the database is regenerated -- tests/test_audit_fixes.py re-checks the joins):
  - every session has a campaign_id that exists in marketing_campaigns;
  - customers.region and sessions.geo_ip_region differ in 99% of sessions;
  - email_open_rate ~11% NULL, age 5% NULL, sessions.pages_viewed 5% NULL;
  - age, session_duration and pages_viewed are stored as REAL.
"""

SCHEMA_DESCRIPTION = """
Tables and columns (SQLite):

customers (
  customer_id INTEGER PRIMARY KEY, age REAL, country_code TEXT,
  region TEXT, signup_date TEXT, loyalty_score REAL,
  email_open_rate REAL, discount_usage_rate REAL, avg_review_score REAL,
  referral_code TEXT, customer_tier TEXT, credit_limit REAL
)

sessions (
  session_id INTEGER PRIMARY KEY, customer_id INTEGER,
  session_timestamp TEXT, session_duration REAL, pages_viewed REAL,
  cart_additions INTEGER, bounce_flag INTEGER, traffic_source TEXT,
  device_type TEXT, campaign_id INTEGER, geo_ip_region TEXT
)

transactions (
  transaction_id INTEGER PRIMARY KEY, customer_id INTEGER,
  transaction_timestamp TEXT, order_value REAL, items_count INTEGER,
  payment_method TEXT, discount_applied INTEGER, shipping_speed TEXT,
  high_value_flag TEXT
)

geo_data (
  geo_ip_region TEXT PRIMARY KEY, average_income REAL, urban_ratio REAL,
  internet_penetration REAL, region_tier TEXT
)

marketing_campaigns (
  campaign_id INTEGER PRIMARY KEY, campaign_type TEXT,
  campaign_budget REAL, region_target TEXT, start_date TEXT, end_date TEXT
)

Relationships (no NULL or dangling keys on any of these, so a plain JOIN is safe):
  sessions.customer_id     -> customers.customer_id
  transactions.customer_id -> customers.customer_id
  sessions.campaign_id     -> marketing_campaigns.campaign_id  (every session has a campaign)
  customers.region         -> geo_data.geo_ip_region           (the customer's HOME region)
  sessions.geo_ip_region   -> geo_data.geo_ip_region           (where that one SESSION came from)
""".strip()

VALID_VALUES_TEXT = """
Valid values for text columns. SQLite text comparison is CASE-SENSITIVE,
so use EXACTLY this spelling:
- customers.country_code: 'US', 'UK', 'DE', 'FR', 'IN'
  (United States/USA/America -> 'US', United Kingdom/Britain -> 'UK',
   Germany -> 'DE', France -> 'FR', India -> 'IN')
- customers.customer_tier: 'Bronze', 'Silver', 'Gold', 'Platinum'
- sessions.traffic_source: 'organic', 'ads', 'social', 'email'
- sessions.device_type: 'desktop', 'mobile', 'tablet'
- transactions.payment_method: 'debit_card', 'credit_card', 'paypal', 'crypto'
- transactions.shipping_speed: 'standard', 'express', 'overnight'
- transactions.high_value_flag: 'Yes', 'No'
- marketing_campaigns.campaign_type: 'Affiliate', 'Email', 'Search', 'Display', 'Social'
- geo_data.region_tier: 'Tier1', 'Tier2', 'Tier3'
- customers.region, sessions.geo_ip_region, geo_data.geo_ip_region,
  marketing_campaigns.region_target: 'region_0' ... 'region_99' -- REGION
  codes, NOT countries.
- All dates/timestamps are 'YYYY-MM-DD' text (no time part).
""".strip()

METRICS_GLOSSARY = """
Orders (transactions):
- revenue = SUM(transactions.order_value)
- orders = COUNT(transactions.transaction_id)
- average order value (AOV) = revenue / orders (0 if orders = 0)
- discount usage rate (of orders) = AVG(CASE WHEN transactions.discount_applied THEN 1.0 ELSE 0.0 END)
- high-value order share = AVG(CASE WHEN transactions.high_value_flag = 'Yes' THEN 1.0 ELSE 0.0 END)
- items per order = AVG(transactions.items_count)
Sessions (sessions):
- sessions = COUNT(sessions.session_id)
- bounce rate = AVG(CASE WHEN sessions.bounce_flag THEN 1.0 ELSE 0.0 END)
- session duration / pages per session / cart additions per session = AVG of session_duration / pages_viewed / cart_additions
Customers (one row each, so AVG the column):
- customer count = COUNT(DISTINCT customers.customer_id); new customers = COUNT(*) filtered on signup_date
- email open rate ("opens emails", email engagement) = AVG(customers.email_open_rate), 0 to 1
- loyalty score = AVG(customers.loyalty_score), 0 to 1
- review score / rating = AVG(customers.avg_review_score)
- credit limit = AVG(customers.credit_limit) (SUM only for "total credit")
- customer age = AVG(customers.age)
- customers.discount_usage_rate is a per-customer attribute, NOT the order-level rate above (~0.33 vs ~0.50): use AVG(customers.discount_usage_rate) only when asked about customers' own discount usage.
Geo (geo_data, one row per region): AVG(average_income), AVG(urban_ratio), AVG(internet_penetration); the two ratios are 0 to 1.
Campaigns (marketing_campaigns): campaigns = COUNT(*); campaign budget = SUM(campaign_budget) from marketing_campaigns ALONE -- joining sessions first multiplies each budget by its session count.

FROM / GROUP BY: the table a metric lives on decides FROM; JOIN only to reach the grouping column.
- customer metrics by country_code / customer_tier / region: customers only.
- order metrics by country / tier / region: transactions JOIN customers; by payment_method / shipping_speed / month: transactions only.
- session metrics by traffic_source / device_type / geo_ip_region: sessions only; by campaign_type: JOIN marketing_campaigns; by country / tier / customer region: JOIN customers.
- geo metrics by region_tier: geo_data only.
""".strip()

COMMON_MISTAKES = """
Common mistakes to avoid:
1. traffic_source / channel exists ONLY on sessions. Never filter a
   revenue/order/AOV query by it.
2. Country lives ONLY in customers.country_code. sessions and
   transactions have no country column: JOIN customers ON customer_id.
3. "purchase" / "order" / "sale" means one row in transactions.
   COUNT(*) on transactions -- NOT SUM(items_count), NOT a customers
   column. To count purchases per customer, JOIN customers with
   transactions and GROUP BY customer_id.
4. "sales" / "how many sales" = NUMBER OF ORDERS (COUNT(*) on
   transactions), not SUM(order_value). Only SUM(order_value) when the
   question says revenue / amount / money / value.
5. "india and us" -> col IN ('IN','US'), NEVER col = 'IN' AND col = 'US'.
   Add GROUP BY that column so each value gets its own row.
6. Never compare a country to sessions.geo_ip_region -- those are region
   codes like 'region_74'.
7. Write literal values directly in the SQL. No parameter placeholders.
8. Always include a LIMIT <= 200, unless the query returns one row.
9. Two region columns disagree in 99% of sessions. customers.region = the
   customer's HOME region: use it for customer metrics (email open rate,
   loyalty, reviews, credit, age) and for revenue / orders by region (JOIN
   customers). sessions.geo_ip_region = where one SESSION came from: use it
   only for session metrics by region (sessions, bounce, device, traffic) or
   when the question says "geo ip". Join geo_data on customers.region for
   customer / order data, on sessions.geo_ip_region for session data.
10. email_open_rate (11% NULL), age and pages_viewed (5% NULL): AVG / MIN /
   MAX skip NULLs, but when ranking rows by one of them add `col IS NOT NULL`
   -- SQLite sorts NULL first in ASC, so "lowest" would return NULL rows.
11. "email" is ambiguous: traffic_source = 'email' is a channel;
   marketing_campaigns.campaign_type = 'Email' is a campaign type (JOIN on
   campaign_id). "email campaign" -> campaign_type; "email traffic / channel"
   -> traffic_source. The two are independent.
12. For "above the average" or "top N per group" use a subquery: WHERE x >
   (SELECT AVG(...) ...), or FROM (SELECT ..., ROW_NUMBER() OVER (PARTITION
   BY ... ORDER BY ...) AS rn ...) AS alias WHERE rn <= N. Always alias a
   derived table.
13. "Above-average customer spend" compares each customer's SUM(order_value)
    with the average of those per-customer sums. Do NOT compare it with
    AVG(order_value), which is the average single order.
14. A customer attribute grouped by sessions.device_type must first build
    DISTINCT (customer_id, device_type) pairs from sessions, then join those
    pairs to customers. This counts each customer once per device, regardless
    of how many sessions they had on that device.
15. When the question prompt gives an exact relative-date range, use those
    exact inclusive/exclusive bounds on transactions.transaction_timestamp.
    Do not guess a different year. Do not apply transaction dates to customer
    attributes that have no date column.
""".strip()

# Stage 5 whitelist.
ALLOWED_TABLES: dict[str, set[str]] = {
    "customers": {
        "customer_id", "age", "country_code", "region", "signup_date",
        "loyalty_score", "email_open_rate", "discount_usage_rate",
        "avg_review_score", "referral_code", "customer_tier", "credit_limit",
    },
    "sessions": {
        "session_id", "customer_id", "session_timestamp", "session_duration",
        "pages_viewed", "cart_additions", "bounce_flag", "traffic_source",
        "device_type", "campaign_id", "geo_ip_region",
    },
    "transactions": {
        "transaction_id", "customer_id", "transaction_timestamp", "order_value",
        "items_count", "payment_method", "discount_applied", "shipping_speed",
        "high_value_flag",
    },
    "geo_data": {
        "geo_ip_region", "average_income", "urban_ratio",
        "internet_penetration", "region_tier",
    },
    "marketing_campaigns": {
        "campaign_id", "campaign_type", "campaign_budget", "region_target",
        "start_date", "end_date",
    },
}
