# Data Dictionary — Global Ecommerce Dashboard MVP

Covers every raw source file in `raw_data/`. Written for Sprint 1 (S1-01).

---

## customers.csv
**Grain:** one row per customer
**Rows:** 200,000
**Primary key:** `customer_id`

| Column | Meaning | Known issues |
|---|---|---|
| customer_id | unique customer identifier | none |
| age | customer age | 10,000 rows have the literal string `"unknown"` instead of a number |
| country_code | customer's country | inconsistent casing/format: `usa`, `U.S.`, `USA`, `US`, `uk`, `UK`, `de`, `DE` all appear |
| region | internal region code (e.g. `region_50`) | none |
| signup_date | date customer signed up | none, parses cleanly, range 2022-01-01 to 2024-09-26 |
| loyalty_score | 0–1 loyalty score | none |
| email_open_rate | 0–1 rate of opened emails | 22,746 nulls |
| discount_usage_rate | 0–1 rate of using discounts | none |
| avg_review_score | average review score left by customer | none |
| referral_code | referral code string | none |
| customer_tier | Bronze / Silver / Gold / Platinum | none, already clean |
| credit_limit | credit limit | stored as text with `$` and comma, e.g. `"$10,966"` |
| noise_customer_0..4 | random noise columns, not meaningful | ignore for modeling and dashboard |

## sessions.csv
**Grain:** one row per web session
**Rows:** 2,000,000
**Keys:** `session_id` (PK), `customer_id` → customers, `campaign_id` → marketing_campaigns, `geo_ip_region` → geo_data

| Column | Meaning | Known issues |
|---|---|---|
| session_id | unique session id | none |
| customer_id | which customer this session belongs to | 0 orphan rows — every id exists in customers |
| session_timestamp | when the session happened | none, parses cleanly |
| session_duration | length of session | none |
| pages_viewed | pages viewed in session | 100,000 nulls |
| cart_additions | items added to cart | none |
| bounce_flag | did the customer bounce | mixed formats: `0`, `1`, `TRUE`, `FALSE`, `Yes`, `No` |
| traffic_source | where the session came from | mixed casing: `organic`/`Organic`, `ads`/`Ads`, `SOCIAL`, `email` |
| device_type | device used | mixed casing: `Desktop`/`desktop`, `Mobile`/`mobile`, `TABLET` |
| campaign_id | linked marketing campaign | 0 orphan rows |
| geo_ip_region | region of the session's IP | 0 orphan rows |

## transactions.csv
**Grain:** one row per transaction
**Rows:** 500,000
**Keys:** `transaction_id` (PK), `customer_id` → customers

| Column | Meaning | Known issues |
|---|---|---|
| transaction_id | unique transaction id | none |
| customer_id | which customer made this purchase | 0 orphan rows |
| transaction_timestamp | when the purchase happened | none, parses cleanly |
| order_value | amount paid | stored as text with `$`, e.g. `"$20.39"` |
| items_count | number of items in the order | none |
| payment_method | debit_card / paypal / credit_card / crypto | already clean |
| discount_applied | was a discount used | mixed formats: `0`, `1`, `TRUE`, `FALSE`, `Yes`, `No` |
| shipping_speed | standard / express / overnight | already clean |
| high_value_flag | pre-computed high-value order flag | already clean (Yes/No) |
| noise_trans_0..1 | random noise columns | ignore for modeling and dashboard |

## geo_data.csv
**Grain:** one row per region
**Rows:** 100
**Primary key:** `geo_ip_region`

| Column | Meaning | Known issues |
|---|---|---|
| geo_ip_region | region code, matches sessions.geo_ip_region | none |
| average_income | average income for the region | none |
| urban_ratio | share of urban population | none |
| internet_penetration | share with internet access | none |
| region_tier | region classification tier | none |

## marketing_campaigns.csv
**Grain:** one row per campaign
**Rows:** 200
**Primary key:** `campaign_id`

| Column | Meaning | Known issues |
|---|---|---|
| campaign_id | campaign identifier, matches sessions.campaign_id | none |
| campaign_type | type of campaign | none |
| campaign_budget | budget for the campaign | none |
| region_target | region the campaign targeted | none |
| start_date | campaign start | parses cleanly |
| end_date | campaign end | parses cleanly |

## train.csv
**Grain:** one row per customer (subset used for model training)
**Rows:** 160,000
**Key:** `customer_id` → customers

| Column | Meaning |
|---|---|
| customer_id | matches customers.customer_id, 0 orphan rows |
| target | 0/1 label — approx 80% are 0, 20% are 1 (imbalanced) |

## test.csv
**Grain:** one row per customer needing a prediction
**Rows:** 40,000
**Key:** `customer_id` → customers, 0 orphan rows

---

## Cross-table integrity (checked directly, not assumed)
- Every `customer_id` in sessions, transactions, train, and test exists in customers — 0 orphans.
- Every `geo_ip_region` in sessions exists in geo_data — 0 orphans.
- Every `campaign_id` in sessions exists in marketing_campaigns — 0 orphans.
- No duplicate rows found in any of the 7 files.
