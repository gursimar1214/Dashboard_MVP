# Dashboard Metrics Glossary

One formula per metric, written down before any chart is built, so the
dashboard and the chatbot never disagree about what a word means.

---

### Customer count
**Definition:** number of distinct customers in the customers table.
```sql
SELECT COUNT(DISTINCT customer_id) FROM customers;
```
**Grain:** one row per customer. **Filters:** none by default; can be sliced by country_code, customer_tier, region.

---

### Order count
**Definition:** number of transactions.
```sql
SELECT COUNT(*) FROM transactions;
```
**Grain:** one row per transaction. Note: "order" and "transaction" mean the same row — this is the term the chatbot's schema doc should also use.

---

### Total sales / total revenue
**Definition:** sum of order_value across all transactions.
```sql
SELECT ROUND(SUM(order_value), 2) FROM transactions;
```
**Date field for trend charts:** transaction_timestamp. **Filters:** none by default; can be sliced by date range, country, payment_method.
**Known value at time of writing (Sprint 2 check):** $60,016,416.77 across 500,000 orders.

---

### Average order value (AOV)
**Definition:** total sales divided by order count. Do not average the per-row order_value directly if any filter is applied first — always compute as sum/count over the filtered set, so it stays mathematically consistent.
```sql
SELECT ROUND(SUM(order_value) * 1.0 / COUNT(*), 2) FROM transactions;
```
**Known value:** $120.03 overall.

---

### Session count
**Definition:** number of sessions.
```sql
SELECT COUNT(*) FROM sessions;
```
**Date field:** session_timestamp.

---

### Bounce rate
**Definition:** share of sessions where bounce_flag is true.
```sql
SELECT ROUND(AVG(CASE WHEN bounce_flag THEN 1.0 ELSE 0.0 END), 4) FROM sessions;
```
**Slice by:** device_type, traffic_source.

---

### High-value order share
**Definition:** share of transactions flagged high_value_flag = 'Yes'.
```sql
SELECT ROUND(AVG(CASE WHEN high_value_flag = 'Yes' THEN 1.0 ELSE 0.0 END), 4) FROM transactions;
```

---

### Discount usage rate
**Definition:** share of transactions where discount_applied is true.
```sql
SELECT ROUND(AVG(CASE WHEN discount_applied THEN 1.0 ELSE 0.0 END), 4) FROM transactions;
```

---

### Customer tier breakdown
**Definition:** count of customers per customer_tier.
```sql
SELECT customer_tier, COUNT(*) FROM customers GROUP BY customer_tier;
```

---

## Rules that apply to every metric above
- All totals/averages are computed directly from `transactions` or `sessions`, never from a pre-aggregated cache, so a filter (date range, country) can always be applied consistently.
- Every metric in this file was checked once against a fresh SQLite connection during Sprint 2 (see `docs/join_validation_report.txt`), not just assumed correct.
- Any new chart added later must have its formula added here **before** the chart is built, not after.
