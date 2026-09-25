# Dashboard -- Sprint 3

Streamlit app reading from `../sqlite_db/ecommerce.db`.

## Run it

From the `dashboard/` folder (or anywhere -- paths are resolved relative to this file):

```bash
pip install -r requirements.txt
streamlit run app.py
```

The database already exists at `../sqlite_db/ecommerce.db` from Sprint 2 -- nothing else to build first.

## Files, and which ticket they cover

| File | Ticket | What it does |
|---|---|---|
| `data_access.py` | **S3-03** | Every query in the app, in one place. Every formula matches `docs/dashboard_metrics_glossary.md` exactly (checked against `docs/join_validation_report.txt`'s hand-checked totals: $60,016,416.77 revenue over 500,000 orders, $120.03 AOV -- this code reproduces both). No-match filters return an empty result, not an exception; a missing/locked database raises one `DataAccessError` that `app.py` shows a clean message for. |
| `filters.py` | **S3-02** | Renders Country, Tier, Channel, date range, and an exact customer-ID lookup once in the sidebar, with sensible defaults (full date range, "All" for dropdowns), and hands back one `Filters` object every page uses. |
| `app.py` | **S3-01, S3-04** | The shell: sidebar nav across Overview/Customers/Sessions/Transactions, a title per page, a spinner while data loads, an info message on empty results, and a caught error message instead of a traceback if the database read fails. |

## Design choices worth knowing about

- **Channel doesn't touch revenue.** `transactions` has no `campaign_id` or channel column in this schema -- only `sessions` does. So the Channel filter narrows session counts and bounce rate, but not orders/revenue/AOV. The Overview page says this in a caption rather than silently ignoring the filter.
- **Customers/Sessions/Transactions pages are capped at 500 rows** (`TABLE_ROW_LIMIT` in `data_access.py`), sorted by lifetime spend or recency. These tables have 200,000 / 2,000,000 / 500,000 rows -- showing all of them would be unusable in a browser. Use the filters (especially the customer-ID lookup) to narrow in on someone specific; that lookup always returns its one row regardless of the cap.
- **Indexes are created on first run**, not stored in the repo. `data_access.py` runs `CREATE INDEX IF NOT EXISTS` on `customer_id`, the two timestamp columns, `country_code`, `customer_tier`, and `traffic_source` the first time it opens the database. This doesn't touch any data, and it's what keeps filtered queries fast against the 2,000,000-row `sessions` table (a filtered query went from ~2s to well under 0.5s after indexing in testing).
- **I couldn't run `streamlit run app.py` itself** in the sandbox that wrote this code (no network to install Streamlit). Every function in `data_access.py` was tested directly against your real `ecommerce.db`, including the empty-result and missing-database error paths, and its overview totals match `join_validation_report.txt` exactly. Worth one real `streamlit run` on your machine before marking S3 Done, to eyeball the actual page layout.
