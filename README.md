# Global Ecommerce Dashboard MVP

## Folder structure

```
Global_Ecommerce_Dashboard_MVP/
├── raw_data/       raw csv files exactly as downloaded, never edit these by hand
├── clean_data/     output of src/clean_data.py, safe to delete and regenerate
├── docs/           data_dictionary.md and data_quality_report.md
├── src/            cleaning and pipeline scripts
├── sqlite_db/      the ecommerce.db file lives here from Sprint 2 onward
├── dashboard/      Streamlit app code, from Sprint 3 onward
└── chatbot/        text-to-SQL chatbot code, from Sprint 5 onward
```

## Sprint 1 — what's done

- `docs/data_dictionary.md` — every table, its grain, keys, dates, and known issues (S1-01)
- `src/clean_data.py` — repeatable cleaning script: fixes country codes, booleans,
  currency strings, dates, and drops duplicates (S1-02)
- `docs/data_quality_report.md` — generated automatically each time the cleaning
  script runs, with real row counts, nulls, duplicates, and key coverage (S1-03)

## How to run this on your machine

1. Make sure Python 3 and pandas are installed:
   ```
   pip install pandas
   ```
2. From inside this folder, run:
   ```
   python src/clean_data.py
   ```
3. Check `clean_data/` for the 5 cleaned CSVs, and `docs/data_quality_report.md`
   for the fresh quality numbers.

## What Sprint 1 found (headline issues)

- `customers.age` had 10,000 rows with the literal text `"unknown"` instead of a number.
- `country_code` had 10 different spellings collapsing down to 5 real countries (US, UK, DE, FR, IN).
- `bounce_flag` and `discount_applied` each had 6 different ways of writing true/false.
- `credit_limit` and `order_value` were stored as text with `$` and commas.
- No duplicate rows and no orphaned foreign keys were found anywhere — the raw data is otherwise structurally sound.

## Sprint 2 — what's done

- `src/build_db.py` — loads the 5 cleaned CSVs into `sqlite_db/ecommerce.db`, one table per file (S2-01)
- `src/validate_db.py` + `docs/join_validation_report.txt` — proves table counts, primary keys, foreign keys, and join cardinality are all correct on the real database (S2-02)
- `docs/dashboard_metrics_glossary.md` — the fixed formula for every KPI (revenue, AOV, bounce rate, etc.) so the dashboard and chatbot never disagree (S2-03)

### How to run Sprint 2 yourself
```
python src/build_db.py
python src/validate_db.py
```

## Next up: Sprint 3

Build the Streamlit dashboard app shell and the first charts, reading from `sqlite_db/ecommerce.db`.
