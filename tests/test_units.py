"""
test_units.py -- fast tests that do NOT need Ollama.

Run from the project root:   python -m unittest tests.test_units -v

Covers: value fixing, the SQL validator (needs sqlglot), and reply formatting.
Every past bug found in live testing has a test here.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chatbot.value_fix import canonicalize_values  # noqa: E402


class ValueFixTests(unittest.TestCase):
    def test_tier_case(self):  # bug: 'platinum' returned 0
        self.assertIn("'Platinum'", canonicalize_values("WHERE customer_tier = 'platinum'"))

    def test_country_name_to_code(self):
        self.assertIn("= 'IN'", canonicalize_values("WHERE c.country_code = 'India'"))

    def test_in_list(self):
        self.assertIn("('IN', 'US')", canonicalize_values("country_code IN ('india', 'usa')"))

    def test_same_word_different_column(self):
        self.assertIn("= 'email'", canonicalize_values("traffic_source = 'Email'"))
        self.assertIn("= 'Email'", canonicalize_values("campaign_type = 'email'"))

    def test_unknown_value_untouched(self):
        self.assertIn("'Diamond'", canonicalize_values("customer_tier = 'Diamond'"))

    def test_dates_untouched(self):
        self.assertIn("'2023-06-22'", canonicalize_values("transaction_timestamp = '2023-06-22'"))


try:
    from chatbot.sql_validator import SQLValidationError, validate_and_cap
    HAVE_SQLGLOT = True
except ImportError:  # sqlglot not installed
    HAVE_SQLGLOT = False


@unittest.skipUnless(HAVE_SQLGLOT, "sqlglot not installed")
class ValidatorTests(unittest.TestCase):
    def ok(self, sql):
        return validate_and_cap(sql)

    def bad(self, sql):
        with self.assertRaises(SQLValidationError):
            validate_and_cap(sql)

    def test_simple_select_passes_and_gets_limit(self):
        self.assertIn("LIMIT", self.ok("SELECT COUNT(*) FROM customers WHERE country_code = 'US'"))

    def test_alias_in_order_by(self):  # bug: 'revenue' rejected as unknown column
        self.ok("SELECT payment_method, SUM(order_value) AS revenue FROM transactions "
                "GROUP BY payment_method ORDER BY revenue DESC")

    def test_join_with_aliases(self):
        self.ok("SELECT c.country_code, COUNT(*) AS orders FROM transactions t "
                "JOIN customers c ON c.customer_id = t.customer_id GROUP BY c.country_code")

    def test_not_in_subquery(self):
        self.ok("SELECT COUNT(*) FROM customers WHERE customer_id NOT IN (SELECT customer_id FROM transactions)")

    def test_subquery_in_from(self):
        self.ok("SELECT COUNT(*) FROM (SELECT customer_id FROM transactions "
                "GROUP BY customer_id HAVING COUNT(*) > 5)")

    def test_wrong_table_column_is_rejected(self):  # bug: items_count on customers
        self.bad("SELECT customer_id FROM customers ORDER BY items_count DESC LIMIT 1")

    def test_qualified_wrong_table_is_rejected(self):
        self.bad("SELECT c.items_count FROM customers c")

    def test_placeholder_rejected(self):  # bug: '?' with no bound params
        self.bad("SELECT COUNT(*) FROM transactions WHERE transaction_timestamp = ?")

    def test_writes_rejected(self):
        self.bad("DELETE FROM customers")
        self.bad("DROP TABLE customers")
        self.bad("UPDATE customers SET age = 1")

    def test_two_statements_rejected(self):
        self.bad("SELECT 1 FROM customers; DROP TABLE customers")

    def test_unknown_table_and_metadata_rejected(self):
        self.bad("SELECT * FROM users")
        self.bad("SELECT name FROM sqlite_master")

    def test_big_limit_capped(self):
        self.assertIn("LIMIT 200", self.ok("SELECT customer_id FROM customers LIMIT 999999"))


class FormattingTests(unittest.TestCase):
    def setUp(self):
        from chatbot import pipeline
        self.p = pipeline

    def test_ids_years_have_no_thousands_separator(self):
        self.assertEqual(self.p._pretty_value("customer_id", 12345), "12345")
        self.assertEqual(self.p._pretty_value("year", 2023), "2023")

    def test_money_and_rates(self):
        self.assertEqual(self.p._pretty_value("revenue", 60016416.77), "$60,016,416.77")
        self.assertEqual(self.p._pretty_value("bounce_rate", 0.4231), "42.3%")

    def test_review_score_is_not_money(self):
        self.assertEqual(self.p._pretty_value("avg_review_score", 4.2), "4.20")

    def test_list_reply(self):
        text = self.p._format_rows_reply([{"country_code": "IN", "orders": 49731},
                                          {"country_code": "US", "orders": 200366}])
        self.assertIn("- IN -- orders: 49,731", text)
        self.assertIn("- US -- orders: 200,366", text)


if __name__ == "__main__":
    unittest.main()
