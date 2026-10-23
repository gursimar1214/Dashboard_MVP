import sqlite3
import unittest

from ..fastpath import match


class FastPathTests(unittest.TestCase):
    def test_above_average_customer_spend_uses_customer_averages(self):
        sql = match("customers whose total spend is above the overall average")
        self.assertIsNotNone(sql)
        self.assertIn("GROUP BY t.customer_id", sql)
        self.assertIn("AVG(customer_spend)", sql)

    def test_count_above_average_customer_spend(self):
        sql = match("how many customers have a total spend above the overall average")
        self.assertIsNotNone(sql)
        self.assertIn("COUNT(*)", sql)
        self.assertIn("AVG(customer_spend)", sql)

    def test_fastpath_query_runs_and_uses_average_customer_total(self):
        sql = match("customers whose total spend is above the overall average")
        con = sqlite3.connect(":memory:")
        con.execute("CREATE TABLE customers(customer_id INTEGER PRIMARY KEY)")
        con.executemany("INSERT INTO customers VALUES (?)", [(1,), (2,), (3,)])
        con.execute("CREATE TABLE transactions(customer_id INTEGER, order_value REAL)")
        con.executemany("INSERT INTO transactions VALUES (?, ?)", [(1, 60), (1, 60), (2, 20), (3, 40)])
        rows = con.execute(sql).fetchall()
        self.assertEqual(rows, [(1, 120.0)])

    def test_aov_slots_resolve_this_year_from_the_data_date(self):
        from ..fastpath import match_slots
        sql = match_slots("aov", None, {"country_code": "DE", "customer_tier": "Gold"},
                          "this year", reference_date="2024-09-26")
        self.assertIn("t.transaction_timestamp >= '2024-01-01'", sql)
        self.assertIn("t.transaction_timestamp < '2024-09-27'", sql)
        con = sqlite3.connect(":memory:")
        con.executescript("""
            CREATE TABLE customers(customer_id INTEGER, country_code TEXT, customer_tier TEXT);
            CREATE TABLE transactions(customer_id INTEGER, order_value REAL, transaction_timestamp TEXT);
            INSERT INTO customers VALUES (1, 'DE', 'Gold'), (2, 'DE', 'Silver');
            INSERT INTO transactions VALUES
                (1, 60, '2024-01-01'), (1, 40, '2024-09-26'),
                (1, 999, '2024-09-27'), (2, 500, '2024-02-01');
        """)
        self.assertEqual(con.execute(sql).fetchone()[0], 50.0)

    def test_email_open_rate_by_device_deduplicates_customer_device_pairs(self):
        from ..fastpath import match_slots
        sql = match_slots("email_open_rate", "device", {"customer_tier": "Gold"},
                          rank=None, reference_date="2024-09-26")
        con = sqlite3.connect(":memory:")
        con.executescript("""
            CREATE TABLE customers(customer_id INTEGER, customer_tier TEXT, email_open_rate REAL);
            CREATE TABLE sessions(customer_id INTEGER, device_type TEXT);
            INSERT INTO customers VALUES (1, 'Gold', 0.8), (2, 'Gold', 0.2);
            INSERT INTO sessions VALUES
                (1, 'desktop'), (1, 'desktop'), (1, 'desktop'), (1, 'mobile'),
                (2, 'desktop');
        """)
        rows = con.execute(sql).fetchall()
        self.assertEqual(rows, [("desktop", 0.5), ("mobile", 0.8)])


if __name__ == "__main__":
    unittest.main()
