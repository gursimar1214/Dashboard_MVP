import unittest

from ..date_resolver import resolve_time_window


class DateResolverTests(unittest.TestCase):
    def test_this_year_uses_latest_data_date_inclusive(self):
        w = resolve_time_window("this year", "2024-09-26")
        self.assertEqual((w.start.isoformat(), w.end.isoformat()), ("2024-01-01", "2024-09-27"))

    def test_last_month_is_previous_calendar_month(self):
        w = resolve_time_window("last month", "2024-09-26")
        self.assertEqual((w.start.isoformat(), w.end.isoformat()), ("2024-08-01", "2024-09-01"))

    def test_last_quarter_crosses_year_boundary(self):
        w = resolve_time_window("last quarter", "2024-02-15")
        self.assertEqual((w.start.isoformat(), w.end.isoformat()), ("2023-10-01", "2024-01-01"))

    def test_unrecognized_phrase_is_not_guessed(self):
        self.assertIsNone(resolve_time_window("last 3 months", "2024-09-26"))


if __name__ == "__main__":
    unittest.main()
