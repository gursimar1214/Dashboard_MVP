import unittest

from ..sql_validator import SQLValidationError, validate_and_cap


class CampaignBudgetFanoutGuardTests(unittest.TestCase):
    def test_rejects_campaign_budget_multiplied_by_raw_sessions(self):
        bad_sql = (
            "SELECT m.campaign_type, SUM(m.campaign_budget) AS total_budget, "
            "COUNT(DISTINCT s.session_id) AS total_sessions "
            "FROM marketing_campaigns AS m "
            "JOIN sessions AS s ON m.campaign_id = s.campaign_id "
            "GROUP BY m.campaign_type LIMIT 200"
        )
        with self.assertRaisesRegex(SQLValidationError, "campaign budget is multiplied"):
            validate_and_cap(bad_sql)

    def test_accepts_preaggregated_sessions_before_campaign_budget_join(self):
        correct_sql = (
            "WITH sessions_by_campaign AS ("
            "SELECT campaign_id, COUNT(DISTINCT session_id) AS total_sessions "
            "FROM sessions GROUP BY campaign_id) "
            "SELECT m.campaign_type, SUM(m.campaign_budget) AS total_budget, "
            "SUM(COALESCE(s.total_sessions, 0)) AS total_sessions "
            "FROM marketing_campaigns AS m "
            "LEFT JOIN sessions_by_campaign AS s ON m.campaign_id = s.campaign_id "
            "GROUP BY m.campaign_type LIMIT 200"
        )
        validate_and_cap(correct_sql)


if __name__ == "__main__":
    unittest.main()