import tempfile
import unittest
from datetime import date
from pathlib import Path

from score_history import JsonScoreHistory


class HistoryTests(unittest.TestCase):
    def test_same_day_is_replaced_and_recent_change_is_first_to_last(self):
        with tempfile.TemporaryDirectory() as folder:
            store = JsonScoreHistory(Path(folder) / "history.json")
            store.record("MU", {"overall": 70}, date(2026, 8, 13))
            store.record("MU", {"overall": 72}, date(2026, 8, 13))
            store.record("MU", {"overall": 78}, date(2026, 8, 14))
            trend = store.recent("MU", "overall")
            self.assertEqual(trend.values, (72.0, 78.0))
            self.assertEqual(trend.change, 6.0)


if __name__ == "__main__":
    unittest.main()
