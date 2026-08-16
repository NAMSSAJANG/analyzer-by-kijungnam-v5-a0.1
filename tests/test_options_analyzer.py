from datetime import date
import unittest

import pandas as pd

from options_analyzer import calculate_max_pain, option_bias, option_entry_readiness, summarize_options


class OptionSummaryTests(unittest.TestCase):
    def setUp(self):
        self.calls = pd.DataFrame({
            "strike": [95, 100, 105], "volume": [20, 100, 40], "openInterest": [200, 500, 300],
            "impliedVolatility": [.30, .32, .35], "bid": [6, 3, 1], "ask": [6.4, 3.4, 1.4],
            "lastPrice": [6.2, 3.2, 1.2],
        })
        self.puts = pd.DataFrame({
            "strike": [95, 100, 105], "volume": [30, 50, 20], "openInterest": [400, 250, 100],
            "impliedVolatility": [.36, .34, .33], "bid": [1, 2.5, 6], "ask": [1.4, 2.9, 6.4],
            "lastPrice": [1.2, 2.7, 6.2],
        })

    def test_ratios_walls_and_expected_move(self):
        result = summarize_options(self.calls, self.puts, 100, "2026-09-18", date(2026, 8, 15))
        self.assertEqual(result.call_volume, 160)
        self.assertEqual(result.put_volume, 100)
        self.assertAlmostEqual(result.volume_ratio, .625)
        self.assertEqual(result.call_wall, 100)
        self.assertEqual(result.put_wall, 95)
        self.assertAlmostEqual(result.expected_move, 5.9)
        self.assertEqual(result.confirmation, "Bullish confirmation")

    def test_sparse_chain_does_not_crash(self):
        result = summarize_options(pd.DataFrame({"strike": [100]}), pd.DataFrame(), 100, "2026-09-18", date(2026, 8, 15))
        self.assertEqual(result.call_volume, 0)
        self.assertEqual(result.put_oi, 0)
        self.assertEqual(result.expected_move, 0)
        self.assertEqual(result.confirmation, "Neutral confirmation")

    def test_max_pain_minimizes_intrinsic_payout(self):
        calls = pd.DataFrame({"strike": [90, 100, 110], "openInterest": [10, 100, 10]})
        puts = pd.DataFrame({"strike": [90, 100, 110], "openInterest": [10, 100, 10]})
        self.assertEqual(calculate_max_pain(calls, puts), 100.0)

    def test_bias_has_five_levels(self):
        result = summarize_options(self.calls, self.puts, 100, "2026-09-18", date(2026, 8, 15))
        self.assertEqual(option_bias(result), "Mild Bullish")

    def test_option_entry_has_five_factors(self):
        summary = summarize_options(self.calls, self.puts, 100, "2026-09-18", date(2026, 8, 15))
        entry = option_entry_readiness(summary, self.calls, self.puts, 100, "2026-09-18")
        self.assertEqual(len(entry.factors), 5)
        self.assertGreaterEqual(entry.score, 0)
        self.assertLessEqual(entry.score, 100)
        self.assertIn("만기까지", entry.details["Time / DTE"])


if __name__ == "__main__":
    unittest.main()
