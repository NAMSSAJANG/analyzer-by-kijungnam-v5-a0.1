import unittest

from entry_engine import build_entry_snapshot


class EntryEngineTests(unittest.TestCase):
    def test_score_uses_six_explainable_factors(self):
        analysis = {"now": 110, "entry_range": (95, 100), "atr": 5, "market": 70}
        quant = {"trend": 80, "momentum": 70, "supply": 65, "volatility": 60}
        result = build_entry_snapshot(analysis, quant)
        self.assertEqual(len(result.factors), 6)
        self.assertLess(result.factors["Price Position"], 45)
        self.assertIn("추격", result.interpretation)
        self.assertIn("ATR", result.details["Volatility"])


if __name__ == "__main__":
    unittest.main()
