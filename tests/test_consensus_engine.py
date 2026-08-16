import unittest

from consensus_engine import Lens, build_consensus


class ConsensusTests(unittest.TestCase):
    def test_missing_options_is_excluded(self):
        result = build_consensus({"overall": Lens("종합", 75), "quant": Lens("퀀트", 80), "options": Lens("옵션", available=False), "market": Lens("시장", 70)})
        self.assertEqual(result.headline, "3 / 3 Positive")
        self.assertEqual(result.available, 3)

    def test_quality_timing_divergence(self):
        result = build_consensus({"overall": Lens("종합", 42), "quant": Lens("퀀트", 82), "options": Lens("옵션", label="Bearish"), "market": Lens("시장", 55)})
        self.assertEqual(result.pattern, "Quality vs Timing")
        self.assertEqual(result.headline, "Divergence Detected")

    def test_strengthening_uses_changes_not_average(self):
        result = build_consensus({"overall": Lens("종합", 70, change=5), "quant": Lens("퀀트", 73, change=4), "market": Lens("시장", 68, change=3)})
        self.assertEqual(result.pattern, "Strengthening")


if __name__ == "__main__":
    unittest.main()
