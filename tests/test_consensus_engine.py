import unittest

from consensus_engine import Lens, build_consensus, confidence_interpretation


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

    def test_neutral_and_reconstructed_data_prevent_100_percent(self):
        result = build_consensus({
            "overall": Lens("종합", 84, change=4.8, data_quality=.82),
            "quant": Lens("퀀트", 84, change=3.7, data_quality=.82),
            "options": Lens("옵션", label="Bullish", data_quality=1),
            "market": Lens("시장", 57, change=3.5, data_quality=.82),
        })
        self.assertLessEqual(result.confidence, 88)
        self.assertGreaterEqual(result.confidence, 80)

    def test_conflicting_signals_are_capped(self):
        result = build_consensus({"overall": Lens("종합", 80), "quant": Lens("퀀트", 82), "options": Lens("옵션", label="Bearish"), "market": Lens("시장", 30)})
        self.assertLessEqual(result.confidence, 72)

    def test_confidence_interpretation_explains_not_probability(self):
        self.assertIn("상승 확률", confidence_interpretation(85))
        self.assertIn("추가 확인", confidence_interpretation(60))

    def test_divergence_interpretation_explains_each_lens_and_action(self):
        result = build_consensus({"overall": Lens("종합", 25), "quant": Lens("퀀트", 30), "options": Lens("옵션", label="Bullish"), "market": Lens("시장", 57)})
        self.assertIn("종합은 25점", result.interpretation)
        self.assertIn("퀀트는 30점", result.interpretation)
        self.assertIn("옵션은 Bullish", result.interpretation)
        self.assertIn("기술적 추세 회복", result.interpretation)


if __name__ == "__main__":
    unittest.main()
