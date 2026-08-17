import unittest

import numpy as np
import pandas as pd

from entry_engine import build_entry_snapshot


class EntryEngineTests(unittest.TestCase):
    @staticmethod
    def case(growth=1.0, rsi=75, volatility=65, market=70, volume_boost=1.5):
        close=pd.Series(np.geomspace(50,100*(1+growth),252))
        volume=pd.Series(np.full(252,1000.0)); volume.iloc[-5:]*=volume_boost
        frame=pd.DataFrame({"Close":close,"Volume":volume})
        now=float(close.iloc[-1]); ma20=float(close.rolling(20).mean().iloc[-1]); ma50=float(close.rolling(50).mean().iloc[-1]); ma200=float(close.rolling(200).mean().iloc[-1])
        analysis={"now":now,"entry_range":(ma20*.97,ma20*1.01),"atr":now*.025,"market":market,"rsi":rsi,"data":frame,"ma20":ma20,"ma50":ma50,"ma200":ma200,"returns":{"6개월":45,"1년":100}}
        quant={"trend":88,"momentum":82,"supply":78,"volatility":volatility}
        return analysis,quant

    def test_score_uses_six_explainable_factors(self):
        analysis = {"now": 110, "entry_range": (95, 100), "atr": 5, "market": 70}
        quant = {"trend": 80, "momentum": 70, "supply": 65, "volatility": 60}
        result = build_entry_snapshot(analysis, quant)
        self.assertEqual(len(result.factors), 6)
        self.assertLess(result.factors["Price Position"], 45)
        self.assertIn("추격", result.interpretation)
        self.assertIn("ATR", result.details["Volatility"])

    def test_mu_like_strong_overbought_trend_is_not_automatic_wait(self):
        analysis,quant=self.case(rsi=76)
        result=build_entry_snapshot(analysis,quant)
        self.assertIn(result.status,("모멘텀 진입 가능","추격주의 / 소액 접근"))

    def test_sndk_like_extreme_extension_limits_size(self):
        analysis,quant=self.case(rsi=84)
        analysis["entry_range"]=(analysis["ma20"]*.80,analysis["ma20"]*.85)
        result=build_entry_snapshot(analysis,quant)
        self.assertEqual(result.status,"추격주의 / 소액 접근")
        self.assertIn("10%",result.allocation)

    def test_regression_weak_structure_is_not_upgraded(self):
        analysis,quant=self.case(growth=-.35,rsi=42,market=35)
        analysis["ma20"]=analysis["now"]*1.05; analysis["ma50"]=analysis["now"]*1.12; analysis["ma200"]=analysis["now"]*1.02
        quant.update(trend=30,momentum=35,supply=40,volatility=15)
        result=build_entry_snapshot(analysis,quant)
        self.assertEqual(result.status,"회피")


if __name__ == "__main__":
    unittest.main()
