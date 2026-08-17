from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


def _clip(value: float) -> float:
    return float(max(0, min(100, value)))


@dataclass(frozen=True)
class EntrySnapshot:
    score: float
    factors: Mapping[str, float]
    interpretation: str
    details: Mapping[str, str]


def build_entry_snapshot(analysis: Mapping, quant: Mapping) -> EntrySnapshot:
    now = float(analysis["now"])
    low, high = map(float, analysis["entry_range"])
    atr = max(float(analysis.get("atr", 0)), 1e-9)
    midpoint = (low + high) / 2
    distance_atr = abs(now - midpoint) / atr
    chase_penalty = max(0, now - high) / atr * 18
    position = _clip(88 - distance_atr * 15 - chase_penalty)
    factors = {
        "Trend": _clip(float(quant["trend"])),
        "Price Position": position,
        "Momentum": _clip(float(quant["momentum"])),
        "Volume / OBV": _clip(float(quant["supply"])),
        "Volatility": _clip(float(quant["volatility"])),
        "Market": _clip(float(analysis["market"])),
    }
    atr_pct = atr / max(now, 1e-9) * 100
    details = {
        "Trend": "이동평균 배열과 중장기 수익률이 우호적입니다." if factors["Trend"] >= 65 else "중장기 추세 확인이 더 필요합니다.",
        "Price Position": f"추천구간 {low:,.2f}~{high:,.2f} 대비 현재 위치를 반영합니다.",
        "Momentum": "RSI·MACD·최근 수익률의 추진력이 양호합니다." if factors["Momentum"] >= 65 else "단기 추진력이 강하게 확인되지 않습니다.",
        "Volume / OBV": "거래량과 OBV 수급 확인이 양호합니다." if factors["Volume / OBV"] >= 65 else "거래량·OBV 확인 강도가 제한적입니다.",
        "Volatility": f"ATR 변동폭은 현재가의 {atr_pct:.1f}%입니다. 점수가 낮을수록 가격 변동 위험이 큽니다.",
        "Market": "시장환경이 진입을 지원합니다." if factors["Market"] >= 65 else "시장환경의 지원이 강하지 않습니다.",
    }
    weights = {"Trend": .25, "Price Position": .20, "Momentum": .20, "Volume / OBV": .15, "Volatility": .10, "Market": .10}
    score = sum(factors[key] * weights[key] for key in factors)
    positives = [key for key, value in factors.items() if value >= 65]
    cautions = [key for key, value in factors.items() if value < 45]
    first = f"{', '.join(positives[:2])} 신호가 우호적입니다." if positives else "현재 뚜렷하게 우세한 진입 요인이 부족합니다."
    if now > high:
        second = f"다만 현재가는 추천 진입구간 상단보다 {(now/high-1)*100:.1f}% 높아 단기 추격 진입 매력은 낮습니다."
    elif cautions:
        second = f"다만 {', '.join(cautions[:2])} 점수가 낮아 확인이 필요합니다."
    else:
        second = "현재가는 참고 진입구간과 가깝지만 지지와 거래량 확인이 필요합니다."
    return EntrySnapshot(round(score, 1), factors, f"{first} {second}", details)

