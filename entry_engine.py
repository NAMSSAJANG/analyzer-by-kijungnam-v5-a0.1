from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


def _clip(value: float) -> float:
    return float(max(0, min(100, value)))


def _ret(close: pd.Series, sessions: int) -> float:
    return float((close.iloc[-1] / close.iloc[-min(sessions, len(close))] - 1) * 100) if len(close) > 1 else 0.0


@dataclass(frozen=True)
class EntrySnapshot:
    score: float
    factors: Mapping[str, float]
    interpretation: str
    details: Mapping[str, str]
    trend_strength: float = 50.0
    status: str = "진입 대기"
    path: str = "wait"
    allocation: str = "신규 진입 보류"
    plan: tuple[tuple[str, str, str], ...] = ()


def build_entry_snapshot(analysis: Mapping, quant: Mapping) -> EntrySnapshot:
    """Add pullback/momentum entry paths without changing upstream score formulas."""
    now = float(analysis["now"])
    low, high = map(float, analysis["entry_range"])
    atr = max(float(analysis.get("atr", 0)), 1e-9)
    atr_pct = atr / max(now, 1e-9) * 100
    distance_atr = abs(now - (low + high) / 2) / atr
    chase_atr = max(0.0, now - high) / atr
    position = _clip(88 - distance_atr * 15 - chase_atr * 18)

    data = analysis.get("data")
    close = pd.Series(dtype=float)
    volume = pd.Series(dtype=float)
    if isinstance(data, pd.DataFrame) and "Close" in data:
        close = data["Close"].astype(float).dropna()
        volume = data.get("Volume", pd.Series(index=data.index, dtype=float)).reindex(close.index).fillna(0).astype(float)
    ma20 = float(analysis.get("ma20", close.rolling(20).mean().iloc[-1] if len(close) >= 20 else now))
    ma50 = float(analysis.get("ma50", close.rolling(50).mean().iloc[-1] if len(close) >= 50 else now))
    ma200 = float(analysis.get("ma200", close.rolling(200).mean().iloc[-1] if len(close) >= 200 else now))
    ordered = now > ma20 > ma50 > ma200
    slope20 = (ma20 / float(close.iloc[-21:-1].mean()) - 1) * 100 if len(close) >= 21 else 0.0
    slope50 = (ma50 / float(close.iloc[-70:-20].mean()) - 1) * 100 if len(close) >= 70 else 0.0
    slope200 = (ma200 / float(close.iloc[-220:-20].mean()) - 1) * 100 if len(close) >= 220 else 0.0
    rising = slope20 > 0 and slope50 > 0 and slope200 >= 0
    ret6 = float(analysis.get("returns", {}).get("6개월", _ret(close, 126)))
    ret12 = float(analysis.get("returns", {}).get("1년", _ret(close, 252)))
    benchmark6 = float(analysis.get("benchmark_returns", {}).get("6개월", 0.0))
    benchmark12 = float(analysis.get("benchmark_returns", {}).get("1년", 0.0))
    relative_strength = _clip(50 + (ret6 - benchmark6) * .7 + (ret12 - benchmark12) * .3)
    vol_ratio = float(volume.tail(5).mean() / max(volume.tail(20).mean(), 1e-9)) if len(volume) >= 20 else 1.0
    obv = (np.sign(close.diff()).fillna(0) * volume).cumsum() if len(close) else pd.Series(dtype=float)
    obv_rising = len(obv) >= 21 and obv.iloc[-1] > obv.iloc[-21]
    prior_high = float(close.iloc[-61:-1].max()) if len(close) >= 61 else high
    breakout = now >= prior_high * .995
    rsi = float(analysis.get("rsi", 50))
    extension20 = (now / max(ma20, 1e-9) - 1) * 100
    extreme = rsi >= 82 or extension20 >= max(18.0, atr_pct * 4.5) or chase_atr >= 4.0
    overheated = rsi >= 70 or extension20 >= max(8.0, atr_pct * 2.0)
    structure_broken = now < ma50 or ma50 < ma200 or slope50 < -1.0

    trend_strength = _clip(float(quant["trend"]) * .45 + relative_strength * .25 +
                           (100 if ordered else 45) * .2 + (85 if rising else 35) * .1)
    factors = {"Trend": trend_strength, "Price Position": position,
               "Momentum": _clip(float(quant["momentum"])), "Volume / OBV": _clip(float(quant["supply"])),
               "Volatility": _clip(float(quant["volatility"])), "Market": _clip(float(analysis["market"]))}
    weights = {"Trend": .25, "Price Position": .20, "Momentum": .20, "Volume / OBV": .15, "Volatility": .10, "Market": .10}
    score = sum(factors[key] * weights[key] for key in factors)
    confirmations = sum((ordered, rising, ret6 > benchmark6 + 10, ret12 > benchmark12 + 15,
                         vol_ratio >= 1.05, obv_rising, breakout, analysis["market"] >= 45))
    strong_trend = trend_strength >= 68 and ordered and rising and ret6 > benchmark6
    pullback = low <= now <= high * 1.02 and not structure_broken and factors["Volatility"] >= 35
    if structure_broken or factors["Volatility"] < 20 or (analysis["market"] < 30 and trend_strength < 60):
        status, path = "회피", "avoid"
    elif pullback and score >= 52:
        status, path = "눌림목 진입 우호적", "pullback"
    elif strong_trend and confirmations >= 6 and not extreme:
        status, path = "모멘텀 진입 가능", "momentum"
    elif strong_trend and confirmations >= 4 and (overheated or extreme or chase_atr >= 2):
        status, path = "추격주의 / 소액 접근", "chase"
    else:
        status, path = "진입 대기", "wait"

    plans = {
        "pullback": (("1차", "추천구간 상단 지지", "30%"), ("2차", "구간 중단 지지", "30%"), ("3차", "구간 하단 지지", "40%")),
        "momentum": (("1차", "돌파 + 거래량 확인", "20%"), ("2차", "돌파선 지지 확인", "30%"), ("3차", "추세 유지·재돌파", "50%")),
        "chase": (("시험 진입", "돌파 유지 시에만", "10% 이내"), ("추가", "20일선/돌파선 눌림 대기", "확인 전 금지")),
        "wait": (("대기", "추세·가격·수급 확인", "신규 진입 보류"),),
        "avoid": (("회피", "구조 회복과 변동성 안정 확인", "신규 진입 금지"),),
    }[path]
    allocation = {"pullback": "기존 1·2·3차 분할", "momentum": "초기 20%, 확인 후 추가", "chase": "최대 10% 시험 진입", "wait": "신규 진입 보류", "avoid": "신규 진입 금지"}[path]
    heat_text = "단기 과열은 추세와 분리해 비중으로 통제합니다." if overheated else "단기 과열 신호는 제한적입니다."
    chase_text = f" 추천구간 위 추격 거리 {chase_atr:.1f} ATR." if now > high else ""
    interpretation = (f"{status} · 추세 강도 {trend_strength:.1f}, 진입 타이밍 {score:.1f}. "
                      f"정배열 {'충족' if ordered else '미충족'}, 모멘텀 확인 {confirmations}/8.{chase_text} {heat_text}")
    details = {
        "Trend": f"20/50/200 배열, 기울기, 6M {ret6:+.1f}%·12M {ret12:+.1f}% 및 상대강도를 반영합니다.",
        "Price Position": f"추천구간 {low:,.2f}~{high:,.2f}, 20일선 이격 {extension20:+.1f}%입니다.",
        "Momentum": f"RSI {rsi:.1f}; 높은 RSI만으로 자동 보류하지 않고 돌파 확인과 함께 봅니다.",
        "Volume / OBV": f"최근 거래량 {vol_ratio:.2f}배, OBV {'상승' if obv_rising else '미확인'}, 돌파 {'확인' if breakout else '미확인'}입니다.",
        "Volatility": f"ATR 변동폭은 현재가의 {atr_pct:.1f}%입니다.",
        "Market": "시장환경이 진입을 지원합니다." if factors["Market"] >= 65 else "시장환경 지원이 강하지 않아 비중을 제한합니다.",
    }
    return EntrySnapshot(round(score, 1), factors, interpretation, details, round(trend_strength, 1), status, path, allocation, plans)
