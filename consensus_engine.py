from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class Lens:
    name: str
    score: float | None = None
    label: str | None = None
    change: float | None = None
    available: bool = True
    data_quality: float = 1.0

    @property
    def direction(self) -> int:
        if not self.available:
            return 0
        if self.score is not None:
            return 1 if self.score >= 65 else -1 if self.score < 45 else 0
        text = (self.label or "").lower()
        return 1 if "bull" in text or "positive" in text else -1 if "bear" in text or "negative" in text else 0


@dataclass(frozen=True)
class Consensus:
    headline: str
    pattern: str
    confidence: int
    interpretation: str
    available: int
    positive: int


def confidence_interpretation(value: int) -> str:
    if value >= 85:
        return "높음 · 분석 방향과 데이터 품질이 대체로 일관적입니다. 상승 확률을 의미하지는 않습니다."
    if value >= 70:
        return "보통 · 주요 방향은 확인되지만 일부 중립·결측·품질 저하 요인이 있습니다."
    return "낮음 · 분석 신호가 엇갈리거나 데이터가 충분하지 않아 추가 확인이 필요합니다."


def _score_view(name: str, score: float | None) -> str:
    if score is None:
        return f"{name}은 N/A입니다."
    if name == "종합":
        view = "현물 전반 평가가 매우 강합니다" if score >= 80 else "현물 전반 평가가 우호적입니다" if score >= 65 else "현물 전반 평가가 중립적입니다" if score >= 45 else "현물 전반 평가가 약합니다" if score >= 30 else "현물 전반 평가가 매우 약합니다"
    elif name == "퀀트":
        view = "기업 품질과 정량 타이밍이 강합니다" if score >= 80 else "기업 품질과 정량 타이밍이 우호적입니다" if score >= 65 else "기업 품질과 타이밍 신호가 혼재합니다" if score >= 45 else "기업 품질·타이밍 확인이 부족합니다" if score >= 30 else "기업 품질·타이밍 평가가 매우 약합니다"
    else:
        view = "시장환경이 강하게 우호적입니다" if score >= 80 else "시장환경이 우호적입니다" if score >= 65 else "시장환경은 중립적입니다" if score >= 45 else "시장환경이 비우호적입니다" if score >= 30 else "시장환경이 매우 비우호적입니다"
    particle = "는" if name == "퀀트" else "은"
    return f"{name}{particle} {score:.0f}점으로 {view}."


def detailed_interpretation(lenses: Mapping[str, Lens], pattern: str) -> str:
    overall, quant, options, market = (lenses.get(key) for key in ("overall", "quant", "options", "market"))
    parts = []
    if overall and overall.available:
        parts.append(_score_view("종합", overall.score))
    if quant and quant.available:
        parts.append(_score_view("퀀트", quant.score))
    if options and options.available:
        bias = options.label or "Neutral"
        option_view = "단기 상승 기대가 강하게 나타납니다" if bias == "Bullish" else "완만한 상승 편향이 나타납니다" if bias == "Mild Bullish" else "뚜렷한 단기 방향을 확인하지 않습니다" if bias == "Neutral" else "완만한 하락 경계가 나타납니다" if bias == "Mild Bearish" else "단기 하락 경계가 강하게 나타납니다"
        parts.append(f"옵션은 {bias}로 {option_view}.")
    else:
        parts.append("옵션은 N/A로 이번 합의 판단에서 제외했습니다.")
    if market and market.available:
        parts.append(_score_view("시장", market.score))

    if overall and quant and options and options.available and overall.direction < 0 and quant.direction < 0 and options.direction > 0:
        action = "옵션 강세가 약한 현물·퀀트 평가와 충돌하므로 추격보다 기술적 추세 회복과 주요 저항 돌파를 먼저 확인하세요."
    elif overall and quant and overall.direction > 0 and quant.direction > 0 and options and options.available and options.direction < 0:
        action = "현물과 기업 평가는 우호적이지만 옵션시장이 경계하고 있으므로 단기 지지 유지와 변동성 확대 여부를 확인하세요."
    elif pattern == "Strengthening":
        action = "여러 점수가 함께 개선되고 있으므로 현재 강도가 유지되는지 지지선과 거래량으로 확인하세요."
    elif pattern == "Broad Strength":
        action = "대부분의 렌즈가 일치하지만 추격보다 제시된 진입가격대와 무효화선을 기준으로 접근하세요."
    elif pattern == "Broad Weakness":
        action = "대부분의 렌즈가 약하므로 신규 진입보다 추세 회복과 시장환경 개선을 먼저 확인하세요."
    elif pattern == "Quality vs Timing":
        action = "좋은 기업 평가와 현재 진입 타이밍이 분리되어 있으므로 가격 지지와 옵션 확인이 생길 때까지 기다리는 편이 유리합니다."
    else:
        action = "분석 방향이 엇갈리므로 단일 점수보다 기술적 지지·저항과 옵션 확인 조건을 우선하세요."
    parts.append(action)
    return " ".join(parts)


def build_consensus(lenses: Mapping[str, Lens]) -> Consensus:
    active = [x for x in lenses.values() if x.available]
    positive = sum(x.direction > 0 for x in active)
    negative = sum(x.direction < 0 for x in active)
    neutral = sum(x.direction == 0 for x in active)
    # Neutral is useful information, not an item to silently remove. It therefore
    # lowers agreement while unavailable data is handled separately by coverage.
    alignment = max(positive, negative, neutral) / len(active) if active else 0
    improving = [x.change for x in active if x.change is not None]
    improving_count = sum(x >= 2 for x in improving)
    weakening_count = sum(x <= -2 for x in improving)
    stable_count = len(improving) - improving_count - weakening_count
    momentum_alignment = max(improving_count, weakening_count, stable_count) / len(improving) if improving else .5
    quality = sum(max(0, min(1, x.data_quality)) for x in active) / len(active) if active else 0
    coverage = len(active) / max(len(lenses), 1)
    confidence = round(100 * (.45 * alignment + .25 * quality + .15 * coverage + .15 * momentum_alignment))
    # Guardrails keep the number interpretable and prevent incomplete or mixed
    # inputs from presenting false precision.
    confidence = min(confidence, 95)
    if neutral:
        confidence = min(confidence, 90)
    if len(active) < len(lenses):
        confidence = min(confidence, 85)
    if positive and negative:
        confidence = min(confidence, 72)
    if any(x.data_quality < .90 for x in active):
        confidence = min(confidence, 88)
    if any(x.name == "옵션" and x.data_quality < .60 for x in active):
        confidence = min(confidence, 80)
    quant = lenses.get("quant")
    overall = lenses.get("overall")
    options = lenses.get("options")
    if positive and negative:
        pattern = "Quality vs Timing" if quant and quant.direction > 0 and ((overall and overall.direction < 1) or (options and options.available and options.direction < 1)) else "Mixed / Divergence"
        headline = "Divergence Detected"
    elif len(improving) >= 2 and improving_count >= 2:
        pattern, headline = "Strengthening", f"{positive} / {len(active)} Positive"
    elif positive >= max(2, len(active) - 1):
        pattern, headline = "Broad Strength", f"{positive} / {len(active)} Positive"
    elif negative >= max(2, len(active) - 1):
        pattern, headline = "Broad Weakness", f"{positive} / {len(active)} Positive"
    elif improving and weakening_count >= 2:
        pattern, headline = "Momentum Fading", "Mixed"
    else:
        pattern, headline = "Mixed / Divergence", "Mixed"
    text = detailed_interpretation(lenses, pattern)
    return Consensus(headline, pattern, max(0, min(100, confidence)), text, len(active), positive)

