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
    missing = [x.name for x in lenses.values() if not x.available]
    if pattern == "Strengthening":
        text = "종합·퀀트·시장환경의 최근 점수 흐름이 함께 개선되어 신호 강도가 증가하는 국면입니다."
    elif pattern == "Broad Strength":
        text = "대부분의 분석 관점이 긍정적으로 일치합니다. 각 분석의 무효화 조건도 함께 확인하세요."
    elif pattern == "Broad Weakness":
        text = "대부분의 분석 관점이 부정적이므로 방어적인 비중 관리가 우선입니다."
    elif pattern == "Quality vs Timing":
        text = "기업 품질은 우호적이지만 현물 타이밍 또는 옵션 확인이 약해 좋은 기업과 좋은 진입시점이 분리되어 있습니다."
    else:
        text = "분석 방향이 엇갈립니다. 단일 점수보다 지지·저항과 확인 조건을 우선해 해석하세요."
    if missing:
        text += f" {', '.join(missing)} 데이터는 N/A로 제외했으며 합의 비율의 분모에 넣지 않았습니다."
    return Consensus(headline, pattern, max(0, min(100, confidence)), text, len(active), positive)
