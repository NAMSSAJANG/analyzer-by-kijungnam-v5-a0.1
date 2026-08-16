from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True)
class ScoreTrend:
    values: tuple[float, ...]
    change: float | None
    label: str
    dates: tuple[str, ...] = ()


class JsonScoreHistory:
    """Small replaceable persistence adapter; an external DB can implement the same API."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> dict:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def record(self, symbol: str, scores: Mapping[str, float], as_of: date | None = None, metadata: Mapping | None = None) -> dict:
        data = self.load()
        rows = data.setdefault(symbol, [])
        row = {"date": (as_of or date.today()).isoformat(), **{k: round(float(v), 1) for k, v in scores.items()}, **dict(metadata or {})}
        rows[:] = [item for item in rows if item.get("date") != row["date"]]
        rows.append(row)
        rows.sort(key=lambda item: item.get("date", ""))
        data[symbol] = rows[-260:]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
        return data

    def recent(self, symbol: str, key: str, count: int = 5) -> ScoreTrend:
        rows = self.load().get(symbol, [])
        pairs = [(str(row.get("date", "")), float(row[key])) for row in rows if row.get(key) is not None][-count:]
        dates = tuple(item[0] for item in pairs)
        values = tuple(item[1] for item in pairs)
        change = round(values[-1] - values[0], 1) if len(values) > 1 else None
        label = "Improving" if change is not None and change >= 2 else "Weakening" if change is not None and change <= -2 else "Stable"
        return ScoreTrend(values, change, label, dates)

    def dates(self, symbol: str) -> set[str]:
        return {str(row.get("date")) for row in self.load().get(symbol, []) if row.get("date")}

    def recent_rows(self, symbol: str, count: int = 5) -> list[dict]:
        return list(self.load().get(symbol, []))[-count:]

    def retain_valid_dates(self, symbol: str, valid_dates: set[str]) -> dict:
        """Remove legacy rows saved on dates absent from the symbol's trading calendar."""
        data = self.load()
        if symbol not in data:
            return data
        filtered = [row for row in data[symbol] if str(row.get("date")) in valid_dates]
        if filtered == data[symbol]:
            return data
        data[symbol] = filtered
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass
        return data


def format_trend(trend: ScoreTrend) -> str:
    sequence = " → ".join(f"{v:.0f}" for v in trend.values) or "N/A"
    delta = "N/A" if trend.change is None else f"{trend.change:+.1f}"
    icon = {"Improving": "🟢", "Weakening": "🔴", "Stable": "🟡"}[trend.label]
    return f"{sequence}  \n5D Change: **{delta}** · Trend: {icon} **{trend.label}**"
