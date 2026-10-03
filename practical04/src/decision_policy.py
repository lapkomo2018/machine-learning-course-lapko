"""Політика перетворення прогнозного бала ризику на операційну дію (ПР4)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ACTIONS = ("standard", "manual_review", "priority_assistance")


@dataclass(frozen=True)
class DecisionPolicy:
    """Політика з двома порогами (правило ``score >= threshold``).

    score <  threshold_low                     -> "standard"
    threshold_low <= score < threshold_high    -> "manual_review"
    score >= threshold_high                    -> "priority_assistance"

    Одно-порогова політика — окремий випадок threshold_low == threshold_high.
    """

    threshold_low: float
    threshold_high: float
    actions: tuple[str, str, str] = ACTIONS

    def __post_init__(self) -> None:
        for name in ("threshold_low", "threshold_high"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.threshold_low > self.threshold_high:
            raise ValueError("threshold_low must not exceed threshold_high")
        if len(self.actions) != 3 or len(set(self.actions)) != 3:
            raise ValueError("exactly three distinct actions are required")

    def decide(self, score: float) -> str:
        if not 0.0 <= score <= 1.0 or np.isnan(score):
            raise ValueError(f"score must be in [0, 1], got {score}")
        if score >= self.threshold_high:
            return self.actions[2]
        if score >= self.threshold_low:
            return self.actions[1]
        return self.actions[0]

    def decide_batch(self, scores: Iterable[float]) -> list[str]:
        """Рішення для пакета; порядок результатів збігається з порядком балів."""
        return [self.decide(float(s)) for s in scores]

    @classmethod
    def from_config(cls, path: str | Path) -> "DecisionPolicy":
        config = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            threshold_low=config["thresholds"]["low"],
            threshold_high=config["thresholds"]["high"],
            actions=tuple(config["actions"]),
        )


def top_k_decisions(scores: Sequence[float], ids: Sequence[object], k: int) -> list[bool]:
    """Жорстка квота: позначити рівно min(k, n) об'єктів із найбільшим балом.

    Ties розв'язуються відтворювано: за спаданням бала, потім за зростанням ідентифікатора.
    Повертає список прапорців у вихідному порядку об'єктів.
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if len(scores) != len(ids):
        raise ValueError("scores and ids must have equal length")
    if len(set(ids)) != len(ids):
        raise ValueError("ids must be unique")
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), ids[i]))
    chosen = set(order[: min(k, len(scores))])
    return [i in chosen for i in range(len(scores))]
