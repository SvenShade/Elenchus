from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_mcts.cognitive import LiftedActionCandidate


@dataclass
class RewardStats:
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def update(self, value: float, weight: float = 1.0) -> None:
        self.visits += 1
        self.value_sum += value * weight


@dataclass
class SearchStatsStore:
    stats: dict[str, RewardStats] = field(default_factory=dict)

    def backup(self, candidate: LiftedActionCandidate | None, utility: float) -> None:
        if candidate is None:
            return
        for key, weight in abstraction_keys(candidate):
            self.stats.setdefault(key, RewardStats()).update(utility, weight=weight)

    def value_hint(self, candidate: LiftedActionCandidate | None) -> float:
        if candidate is None:
            return 0.0
        weighted_sum = 0.0
        weight_sum = 0.0
        for key, weight in abstraction_keys(candidate):
            stat = self.stats.get(key)
            if stat is None or stat.visits == 0:
                continue
            weighted_sum += stat.mean * weight
            weight_sum += weight
        if weight_sum <= 0:
            return 0.0
        return weighted_sum / weight_sum

    def compact(self, limit: int = 80) -> dict[str, Any]:
        items = sorted(
            self.stats.items(),
            key=lambda item: (-item[1].visits, item[0]),
        )[:limit]
        return {
            key: {
                "visits": stat.visits,
                "value": stat.mean,
                "value_sum": stat.value_sum,
            }
            for key, stat in items
        }


def abstraction_keys(candidate: LiftedActionCandidate) -> list[tuple[str, float]]:
    metadata = candidate.action_metadata or {}
    keys: list[tuple[str, float]] = [
        (f"candidate:{candidate.candidate_id}", 1.0),
        (f"base_action:{candidate.base_action_id}", 0.7),
    ]
    for value in _values(candidate.diagnostic_operator):
        keys.append((f"diagnostic_operator:{value}", 0.7))
    for value in _values(candidate.repair_operator):
        keys.append((f"repair_operator:{value}", 0.7))
    for value in _values(candidate.dialogue_form):
        keys.append((f"dialogue_form:{value}", 0.7))
    for value in _values(metadata.get("failure_family_ids")):
        keys.append((f"failure_family:{value}", 0.45))
    for value in _values(metadata.get("target_types") or candidate.target_ids):
        keys.append((f"target_type:{value}", 0.45))
    for value in _values(metadata.get("codex_quadrant")):
        keys.append((f"codex_quadrant:{value}", 0.45))
    for value in _values(_directness_bin(candidate.directness)):
        keys.append((f"directness_bin:{value}", 0.45))
    for value in _values(metadata.get("user_particle_cluster")):
        keys.append((f"user_particle_cluster:{value}", 0.35))
    return keys


def _values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _directness_bin(value: float) -> str:
    try:
        directness = float(value)
    except (TypeError, ValueError):
        directness = 0.5
    directness = max(0.0, min(1.0, directness))
    bucket = int(directness * 5)
    if bucket >= 5:
        bucket = 4
    return f"{bucket / 5:.1f}-{(bucket + 1) / 5:.1f}"
