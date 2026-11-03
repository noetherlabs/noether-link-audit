"""Counterfactual data accounting and Shapley attribution of lost data to causes.

A pass is a sample series of received fiber power and lock state. Removing a
cause adds its dB correction to the power series and/or restores lock on the
samples it blocked. Data counts samples that are locked and above sensitivity.
Shapley values split interacting causes fairly; they sum to the data recovered
when every cause is removed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


@dataclass
class LossModel:
    rate_gbps: float
    sample_s: float
    sensitivity_dbm: float
    base_dbm: Float32Array
    locked: BoolArray
    additive: dict[str, Float32Array] = field(default_factory=dict)
    groups: list[dict[frozenset[str], Float32Array]] = field(default_factory=list)
    restore: dict[str, BoolArray] = field(default_factory=dict)

    def possible_gb(self) -> float:
        return len(self.base_dbm) * self.sample_s * self.rate_gbps

    def active_causes(self, candidates: tuple[str, ...]) -> list[str]:
        active = []
        for cause in candidates:
            effect = False
            if cause in self.additive and np.any(np.abs(self.additive[cause]) > 1e-6):
                effect = True
            if cause in self.restore and np.any(self.restore[cause] & ~self.locked):
                effect = True
            for group in self.groups:
                for key, values in group.items():
                    if cause in key and np.any(np.abs(values) > 1e-6):
                        effect = True
            if effect:
                active.append(cause)
        return active

    def delivered_gb(
        self, removed: frozenset[str] = frozenset(), extra_db: Float32Array | None = None
    ) -> float:
        power = self.base_dbm.astype(np.float32, copy=True)
        for cause in removed:
            if cause in self.additive:
                power += self.additive[cause]
        for group in self.groups:
            members = frozenset().union(*group.keys())
            key = frozenset(removed & members)
            if key in group:
                power += group[key]
        if extra_db is not None:
            power += extra_db
        locked = self.locked.copy()
        for cause in removed:
            if cause in self.restore:
                locked |= self.restore[cause]
        good = np.count_nonzero(locked & (power >= np.float32(self.sensitivity_dbm)))
        return float(good) * self.sample_s * self.rate_gbps


@dataclass(frozen=True)
class Attribution:
    possible_gb: float
    delivered_gb: float
    all_removed_gb: float
    shapley_gb: dict[str, float]

    @property
    def link_budget_residual_gb(self) -> float:
        return self.possible_gb - self.all_removed_gb


def shapley(model: LossModel, candidates: tuple[str, ...]) -> Attribution:
    causes = model.active_causes(candidates)
    n = len(causes)
    values = np.zeros(1 << n)
    for mask in range(1 << n):
        removed = frozenset(c for i, c in enumerate(causes) if mask >> i & 1)
        values[mask] = model.delivered_gb(removed)
    weights = [math.factorial(s) * math.factorial(n - s - 1) / math.factorial(n) for s in range(n)]
    phi = {c: 0.0 for c in candidates}
    for i, cause in enumerate(causes):
        total = 0.0
        for mask in range(1 << n):
            if mask >> i & 1:
                continue
            total += weights[bin(mask).count("1")] * (values[mask | 1 << i] - values[mask])
        phi[cause] = float(total)
    return Attribution(
        possible_gb=model.possible_gb(),
        delivered_gb=float(values[0]),
        all_removed_gb=float(values[(1 << n) - 1]),
        shapley_gb=phi,
    )
