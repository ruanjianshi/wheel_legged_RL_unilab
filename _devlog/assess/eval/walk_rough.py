"""不平坦地形行走评估 (CLAUDE.md §7.4).

存活≥90% · 机身高度波动小 (base_height_std) · 微动平衡
"""

from __future__ import annotations

from _devlog.assess.eval._walk import evaluate_walk


def evaluate(env, policy, args) -> dict[str, float]:
    """粗糙地形行走评估 (含机身高度波动)."""
    return evaluate_walk(env, policy, args, rough_std=True)
