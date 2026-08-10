"""点足平地行走评估 (CLAUDE.md §7.3).

抬腿平缓 (leg_jerk) · 机身高度≈0.52±0.05 · 追踪 + 微动平衡
"""

from __future__ import annotations

from _devlog.assess.eval._walk import evaluate_walk


def evaluate(env, policy, args) -> dict[str, float]:
    """点足行走评估 (含机身高度 + 抬腿平缓度)."""
    return evaluate_walk(env, policy, args, height_err=True, jerk=True)
