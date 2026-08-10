"""平地滚动行走评估 (CLAUDE.md §7.2).

Vx/Vy 追踪误差<0.1 · 存活≥95% · 侧移 Vy>0.25 · 无指令微动平衡 (linvel_xy<0.2, gyro<1)
"""

from __future__ import annotations

from _devlog.assess.eval._walk import evaluate_walk


def evaluate(env, policy, args) -> dict[str, float]:
    """标准行走评估 (追踪 + 微动平衡)."""
    return evaluate_walk(env, policy, args)
