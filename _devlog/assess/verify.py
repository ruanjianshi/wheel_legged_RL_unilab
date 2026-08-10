"""§7.x 达标判定 (CLAUDE.md §7.2-7.9 + 附录 A).

把 eval 模块产出的指标字典对照 TaskDef.thresholds 逐项判定 ✅/❌。
指标缺失或 NaN → 记为未达标 (False), 提示补测。
"""

from __future__ import annotations

import math

from _devlog.assess.tasks import TaskDef, Threshold


def _compare(value: float, thr: Threshold) -> bool:
    if thr.op == "<":
        return value < thr.value
    if thr.op == "<=":
        return value <= thr.value
    if thr.op == ">":
        return value > thr.value
    if thr.op == ">=":
        return value >= thr.value
    if thr.op == "==":
        return abs(value - thr.value) < 1e-9
    if thr.op == "≈":
        return abs(value - thr.value) < abs(thr.value) * 0.1
    raise ValueError(f"未知判定符: {thr.op}")


def check(task: TaskDef, metrics: dict[str, float]) -> dict[str, dict]:
    """逐项判定, 返回 {metric: {value, threshold, passed, ok}}."""
    verdicts: dict[str, dict] = {}
    for key, thr in task.thresholds.items():
        if key not in metrics:
            verdicts[key] = {"value": None, "threshold": thr, "passed": False, "ok": False}
            continue
        value = metrics[key]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            verdicts[key] = {"value": None, "threshold": thr, "passed": False, "ok": False}
            continue
        verdicts[key] = {
            "value": value,
            "threshold": thr,
            "passed": _compare(value, thr),
            "ok": True,
        }
    return verdicts


def overall(verdicts: dict[str, dict]) -> tuple[bool, int, int]:
    """总体判定: (是否全过, 通过数, 总项数)."""
    total = len(verdicts)
    passed = sum(1 for v in verdicts.values() if v["passed"])
    return passed == total, passed, total
