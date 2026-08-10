"""姿态反推统计 (CLAUDE.md §1.3 反推表 + §1.5.2).

把评估中采集的 Trace 逐行按 §1.3 "关节角度 → 姿态映射表" 判姿态,
统计各姿态出现帧数/时长/占比。判定逻辑复用 tools/xqrobotwl/infer_pose_from_csv
(同一套阈值, 保证与独立 CSV 反推工具结果一致)。
"""

from __future__ import annotations

from collections import Counter

from _devlog.assess.metrics import Trace
from tools.xqrobotwl.infer_pose_from_csv import _classify_row  # noqa: F401  (复用阈值+判定)


def _sample_row(sample) -> dict[str, str]:
    """把 StepSample 转成 infer_pose_from_csv 期望的 str 值字典."""
    dp, eu, gy, wv, wc = (
        sample.dof_pos,
        sample.euler,
        sample.gyro,
        sample.wheel_vel,
        sample.wheel_contact,
    )
    return {
        "base_z": str(sample.base_pos[2]),
        "up_z": str(sample.up_z),
        "L_hip_roll_rad": str(dp[0]),
        "L_hip_pitch_rad": str(dp[1]),
        "L_knee_rad": str(dp[2]),
        "R_hip_roll_rad": str(dp[3]),
        "R_hip_pitch_rad": str(dp[4]),
        "R_knee_rad": str(dp[5]),
        "base_roll_rad": str(eu[0]),
        "base_pitch_rad": str(eu[1]),
        "base_yaw_rad": str(eu[2]),
        "gyro_x": str(gy[0]),
        "gyro_y": str(gy[1]),
        "gyro_z": str(gy[2]),
        "wheel_L_rads": str(wv[0]),
        "wheel_R_rads": str(wv[1]),
        "wheel_contact_L": str(int(wc[0])),
        "wheel_contact_R": str(int(wc[1])),
    }


def classify_trace(trace: Trace) -> list[str]:
    """逐行判定姿态类别."""
    return [_classify_row(_sample_row(s)) for s in trace]


def pose_stats(traces: list[Trace], ctrl_dt: float = 0.01) -> dict[str, dict]:
    """各姿态帧数/时长(s)/占比. 汇总所有 trace."""
    counter: Counter[str] = Counter()
    n = 0
    for trace in traces:
        for label in classify_trace(trace):
            counter[label] += 1
            n += 1
    return {
        pose: {
            "frames": cnt,
            "duration_s": round(cnt * ctrl_dt, 3),
            "ratio": round(cnt / n, 4) if n else 0.0,
        }
        for pose, cnt in sorted(counter.items())
    }


def print_pose_stats(stats: dict[str, dict]) -> None:
    print("== 姿态分布 (§1.3 反推) ==")
    if not stats:
        print("  (无数据)")
        return
    for pose, st in stats.items():
        print(
            f"  {pose:<10} {st['frames']:>5} 帧  {st['duration_s']:>7.3f}s  {st['ratio'] * 100:>5.1f}%"
        )
