"""姿态数据导出 (CLAUDE.md §1.5.1: 数据优先, CSV 才是评估依据).

把 engine 采集的 Trace (StepSample 序列) 写成 dump_pose_data 同款的 26 列 CSV,
数值保留两位小数 (用户约定). 输出到 logs/pose_data/ (git 忽略).
"""

from __future__ import annotations

import csv
from pathlib import Path

from _devlog.assess.metrics import Trace

ROOT = Path(__file__).resolve().parent.parent.parent
POSE_OUT_DIR = ROOT / "logs" / "pose_data"

# 与 dump_pose_data.py 一致 (§1.5.1)
COLUMNS = [
    "step",
    "time_s",
    "L_hip_roll_rad",
    "L_hip_pitch_rad",
    "L_knee_rad",
    "R_hip_roll_rad",
    "R_hip_pitch_rad",
    "R_knee_rad",
    "base_roll_rad",
    "base_pitch_rad",
    "base_yaw_rad",
    "base_x",
    "base_y",
    "base_z",
    "linvel_x",
    "linvel_y",
    "linvel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "wheel_L_rads",
    "wheel_R_rads",
    "up_z",
    "wheel_contact_L",
    "wheel_contact_R",
    "recover_completed",
]

_COL_IDX = {name: i for i, name in enumerate(COLUMNS)}


def write_pose_csv(trace: Trace, out_path: str | Path) -> Path:
    """把一条 Trace 写成姿态数据 CSV (26 列, 两位小数)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(s.to_row(_COL_IDX) for s in trace)
    return out


def export_traces(traces: list[Trace], prefix: str, suffix: str = "") -> list[Path]:
    """把多条 Trace 导出为 logs/pose_data/<prefix>_<i>_<suffix>.csv."""
    POSE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, trace in enumerate(traces):
        name = f"{prefix}_{i}{('_' + suffix) if suffix else ''}.csv"
        paths.append(write_pose_csv(trace, POSE_OUT_DIR / name))
    return paths
