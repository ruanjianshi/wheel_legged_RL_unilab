#!/usr/bin/env python3
"""从姿态数据 CSV 反推机器人姿态 (规范 CLAUDE.md §1.3 / §1.5.2)。

数据优先: 姿态数据 CSV (dump_pose_data.py 产出, logs/pose_data/) 才是评估依据,
视频/图只是给负责人看的展示。本工具把 CSV 里的 6 关节角度 + base 姿态数据,
按 §1.3 "关节角度 → 姿态映射表" 逐行判定姿态类别, 并统计各姿态时长/占比。

用法:
  uv run tools/xqrobotwl/infer_pose_from_csv.py \
      logs/pose_data/<prefix>_model_XXX.pt_<姿态>.csv \
      [--out <带姿态列的逐行结果.csv>] [--json <统计摘要.json>]

输出:
  stdout 摘要: 各姿态帧数/时长(s)/占比(%), 站立期微动指标 (mean|linvel_xy|, mean|gyro|,
              yaw 累计, 轮子离地率), 恢复指标 (recover_completed)
  --out : 原 CSV 每行追加 pose 列
  --json: 机器可读统计摘要

分类优先级 (与 §1.5.2 示例链一致, 并扩充 §1.3 反推表):
  倒地 > 左右腿一前一后 > 髋外展/内收 > 左右高低腿 > 下蹲 > 伸腿/过高
  > 前倾/后倾 > 左右倾斜 > 站立(可被 摇摆/轮子点地/转圈 覆盖) > 过渡
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# dump_pose_data.py 的 COLUMNS (规范 §1.5.1), 缺少任一列则报错
REQUIRED_COLUMNS = [
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

# §1.3 自然站姿 (walk 实测 standing_angles), 仅作展示参考; 判定主要靠 up_z/base_z
STANDING_ANGLES = {
    "L_hip_roll": 0.102,
    "L_hip_pitch": 0.083,
    "L_knee": -0.079,
    "R_hip_roll": 0.013,
    "R_hip_pitch": -0.108,
    "R_knee": 0.019,
}

# §1.3 判定阈值
Z_LYING = 0.25  # base_z < 0.25 → 倒地
Z_TOO_HIGH = 0.60  # base_z > 0.60 → 伸腿/过高
Z_CROUCH_LOW = 0.45  # 下蹲时的 base_z 上界
PITCH_ASYMM = 0.30  # |L_pitch - R_pitch| > 0.30 → 左右腿一前一后
HIP_ABDUCT = 0.50  # L_hip_roll > 0.50 或 R_hip_roll < -0.50 → 髋外展/内收
KNEE_ASYMM = 0.30  # |L_knee - R_knee| > 0.30 → 左右高低腿
KNEE_DEEP = 0.50  # 双膝 |knee| > 0.50 → 深屈 (下蹲)
TILT_ABNORM = 0.20  # |base_roll|/|base_pitch| > 0.20 → 左右倾斜/前后倾
UP_STAND = 0.85  # up_z > 0.85 → 直立
GYRO_STAND = 1.0  # 站立期 |gyro| > 1 rad/s → 摇摆
YAW_SPIN = 1.0  # 站立期 yaw 累计漂移 > 1 rad (~57°) → 转圈
WHEEL_DIFF_SPIN = 0.5  # 转圈判定的左右轮速差下限 rad/s


def _f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _disp_width(s: str) -> int:
    """终端显示宽度: 东亚宽字符 (中文等) 占 2 列, 用于对齐表格。"""
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in s)


def _pad(s: str, width: int) -> str:
    return s + " " * max(0, width - _disp_width(s))


def _unwrap(vals: list[float]) -> list[float]:
    """把角度序列展开成无跳变累加值 (用于计算 yaw 累计旋转)。"""
    out: list[float] = []
    prev = 0.0
    for v in vals:
        if out:
            d = v - prev
            d = (d + math.pi) % (2.0 * math.pi) - math.pi
            out.append(out[-1] + d)
        else:
            out.append(v)
        prev = v
    return out


def _classify_row(r: dict[str, str]) -> str:
    """按 §1.3 反推表 + §1.5.2 示例链, 对单行判定姿态 (优先级从上到下)。"""
    z = _f(r, "base_z")
    up = _f(r, "up_z")
    lp = _f(r, "L_hip_pitch_rad")
    rp = _f(r, "R_hip_pitch_rad")
    lr = _f(r, "L_hip_roll_rad")
    rr = _f(r, "R_hip_roll_rad")
    lk = _f(r, "L_knee_rad")
    rk = _f(r, "R_knee_rad")
    roll = _f(r, "base_roll_rad")
    pitch = _f(r, "base_pitch_rad")
    gyro = (_f(r, "gyro_x"), _f(r, "gyro_y"), _f(r, "gyro_z"))
    gyro_mag = math.sqrt(gyro[0] ** 2 + gyro[1] ** 2 + gyro[2] ** 2)

    if z < Z_LYING:
        return "倒地"
    if abs(lp - rp) > PITCH_ASYMM:
        return "左右腿一前一后"
    if lr > HIP_ABDUCT or rr < -HIP_ABDUCT:
        return "髋外展/内收"
    if abs(lk - rk) > KNEE_ASYMM and abs(lp - rp) > PITCH_ASYMM / 2:
        return "左右高低腿"
    if abs(lk) > KNEE_DEEP and abs(rk) > KNEE_DEEP and z < Z_CROUCH_LOW:
        return "下蹲"
    if z > Z_TOO_HIGH:
        return "伸腿/过高"
    if pitch > TILT_ABNORM:
        return "后倾"
    if pitch < -TILT_ABNORM:
        return "前倾"
    if abs(roll) > TILT_ABNORM:
        return "左右倾斜"
    if Z_LYING <= z <= Z_TOO_HIGH and up > UP_STAND:
        # 站立候选, 先看是否满足"微动平衡"异常条件
        if gyro_mag > GYRO_STAND:
            return "摇摆"
        if _f(r, "wheel_contact_L") == 0 or _f(r, "wheel_contact_R") == 0:
            return "轮子点地"
        return "站立"
    return "过渡"


def _summary_stats(rows: list[dict[str, str]], poses: list[str]) -> dict[str, object]:
    """计算各姿态时长/占比 + 站立期微动指标 + 恢复指标。"""
    dt_vals = []
    for i in range(1, len(rows)):
        dt_vals.append(_f(rows[i], "time_s") - _f(rows[i - 1], "time_s"))
    mean_dt = sum(dt_vals) / len(dt_vals) if dt_vals else (0.01 if rows else 0.0)
    n = len(rows)
    counter = Counter(poses)
    total_s = n * mean_dt

    # 站立行 (含 摇摆/轮子点地/转圈 覆盖前的原始 站立) 的微动指标
    standing = [(r, p) for r, p in zip(rows, poses) if p in ("站立", "摇摆", "轮子点地", "转圈")]
    linvel_xy = [math.hypot(_f(r, "linvel_x"), _f(r, "linvel_y")) for r, _ in standing]
    gyro = [
        math.sqrt(_f(r, "gyro_x") ** 2 + _f(r, "gyro_y") ** 2 + _f(r, "gyro_z") ** 2)
        for r, _ in standing
    ]
    wheel_off = sum(
        1 for r, _ in standing if _f(r, "wheel_contact_L") == 0 or _f(r, "wheel_contact_R") == 0
    )
    yaw = [_f(r, "base_yaw_rad") for r, _ in standing]
    yaw_unwrapped = _unwrap(yaw) if yaw else []
    yaw_drift = (yaw_unwrapped[-1] - yaw_unwrapped[0]) if len(yaw_unwrapped) > 1 else 0.0

    # 恢复指标 (fall_recovery 任务)
    recover_val = _f(rows[-1], "recover_completed") if rows else 0.0
    first_recover = next((i for i, r in enumerate(rows) if _f(r, "recover_completed") == 1), None)
    base_z = [_f(r, "base_z") for r in rows]

    return {
        "rows": n,
        "duration_s": round(total_s, 3),
        "mean_dt_s": round(mean_dt, 4),
        "pose_stats": {
            pose: {
                "frames": cnt,
                "duration_s": round(cnt * mean_dt, 3),
                "ratio": round(cnt / n, 4) if n else 0.0,
            }
            for pose, cnt in sorted(counter.items())
        },
        "standing": {
            "mean_linvel_xy": round(sum(linvel_xy) / len(linvel_xy), 4) if linvel_xy else None,
            "mean_gyro": round(sum(gyro) / len(gyro), 4) if gyro else None,
            "yaw_drift_rad": round(yaw_drift, 4),
            "wheel_off_rate": round(wheel_off / len(standing), 4) if standing else None,
            "frames": len(standing),
        },
        "recovery": {
            "recover_completed": recover_val,
            "first_recover_step": first_recover,
        },
        "base_z": {
            "min": round(min(base_z), 4) if base_z else None,
            "max": round(max(base_z), 4) if base_z else None,
        },
    }


def _print_summary(summary: dict[str, object]) -> None:
    pose_stats = summary["pose_stats"]  # type: ignore[assignment]
    print("== 姿态分布 ==")
    for pose, st in pose_stats.items():
        print(
            f"  {_pad(pose, 10)} {st['frames']:>5} 帧  {st['duration_s']:>7.3f}s  "
            f"{st['ratio'] * 100:>5.1f}%"
        )
    print("== 站立期微动指标 ==")
    st = summary["standing"]
    print(f"  帧数            {st['frames']}")
    print(f"  mean |linvel_xy|  {st['mean_linvel_xy']} m/s   (<0.2 达标)")
    print(f"  mean |gyro|       {st['mean_gyro']} rad/s   (<1 达标)")
    print(f"  yaw 累计漂移     {st['yaw_drift_rad']} rad   (<~1 达标)")
    print(f"  轮子离地率       {st['wheel_off_rate']}   (=0 达标)")
    print("== 恢复指标 ==")
    rc = summary["recovery"]
    print(
        f"  recover_completed  {rc['recover_completed']}  "
        f"(首次恢复 step={rc['first_recover_step']})"
    )
    print("== base_z ==")
    bz = summary["base_z"]
    print(f"  min {bz['min']} / max {bz['max']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="从姿态数据 CSV 反推姿态并统计 (CLAUDE.md §1.3/§1.5.2)"
    )
    parser.add_argument("csv_path", type=Path, help="dump_pose_data 产出的姿态 CSV")
    parser.add_argument("--out", type=Path, default=None, help="输出逐行结果 CSV (追加 pose 列)")
    parser.add_argument(
        "--json", dest="json_path", type=Path, default=None, help="输出统计摘要 JSON"
    )
    args = parser.parse_args(argv)

    if not args.csv_path.exists():
        print(f"[error] 找不到姿态 CSV: {args.csv_path}", file=sys.stderr)
        return 1

    with args.csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            print(
                f"[error] CSV 缺少规范 §1.5.1 要求的列: {missing}",
                file=sys.stderr,
            )
            return 1
        rows = list(reader)

    if not rows:
        print("[error] CSV 无数据行", file=sys.stderr)
        return 1

    poses = [_classify_row(r) for r in rows]

    # 转圈: 按连续"站立"段检测, 段内 yaw 累计漂移 > 阈值 且 左右轮速差大 → 该段改判 转圈
    i = 0
    while i < len(rows):
        if poses[i] != "站立":
            i += 1
            continue
        j = i
        while j + 1 < len(rows) and poses[j + 1] == "站立":
            j += 1
        if j > i:
            yaw_vals = [_f(rows[k], "base_yaw_rad") for k in range(i, j + 1)]
            yaw_unwrapped = _unwrap(yaw_vals)
            yaw_drift = yaw_unwrapped[-1] - yaw_unwrapped[0]
            wheel_diff = [
                abs(_f(rows[k], "wheel_L_rads") - _f(rows[k], "wheel_R_rads"))
                for k in range(i, j + 1)
            ]
            mean_wheel_diff = sum(wheel_diff) / len(wheel_diff)
            if abs(yaw_drift) > YAW_SPIN and mean_wheel_diff > WHEEL_DIFF_SPIN:
                for k in range(i, j + 1):
                    poses[k] = "转圈"
        i = j + 1

    summary = _summary_stats(rows, poses)
    _print_summary(summary)

    if args.json_path:
        args.json_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.out:
        with args.out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=[*REQUIRED_COLUMNS, "pose"])
            writer.writeheader()
            for r, p in zip(rows, poses):
                writer.writerow({**r, "pose": p})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
