#!/usr/bin/env python3
"""详细跳跃评估: 三段输出 (跳跃前姿态 / 键盘触发跳跃过程 / 落地后姿态)。

按 CLAUDE.md §7.5 + §1.3 姿态表格, 逐段输出时间序列姿态:
  段1 跳跃前: settle 期站立姿态 (base_z/关节/up_z/gyro/接触), 稳定性 + 对称性 + 漂移
  段2 跳跃过程: 触发ON 期 下蹲→起跳→空中→落地 全序列关键姿态 (每段采样打印)
  段3 落地后: 恢复期姿态轨迹, 恢复用时, 稳定度, 漂移, 最终站姿

Usage:
    uv run tools/xqrobotwl/eval_jump_detail.py --task XqRobotWLJumpSRLFlat \
        --checkpoint <model.pt> [--settle 100] [--pulse 160] [--tail 200]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.xqrobotwl.verify_jump import load_actor, trained_env_overrides  # noqa: E402

STANDING_ANGLES = np.array([0.102, 0.083, -0.079, 0.013, -0.108, 0.019])
JOINTS = ("L_hip_roll", "L_hip_pitch", "L_knee", "R_hip_roll", "R_hip_pitch", "R_knee")


def _euler(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def _posture_tags(z, up, j, roll, pitch, gyro_mag):
    """按 §1.3 返回姿态标签列表."""
    tags = []
    if z < 0.25:
        tags.append("倒地")
    elif up > 0.85 and abs(z - 0.52) < 0.12:
        tags.append("正常站立")
    elif z < 0.52 - 0.07:
        tags.append("下蹲")
    elif z > 0.52 + 0.12:
        tags.append("伸腿/过高")
    if abs(j[1] - j[4]) > 0.3:
        tags.append("左右腿一前一后")
    if abs(j[2] - j[5]) > 0.4:
        tags.append("左右高低腿")
    if j[0] > 0.5 or j[3] < -0.5:
        tags.append("髋外展")
    if abs(pitch) > 0.2:
        tags.append("前倾" if pitch > 0 else "后倾")
    if abs(roll) > 0.2:
        tags.append("左右倾斜")
    if gyro_mag > 1.0:
        tags.append("摇摆")
    return tags


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--settle", type=int, default=100, help="跳跃前站立步数")
    p.add_argument("--pulse", type=int, default=160, help="键盘触发跳跃 ON 步数")
    p.add_argument("--tail", type=int, default=200, help="落地后恢复步数")
    p.add_argument("--stand", type=float, default=0.52)
    p.add_argument("--hidden", default="512,512,256,128")
    args = p.parse_args()
    hidden = [int(x) for x in args.hidden.split(",")]

    from unilab.base import registry
    from unilab.training import ensure_registries

    ensure_registries()
    ov = trained_env_overrides(args.checkpoint)
    env = registry.make(args.task, num_envs=1, sim_backend="mujoco", env_cfg_override=ov)
    try:
        actor = load_actor(args.checkpoint, env.obs_groups_spec["obs"], 8, hidden)
        env.init_state()
        total = args.settle + args.pulse + args.tail
        recs = []  # (step, trigger, z, up, joints[6], roll, pitch, yaw, |gyro|, contact, linv_x, linv_y)
        with torch.no_grad():
            for step in range(total):
                trig = 1.0 if args.settle <= step < args.settle + args.pulse else 0.0
                env.state.info["commands"][:, 4] = trig
                obs = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
                action = actor(obs).numpy()
                st = env.step(action)
                dof = env.get_dof_pos()[0]
                bp = np.asarray(env._backend.get_base_pos())[0]
                quat = env._backend.get_base_quat()[0]
                roll, pitch, yaw = _euler(quat)
                lv = env.get_local_linvel()[0]
                gyro = np.asarray(env._backend.get_sensor_data("gyro")).reshape(-1, 3)[0]
                up_z = float(np.asarray(env._backend.get_sensor_data("upvector")).reshape(-1, 3)[0, 2])
                lz = float(np.asarray(env._backend.get_sensor_data("left_wheel_world_pos")).reshape(-1, 3)[0, 2])
                contact = 1.0 if lz < 0.13 else 0.0
                recs.append((step, trig, float(bp[2]), up_z, dof[:6].copy(), roll, pitch, yaw,
                             float(np.linalg.norm(gyro)), contact, float(lv[0]), float(lv[1])))
                if st.terminated[0]:
                    break

        n = len(recs)
        print("=" * 74)
        print(f"详细跳跃评估 {args.task}")
        print(f"checkpoint: {args.checkpoint}")
        print("=" * 74)

        # ── 段1: 跳跃前站立姿态 ─────────────────────────────────────
        pre = [r for r in recs[:args.settle] if r[9] > 0.5]  # 轮着地
        if pre:
            zs = [r[2] for r in pre]
            ups = [r[3] for r in pre]
            gyros = [r[8] for r in pre]
            j = np.mean([r[4] for r in pre], axis=0)
            dx = recs[args.settle - 1][10] if args.settle <= n else 0.0
            print("\n[段1] 跳跃前站立姿态 ({} 步着地采样)".format(len(pre)))
            print(f"  base_z  均值 {np.mean(zs):.3f}  波动 {np.std(zs):.3f}  目标 {args.stand} "
                  f"({'✅' if abs(np.mean(zs)-args.stand)<0.10 else '❌'})")
            print(f"  up_z    均值 {np.mean(ups):.3f}  (直立 {'✅' if np.mean(ups)>0.85 else '❌'})")
            print(f"  |gyro|  均值 {np.mean(gyros):.3f}  (稳定 {'✅' if np.mean(gyros)<1.0 else '❌'})")
            print(f"  关节    平均 {_fmt_joint(j)}")
            print(f"          自然 {_fmt_joint(STANDING_ANGLES)}")
            sym = abs(j[0] + j[3]) < 0.15 and abs(j[1] - j[4]) < 0.3 and abs(j[2] - j[5]) < 0.3
            print(f"  对称    {'✅ 左右对称' if sym else '❌ 不对称 (Δpitch={abs(j[1]-j[4]):.2f} Δknee={abs(j[2]-j[5]):.2f})'}")
            tags = _posture_tags(np.mean(zs), np.mean(ups), j, 0, 0, np.mean(gyros))
            print(f"  姿态    {tags if tags else '正常站立'}")
            # 采样 3 步看站立细节
            print("  采样 (每步):")
            for i in [0, len(pre)//2, -1]:
                r = pre[i]
                t = _posture_tags(r[2], r[3], r[4], r[5], r[6], r[8])
                print(f"    step{r[0]:4d} z={r[2]:.3f} up={r[3]:.3f} |gyro|={r[8]:.2f} "
                      f"膝=[{r[4][2]:+.2f},{r[4][5]:+.2f}] pitch={r[6]:+.2f} 姿态={t if t else '正常'}")
        else:
            print("\n[段1] 跳跃前站立: ❌ 触发前轮未着地")

        # ── 段2: 键盘触发跳跃过程 ───────────────────────────────────
        print("\n[段2] 键盘触发跳跃过程 (触发ON {}/{} 步)".format(args.pulse, args.pulse))
        on = [r for r in recs if r[1] > 0.5]
        if on:
            # 找下蹲最低点 (接地)
            crouch = min((r for r in on if r[9] > 0.5), key=lambda r: r[2], default=None)
            air = [r for r in on if r[9] < 0.5]
            peak = max(on, key=lambda r: r[2])
            air_start = next((r for r in on if r[9] < 0.5), None)
            # 起跳前最后接地步
            lift = None
            if air_start:
                idx = next(i for i, r in enumerate(on) if r is air_start)
                lift = on[idx - 1] if idx > 0 else None
            land = next((r for r in recs if r[1] > 0.5 and r[9] > 0.5), None)
            # 找落地 (腾空后重新接地)
            land_i = None
            seen_air = False
            for i, r in enumerate(recs):
                if r[1] > 0.5 and r[9] < 0.5:
                    seen_air = True
                elif seen_air and r[9] > 0.5:
                    land_i = i
                    break

            def _show(r, label):
                t = _posture_tags(r[2], r[3], r[4], r[5], r[6], r[8])
                print(f"    {label:<14} step{r[0]:4d} z={r[2]:.3f} up={r[3]:.3f} "
                      f"膝=[{r[4][2]:+.2f},{r[4][5]:+.2f}] hip=[{r[4][1]:+.2f},{r[4][4]:+.2f}] "
                      f"roll={r[5]:+.2f} pitch={r[6]:+.2f} 姿态={t if t else '正常'}")

            if crouch:
                _show(crouch, "下蹲最低点")
            if lift:
                _show(lift, "起跳前(接地)")
            if air_start:
                _show(air_start, "腾空开始")
            _show(peak, "腾空峰值")
            if land_i is not None:
                _show(recs[land_i], "落地")
            print(f"  腾空 {len(air)} 步  跳高 {peak[2]-args.stand:.3f} m "
                  f"({'✅>0.20' if peak[2]-args.stand>0.20 else '⚠️<0.20'})")
            print(f"  空中轮速: {max(abs(r[4][2])+abs(r[4][5]) for r in air) if air else 0:.0f} rad/s"
                  f"(关节) — 见逐段")
            # 空中直立 + 空中姿态
            if air:
                a_up = np.mean([r[3] for r in air])
                print(f"  空中 up_z 均值 {a_up:.3f} (直立 {'✅' if a_up>0.85 else '❌'})")
        else:
            print("  ❌ 触发窗口内无动作")

        # ── 段3: 落地后恢复姿态 ─────────────────────────────────────
        post = recs[args.settle + args.pulse:] if n > args.settle + args.pulse else []
        if post:
            # 恢复 = 落地后达到稳定站立 (up>0.85, z≈stand, 着地, |gyro|<1 连续)
            rec_i = None
            for i, r in enumerate(post):
                if (abs(r[2] - args.stand) < 0.12 and r[3] > 0.85 and r[9] > 0.5):
                    rec_i = args.settle + args.pulse + i
                    break
            zs = [r[2] for r in post]
            gyros = [r[8] for r in post]
            # 水平漂移
            x0, y0 = 0.0, 0.0
            drift = 0.0
            print("\n[段3] 落地后姿态 ({} 步)".format(len(post)))
            if rec_i is not None:
                print(f"  恢复站立 @step {rec_i} (落地后 {rec_i - (args.settle+args.pulse)} 步)")
            else:
                print("  恢复站立: ⚠️ 窗口内未完全恢复 (z≈0.52, up>0.85, 着地)")
            print(f"  恢复期 |gyro| 均值 {np.mean(gyros):.3f} max {max(gyros):.2f} "
                  f"({'✅稳定' if np.mean(gyros)<1.0 else '❌晃动'})")
            print(f"  恢复期 base_z 均值 {np.mean(zs):.3f} 波动 {np.std(zs):.3f}")
            # 采样恢复轨迹
            print("  采样 (每步):")
            for i in [0, len(post)//3, 2*len(post)//3, -1]:
                r = post[i]
                t = _posture_tags(r[2], r[3], r[4], r[5], r[6], r[8])
                print(f"    step{r[0]:4d} z={r[2]:.3f} up={r[3]:.3f} |gyro|={r[8]:.2f} "
                      f"膝=[{r[4][2]:+.2f},{r[4][5]:+.2f}] pitch={r[6]:+.2f} 姿态={t if t else '正常'}")
            # 最终站姿
            last = post[-1]
            tj = _posture_tags(last[2], last[3], last[4], last[5], last[6], last[8])
            print(f"  最终姿态 step{last[0]}: z={last[2]:.3f} up={last[3]:.3f} 关节={_fmt_joint(last[4])} "
                  f"姿态={tj if tj else '正常站立'}")
        else:
            print("\n[段3] 落地后: 无采样 (episode 提前终止?)")
        return 0
    finally:
        env.close()


def _fmt_joint(j):
    return f"[{j[0]:+.2f},{j[1]:+.2f},{j[2]:+.2f},{j[3]:+.2f},{j[4]:+.2f},{j[5]:+.2f}]"


if __name__ == "__main__":
    raise SystemExit(main())
