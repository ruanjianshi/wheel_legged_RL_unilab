#!/usr/bin/env python3
"""完整跳跃评估 (CLAUDE.md §7.5 + §1.3 姿态表格)。

按开发规范对一次跳跃周期做逐阶段姿态监控与达标判定:
  站立 → 触发(下蹲蓄力) → 起跳 → 空中 → 落地 → 恢复站立

每个阶段: 读 6 关节角度 + base 欧拉角 + up_z + 轮地接触 + 轮速, 按 §1.3 姿态表格
分类 (正常站立/下蹲/伸腿/前倾/后倾/左右倾斜/高低腿/髋外展/轮子点地/摇摆/转圈),
输出达标/异常判定。

Usage:
    uv run tools/xqrobotwl/eval_jump_full.py --task XqRobotWLJumpVMC \
        --checkpoint logs/rsl_rl_ppo/XqRobotWLJumpVMC/<run>/model_N.pt \
        [--out logs/pose_data/<prefix>_jump_full.csv] [--settle 60] [--pulse 140] [--tail 160]
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.xqrobotwl.verify_jump import load_actor, trained_env_overrides  # noqa: E402

# 自然站姿 (CLAUDE.md §1.3, walk 实测)
STANDING_ANGLES = np.array([0.102, 0.083, -0.079, 0.013, -0.108, 0.019])
STANDING_Z = 0.52


# ── §1.3 姿态分类: 6 关节角度 + base 欧拉 → 姿态类别 ───────────────
def classify_pose(z, up_z, j, euler, gyro_mag, contact, standing_z=STANDING_Z):
    """j = [L_hip_roll, L_hip_pitch, L_knee, R_hip_roll, R_hip_pitch, R_knee] (rad)."""
    issues = []
    # 左右腿一前一后 / 高低腿 / 髋外展
    dpitch = abs(j[1] - j[4])
    dknee = abs(j[2] - j[5])
    if dpitch > 0.3:
        issues.append("左右腿一前一后")
    if dknee > 0.4:
        issues.append("左右高低腿")
    if j[0] > 0.5 or j[3] < -0.5:
        issues.append("髋外展/内收")
    # 前后倾 (用 base_pitch 近似 up 方向)
    if abs(euler[1]) > 0.2:
        issues.append("前倾" if euler[1] > 0 else "后倾")
    # 左右倾斜
    if abs(euler[0]) > 0.2:
        issues.append("左右倾斜")
    # 摇摆 (站立时角速度大)
    if gyro_mag > 1.0:
        issues.append("摇摆")
    # 高度异常
    if z > standing_z + 0.12:
        issues.append("伸腿/过高")
    elif z < 0.25:
        issues.append("倒地")
    return issues


def _phase_stats(rows):
    """从 CSV 行 (dict) 提取阶段平均关节/姿态。"""
    n = len(rows)
    if n == 0:
        return None
    z = np.mean([float(r["base_z"]) for r in rows])
    up = np.mean([float(r["up_z"]) for r in rows])
    j = np.array([
        np.mean([float(r[k]) for r in rows])
        for k in ("L_hip_roll", "L_hip_pitch", "L_knee", "R_hip_roll", "R_hip_pitch", "R_knee")
    ])
    gyro = np.mean([(float(r["gyro_x"])**2 + float(r["gyro_y"])**2 + float(r["gyro_z"])**2)**0.5 for r in rows])
    roll = np.mean([float(r["base_roll"]) for r in rows])
    pitch = np.mean([float(r["base_pitch"]) for r in rows])
    wc = np.mean([float(r["wheel_contact_L"]) for r in rows])
    dw = np.mean([abs(float(r["L_wheel_rads"])) + abs(float(r["R_wheel_rads"])) for r in rows])
    return dict(z=z, up=up, j=j, gyro=gyro, roll=roll, pitch=pitch, wc=wc, dw=dw)


def _fmt_joint(j):
    return f"[{j[0]:+.2f},{j[1]:+.2f},{j[2]:+.2f},{j[3]:+.2f},{j[4]:+.2f},{j[5]:+.2f}]"


def main() -> int:
    p = argparse.ArgumentParser(description="完整跳跃评估 §7.5 + §1.3 姿态表格")
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--settle", type=int, default=80, help="触发前站立步数")
    p.add_argument("--pulse", type=int, default=160, help="触发 ON 步数")
    p.add_argument("--tail", type=int, default=200, help="触发后恢复步数")
    p.add_argument("--standing", type=float, default=STANDING_Z)
    p.add_argument("--wheel_radius", type=float, default=0.11)
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
        cols = [
            "step", "trigger", "base_z",
            "L_hip_roll", "L_hip_pitch", "L_knee", "R_hip_roll", "R_hip_pitch", "R_knee",
            "L_wheel_rads", "R_wheel_rads", "base_roll", "base_pitch", "base_yaw",
            "linvel_x", "linvel_y", "linvel_z", "gyro_x", "gyro_y", "gyro_z",
            "up_z", "wheel_contact_L", "wheel_contact_R", "jump_phase",
        ]
        rows = []
        with torch.no_grad():
            for step in range(total):
                trig = 1.0 if args.settle <= step < args.settle + args.pulse else 0.0
                env.state.info["commands"][:, 4] = trig
                obs = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
                action = actor(obs).numpy()
                st = env.step(action)
                dof = env.get_dof_pos()[0]
                dv = env.get_dof_vel()[0]
                bp = np.asarray(env._backend.get_base_pos())[0]
                quat = env._backend.get_base_quat()[0]
                w, x, y, z = quat[0], quat[1], quat[2], quat[3]
                roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
                pitch = math.asin(max(-1.0, min(1.0, 2*(w*y - z*x))))
                yaw = math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
                lv = env.get_local_linvel()[0]
                gyro = np.asarray(env._backend.get_sensor_data("gyro")).reshape(-1, 3)[0]
                up_z = float(np.asarray(env._backend.get_sensor_data("upvector")).reshape(-1, 3)[0, 2])
                lz = float(np.asarray(env._backend.get_sensor_data("left_wheel_world_pos")).reshape(-1, 3)[0, 2])
                contact = 1.0 if lz < args.wheel_radius + 0.02 else 0.0
                phase = float(st.info.get("jump_phase", np.zeros(1))[0])
                rows.append({
                    "step": step, "trigger": trig, "base_z": float(bp[2]),
                    "L_hip_roll": float(dof[0]), "L_hip_pitch": float(dof[1]), "L_knee": float(dof[2]),
                    "R_hip_roll": float(dof[3]), "R_hip_pitch": float(dof[4]), "R_knee": float(dof[5]),
                    "L_wheel_rads": float(dv[6]), "R_wheel_rads": float(dv[7]),
                    "base_roll": roll, "base_pitch": pitch, "base_yaw": yaw,
                    "linvel_x": float(lv[0]), "linvel_y": float(lv[1]), "linvel_z": float(lv[2]),
                    "gyro_x": float(gyro[0]), "gyro_y": float(gyro[1]), "gyro_z": float(gyro[2]),
                    "up_z": up_z, "wheel_contact_L": contact, "wheel_contact_R": contact, "jump_phase": phase,
                })
                if st.terminated[0]:
                    break

        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)
            print(f"[data] wrote {args.out} ({len(rows)} rows)")

        n = len(rows)
        trigs = [r["trigger"] for r in rows]
        zs = [r["base_z"] for r in rows]
        ups = [r["up_z"] for r in rows]
        wcs = [r["wheel_contact_L"] for r in rows]
        on = [i for i in range(n) if trigs[i] > 0.5]

        print("=" * 72)
        print(f"完整跳跃评估 {args.task}")
        print(f"checkpoint: {args.checkpoint}")
        print(f"rows={n} terminated={rows[-1].get('terminated', False)}")
        print("=" * 72)

        # ── 1. 站立姿态 (触发前 settle) ─────────────────────────────
        stand_sel = [r for r in rows[:args.settle] if r["wheel_contact_L"] > 0.5]
        if stand_sel:
            s = _phase_stats(stand_sel)
            j = s["j"]
            issues = classify_pose(s["z"], s["up"], j, [s["roll"], s["pitch"], 0], s["gyro"], s["wc"] > 0.9, args.standing)
            # 站姿对称性 vs 自然站姿
            sym = abs(j[0] + j[3]) < 0.15 and abs(j[1] - j[4]) < 0.3 and abs(j[2] - j[5]) < 0.3
            height_ok = abs(s["z"] - args.standing) < 0.12
            drift = float(np.hypot(rows[-1]["linvel_x"], rows[-1]["linvel_y"]))  # 最后步水平速度
            print(f"\n[1] 站立姿态 (触发前 {len(stand_sel)} 步, 轮着地)")
            print(f"    base_z={s['z']:.3f} (目标 {args.standing}, {'✅' if height_ok else '❌'}) "
                  f"up_z={s['up']:.3f} |gyro|={s['gyro']:.2f} 对称={'✅' if sym else '❌'}")
            print(f"    关节={_fmt_joint(j)} (自然站姿 {_fmt_joint(STANDING_ANGLES)})")
            print(f"    异常: {issues if issues else '✅ 无'}")
        else:
            print("\n[1] 站立姿态: ❌ 触发前轮未着地 (无法评估)")

        # ── 2. 触发 → 下蹲蓄力 ──────────────────────────────────────
        on_sel = [r for r in rows if r["trigger"] > 0.5]
        # 找深蹲 (trigger 窗口内最低接地 z)
        crouch = min((r for r in on_sel if r["wheel_contact_L"] > 0.5), key=lambda r: r["base_z"], default=None)
        if crouch:
            j = np.array([crouch[k] for k in ("L_hip_roll","L_hip_pitch","L_knee","R_hip_roll","R_hip_pitch","R_knee")])
            z_c = crouch["base_z"]
            knee_bend = abs(j[2]) + abs(j[5])
            forward = (j[1] > 0.05) and (-j[4] > 0.05) or (j[1] - j[4] > 0)
            print(f"\n[2] 触发→下蹲蓄力 @step {crouch['step']}")
            print(f"    base_z={z_c:.3f} (下蹲深度 {args.standing - z_c:+.3f}, "
                  f"膝屈 {knee_bend:.2f} rad, {'✅ 深蹲' if z_c < 0.45 else '⚠️ 下蹲不足' if z_c < 0.52 else '❌ 未下蹲'})")
            print(f"    关节={_fmt_joint(j)} 异常: {classify_pose(z_c, crouch['up_z'], j, [crouch['base_roll'], crouch['base_pitch'], 0], 0, True, args.standing) or '✅ 无'}")
        else:
            print("\n[2] 触发→下蹲: ❌ 未检测到下蹲 (可能未起跳)")

        # ── 3. 起跳 (轮离地瞬间) ────────────────────────────────────
        air_start = next((i for i in range(n) if trigs[i] > 0.5 and wcs[i] < 0.5), None)
        # 起跳前最后接地步
        lift = None
        if air_start is not None:
            for i in range(air_start, -1, -1):
                if wcs[i] > 0.5:
                    lift = i
                    break
        if air_start is not None:
            delay = air_start - args.settle
            print(f"\n[3] 起跳 (轮离地) @step {air_start} (触发后 {delay} 步, "
                  f"{'✅ 及时' if delay < 100 else '⚠️ 延迟'})")
            if lift is not None:
                j = np.array([rows[lift][k] for k in ("L_hip_roll","L_hip_pitch","L_knee","R_hip_roll","R_hip_pitch","R_knee")])
                print(f"    起跳前接地 @step {lift}: z={rows[lift]['base_z']:.3f} 膝={j[2]:+.2f},{j[5]:+.2f} "
                      f"(蹬伸 {'✅' if abs(j[2]) < 0.2 or abs(j[5]) < 0.2 else '⚠️ 膝未伸直'})")
        else:
            print("\n[3] 起跳: ❌ 触发窗口内轮从未离地 (未腾空)")

        # ── 4. 空中姿态 ─────────────────────────────────────────────
        air_sel = [r for r in rows if r["wheel_contact_L"] < 0.5 and r["trigger"] > 0.5]
        if air_sel:
            peak = max(air_sel, key=lambda r: r["base_z"])
            a = _phase_stats(air_sel)
            j = a["j"]
            issues = classify_pose(a["z"], a["up"], j, [a["roll"], a["pitch"], 0], a["gyro"], False, args.standing)
            air_steps = len(air_sel)
            jh = peak["base_z"] - args.standing
            print(f"\n[4] 空中姿态 ({air_steps} 步腾空, 峰值 @step {peak['step']} z={peak['base_z']:.3f}, 跳高 {jh:.3f} m)")
            print(f"    up_z={a['up']:.3f} base_pitch={a['pitch']:+.2f} 轮速={a['dw']:.1f} rad/s "
                  f"({'✅ 空中收腿' if a['dw'] < 15 else '⚠️ 空中转轮'})")
            print(f"    关节={_fmt_joint(j)} 异常: {issues if issues else '✅ 无'}")
        else:
            print("\n[4] 空中姿态: ❌ 无腾空")

        # ── 5. 落地姿态 ─────────────────────────────────────────────
        land_i = None
        if air_start is not None:
            for i in range(air_start, n):
                if wcs[i] > 0.5:
                    land_i = i
                    break
        if land_i is not None:
            # 落地缓冲: 落地后 20 步的最低 z (吸收深度)
            absorb_sel = rows[land_i:land_i + 25]
            absorb_z = min((r["base_z"] for r in absorb_sel), default=rows[land_i]["base_z"])
            impact = float(np.hypot(rows[land_i]["linvel_x"], rows[land_i]["linvel_z"]))
            pre = max(0, land_i - 4)
            avg_w = np.mean([abs(float(r["L_wheel_rads"])) + abs(float(r["R_wheel_rads"])) for r in rows[pre:land_i + 1]])
            print(f"\n[5] 落地 @step {land_i}")
            print(f"    落地冲击 vx/vz={rows[land_i]['linvel_x']:.2f}/{rows[land_i]['linvel_z']:.2f} m/s "
                  f"({'✅' if impact < 2.0 else '⚠️ 冲击偏大'})")
            print(f"    落地缓冲: 落地后最低 z={absorb_z:.3f} "
                  f"({'✅ 缓冲正常' if absorb_z > 0.25 else '❌ 塌陷'})")
            print(f"    落地时轮速 |w|={avg_w:.1f} rad/s "
                  f"({'✅ 不空转' if avg_w < 5 else '⚠️ 空转/打滑'})")
        else:
            print("\n[5] 落地: ❌ 未检测到落地")

        # ── 6. 恢复站立 ─────────────────────────────────────────────
        if land_i is not None:
            rec_i = None
            for i in range(land_i, min(land_i + 100, n)):
                if (abs(rows[i]["base_z"] - args.standing) < 0.12 and rows[i]["up_z"] > 0.85
                        and wcs[i] > 0.5):
                    rec_i = i
                    break
            if rec_i is not None:
                rec_sel = rows[rec_i:rec_i + 20]
                r = _phase_stats(rec_sel)
                j = r["j"]
                issues = classify_pose(r["z"], r["up"], j, [r["roll"], r["pitch"], 0], r["gyro"], True, args.standing)
                # 微动平衡: 恢复后 20 步水平漂移
                x0 = float(rows[rec_i]["linvel_x"]); y0 = float(rows[rec_i]["linvel_y"])
                print(f"\n[6] 恢复站立 @step {rec_i} (落地后 {rec_i - land_i} 步)")
                print(f"    base_z={r['z']:.3f} up_z={r['up']:.3f} |gyro|={r['gyro']:.2f} "
                      f"(稳定 {'✅' if r['gyro'] < 1.0 else '❌ 晃动'})")
                print(f"    关节={_fmt_joint(j)} 异常: {issues if issues else '✅ 无'}")
            else:
                print("\n[6] 恢复站立: ❌ 落地后 {:.0f} 步内未恢复到稳定站立 (z≈{:.2f}, up>0.85, 轮着地)".format(args.standing))
        else:
            print("\n[6] 恢复站立: ❌ 未落地无法评估")

        # ── 汇总 ─────────────────────────────────────────────────────
        print("\n" + "=" * 72)
        print("汇总判定")
        jh = max(zs) - args.standing
        have_air = air_start is not None
        have_land = land_i is not None
        # 防御: 各阶段可能未走到, 用安全默认
        height_ok_s = locals().get("height_ok", False)
        crouch_ok = locals().get("crouch") is not None and locals().get("crouch")["base_z"] < 0.45
        crouch_warn = locals().get("crouch") is not None
        air_steps_s = locals().get("air_steps", 0)
        air_dw = locals().get("a", {}).get("dw", 0) if isinstance(locals().get("a"), dict) else 0
        absorb_s = locals().get("absorb_z", 0.0)
        rec_ok = locals().get("rec_i") is not None
        checks = [
            ("站立姿态", "✅" if locals().get("stand_sel") and height_ok_s else "⚠️" if locals().get("stand_sel") else "❌"),
            ("下蹲蓄力", "✅" if crouch_ok else "⚠️" if crouch_warn else "❌"),
            ("腾空", "✅" if have_air and air_steps_s > 5 else "⚠️" if have_air else "❌"),
            ("跳高>0.2m", "✅" if jh > 0.20 else "⚠️" if jh > 0.10 else "❌"),
            ("落地", "✅" if have_land and absorb_s > 0.25 else "❌"),
            ("恢复站立", "✅" if rec_ok else "❌"),
            ("空中轮速匹配", "✅" if have_air and air_dw < 15 else "⚠️"),
        ]
        for name, v in checks:
            print(f"  {name}: {v}")
        score = sum(1 for _, v in checks if v == "✅")
        print(f"  达标 {score}/{len(checks)}")
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
