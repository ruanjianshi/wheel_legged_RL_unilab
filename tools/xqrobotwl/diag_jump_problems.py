#!/usr/bin/env python3
"""量化诊断四算法可视化观察到的四个问题 (2026-08-16).

问题清单 (用户可视化观察):
  P1  纯PPO    — 基本没有跳跃过程 (无下蹲/无起跳)
  P2  PPO+VMC  — 有跳跃但无正常蹲→起跳时序 (下蹲蓄力不足/膝过伸)
  P3  SRL      — 无指令(trigger=0)也持续跳跃 + 站立姿态偏差大
  P4  SRL+VMC  — 无下蹲起跳 + 髋关节外展(hip_roll)方式跳跃

每个算法跑两组确定性测试 (禁随机命令重采样, 保证 trigger 精确受控):

  test_self_trigger: trigger 恒 0 共 300 ctrl 步
      → 站立姿态 (base_z/|gyro|/关节角) + 自跳检测 (base_z 周期性上跳)
  test_jump: settle 80 + pulse 160 + tail 120 单次跳跃脉冲
      → 下蹲深度 / 起跳 vz / 跳高 / hip_roll 外展 / 膝过伸 / 相位时序 / 腾空步数

用法:
  uv run tools/xqrobotwl/diag_jump_problems.py                # 全部四算法
  uv run tools/xqrobotwl/diag_jump_problems.py --algos SRL    # 单算法
  uv run tools/xqrobotwl/diag_jump_problems.py --algos SRL,SRL+VMC
  uv run tools/xqrobotwl/diag_jump_problems.py --out logs/pose_data/jump_problem_diag.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.xqrobotwl.verify_jump import load_actor, trained_env_overrides  # noqa: E402

ALGOS = {
    "纯PPO": ("XqRobotWLJumpFlat",
              "logs/rsl_rl_ppo/XqRobotWLJumpFlat/2026-08-16_01-53-39_mujoco/model_9999.pt"),
    "PPO+VMC": ("XqRobotWLJumpVMC",
                "logs/rsl_rl_ppo/XqRobotWLJumpVMC/2026-08-16_01-53-43_mujoco/model_1000.pt"),
    "SRL": ("XqRobotWLJumpSRLFlat",
            "logs/rsl_rl_ppo/XqRobotWLJumpSRLFlat/2026-08-16_13-36-25_mujoco/model_3999.pt"),
    "SRL+VMC": ("XqRobotWLJumpSRLVMC",
                "logs/rsl_rl_ppo/XqRobotWLJumpSRLVMC/2026-08-16_14-08-53_mujoco/model_3999.pt"),
}

SETTLE, PULSE, TAIL = 80, 160, 120
# 关节序: [L_roll, L_pitch, L_knee, R_roll, R_pitch, R_knee, L_wheel, R_wheel]
ROLL_L, PITCH_L, KNEE_L, ROLL_R, PITCH_R, KNEE_R = 0, 1, 2, 3, 4, 5
KNEE_LIMIT = 0.85  # 膝关节极限 (±rad, CLAUDE.md §1.3)


def _make_env(task, ckpt):
    from unilab.base import registry
    from unilab.training import ensure_registries

    ensure_registries()
    ov = trained_env_overrides(ckpt) or {}
    ov["commands"] = {"resampling_time": 0.0}  # 禁随机命令重采样 → trigger 精确受控
    env = registry.make(task, num_envs=1, sim_backend="mujoco", env_cfg_override=ov)
    rt = getattr(getattr(env._cfg, "commands", None), "resampling_time", None)
    cdt = getattr(env._cfg, "ctrl_dt", None)
    print(f"    [env] resampling_time={rt} ctrl_dt={cdt} reward.feedback_gain="
          f"{getattr(getattr(env, '_jump_cfg', None), 'feedback_gain', None)}", flush=True)
    return env


def _wheel_contact(env):
    try:
        left = np.asarray(
            env._backend.get_sensor_data("left_wheel_world_pos"), dtype=np.float64
        ).reshape(-1, 3)[0]
        return float(left[2] < 0.13)
    except (KeyError, AttributeError):
        return 1.0


def _run(task, ckpt, hidden, total, trigger_fn):
    env = _make_env(task, ckpt)
    try:
        obs_dim = env.obs_groups_spec["obs"]
        actor = load_actor(ckpt, obs_dim, 8, hidden)
        env.init_state()
        recs = []
        with torch.no_grad():
            for step in range(total):
                env.state.info["commands"][:, 4] = trigger_fn(step)
                obs = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
                action = actor(obs).numpy()
                st = env.step(action)
                bp = np.asarray(env._backend.get_base_pos())[0]
                dof = np.asarray(env.get_dof_pos())[0]
                lv = env.get_local_linvel()[0]
                gyro = np.asarray(env._backend.get_sensor_data("gyro")).reshape(-1, 3)[0]
                cmd4 = float(np.asarray(env.state.info["commands"])[0, 4])
                recs.append(dict(
                    step=step,
                    trigger=float(trigger_fn(step)),
                    cmd4=cmd4,
                    z=float(bp[2]),
                    vz=float(lv[2]),
                    roll_L=float(dof[ROLL_L]), pitch_L=float(dof[PITCH_L]), knee_L=float(dof[KNEE_L]),
                    roll_R=float(dof[ROLL_R]), pitch_R=float(dof[PITCH_R]), knee_R=float(dof[KNEE_R]),
                    wL=float(dof[6]), wR=float(dof[7]),
                    gyro=float(np.linalg.norm(gyro)),
                    contact=float(_wheel_contact(env)),
                    phase=float(getattr(env, "_fsm_state", np.array([-1.0]))[0]),
                ))
                if st.terminated[0]:
                    break
        return recs
    finally:
        env.close()


def _find_jump_events(base_z, stand, thr=0.12, min_fall=0.05):
    """统计 trigger 关闭时 base_z 的自发上跳事件 (反复弹跳检测)."""
    n = len(base_z)
    jumps, in_air, starts = 0, False, []
    for i in range(n):
        if base_z[i] > stand + thr and not in_air:
            jumps += 1
            in_air = True
            starts.append(i)
        elif base_z[i] < stand + min_fall:
            in_air = False
    return jumps, starts


def test_self_trigger(task, ckpt, hidden, steps=300):
    recs = _run(task, ckpt, hidden, steps, lambda s: 0.0)
    n = len(recs)
    z = np.array([r["z"] for r in recs])
    g = np.array([r["gyro"] for r in recs])
    stand = float(np.median(z[: min(30, n)]))
    jumps, starts = _find_jump_events(z, stand)
    phase = np.array([r["phase"] for r in recs])
    # 站立期窗口 (若发生自跳, 取前 60 步与末 30 步)
    win = recs[: min(60, n)] if n > 90 else recs
    last = recs[-min(30, n):]
    def _avg(rs, k):
        return float(np.mean([r[k] for r in rs])) if rs else float("nan")
    return dict(
        steps=n,
        stand_z=stand,
        max_z=float(z.max()),
        max_z_rel=float(z.max() - stand),
        jump_events=jumps,
        jump_starts=starts,
        fsm_idle_frac=float((phase == -1).mean()),
        thrust_frac=float((phase == 1).mean()),
        stand_gyro_mean=_avg(win, "gyro"),
        stand_gyro_max=float(g[: min(60, n)].max()) if n else float("nan"),
        end_gyro_mean=_avg(last, "gyro"),
        end_z=float(z[-1]),
        end_roll=(_avg(last, "roll_L"), _avg(last, "roll_R")),
        end_pitch=(_avg(last, "pitch_L"), _avg(last, "pitch_R")),
        end_knee=(_avg(last, "knee_L"), _avg(last, "knee_R")),
    )


def test_jump(task, ckpt, hidden):
    recs = _run(task, ckpt, hidden, SETTLE + PULSE + TAIL, lambda s: 1.0 if SETTLE <= s < SETTLE + PULSE else 0.0)
    n = len(recs)
    on = [r for r in recs if r["trigger"] > 0.5]
    z = np.array([r["z"] for r in recs])
    stand = float(np.median([r["z"] for r in recs[: min(30, n)] if r["contact"] > 0.5])) or float(np.median(z[: min(30, n)]))
    peak = max((r["z"] for r in on), default=stand)
    jump_height = peak - stand
    # 腾空步数 (几何接触: 轮心 z<0.13 → 离地)
    air_steps = sum(1 for r in on if r["contact"] < 0.5)
    # 下蹲: 触发后起跳前最低点
    vz = np.array([r["vz"] for r in recs])
    takeoff_i = None
    for i, r in enumerate(recs):
        if r["trigger"] > 0.5 and vz[i] > 0.2:
            takeoff_i = i
            break
    if takeoff_i is not None:
        crouch_win = [r["z"] for r in recs[SETTLE:takeoff_i + 1]]
        min_crouch_z = float(min(crouch_win)) if crouch_win else stand
    else:
        min_crouch_z = float(min((r["z"] for r in on), default=stand))
    # 髋外展 (hip_roll) 峰值 — 触发窗口内
    roll_L = [r["roll_L"] for r in on]
    roll_R = [r["roll_R"] for r in on]
    max_roll_L = float(max(map(abs, roll_L))) if roll_L else 0.0
    max_roll_R = float(max(map(abs, roll_R))) if roll_R else 0.0
    # 膝过伸 (|knee| 峰值) — 触发窗口内
    knee_L = [r["knee_L"] for r in on]
    knee_R = [r["knee_R"] for r in on]
    max_knee_L = float(max(map(abs, knee_L))) if knee_L else 0.0
    max_knee_R = float(max(map(abs, knee_R))) if knee_R else 0.0
    # 相位序列 (去重)
    seq, last_ph = [], None
    for r in recs:
        p = int(r["phase"])
        if p != last_ph:
            seq.append(p)
            last_ph = p
    # 起跳时姿态 (若找到起跳)
    takeoff_pose = None
    if takeoff_i is not None:
        r = recs[takeoff_i]
        takeoff_pose = dict(
            z=round(r["z"], 3), vz=round(r["vz"], 3),
            roll=(round(r["roll_L"], 3), round(r["roll_R"], 3)),
            pitch=(round(r["pitch_L"], 3), round(r["pitch_R"], 3)),
            knee=(round(r["knee_L"], 3), round(r["knee_R"], 3)),
        )
    return dict(
        stand_z=round(stand, 3),
        min_crouch_z=round(min_crouch_z, 3),
        crouch_depth=round(stand - min_crouch_z, 3),
        peak_z=round(peak, 3),
        jump_height=round(jump_height, 3),
        takeoff_step=takeoff_i,
        takeoff_vz=round(float(vz[takeoff_i]), 3) if takeoff_i is not None else None,
        takeoff_pose=takeoff_pose,
        max_roll_L=round(max_roll_L, 3),
        max_roll_R=round(max_roll_R, 3),
        max_roll=max(round(max_roll_L, 3), round(max_roll_R, 3)),
        max_knee_L=round(max_knee_L, 3),
        max_knee_R=round(max_knee_R, 3),
        max_knee=max(round(max_knee_L, 3), round(max_knee_R, 3)),
        knee_overextend=(max_knee_L > KNEE_LIMIT) or (max_knee_R > KNEE_LIMIT),
        air_steps=air_steps,
        phase_seq=seq,
        terminated=n < SETTLE + PULSE + TAIL,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--algos", default="纯PPO,PPO+VMC,SRL,SRL+VMC")
    p.add_argument("--hidden", default="512,512,256,128")
    p.add_argument("--out", default="logs/pose_data/jump_problem_diag.json")
    p.add_argument("--self_steps", type=int, default=300)
    args = p.parse_args()
    hidden = [int(x) for x in args.hidden.split(",")]
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = {}
    for algo in args.algos.split(","):
        task, ckpt = ALGOS[algo]
        print(f"\n========== [{algo}] task={task} ==========", flush=True)
        print(f"[self_trigger] trigger=0 x {args.self_steps} 步...", flush=True)
        st = test_self_trigger(task, ckpt, hidden, args.self_steps)
        print(f"  stand_z={st['stand_z']:.3f} max_z={st['max_z']:.3f} "
              f"max_z_rel={st['max_z_rel']:.3f} jump_events={st['jump_events']} "
              f"starts={st['jump_starts']}")
        print(f"  fsm idle={st['fsm_idle_frac']*100:.0f}% thrust={st['thrust_frac']*100:.0f}% "
              f"stand|gyro|={st['stand_gyro_mean']:.3f}(max {st['stand_gyro_max']:.3f}) "
              f"end|gyro|={st['end_gyro_mean']:.3f}")
        print(f"  end pose: z={st['end_z']:.3f} roll=({st['end_roll'][0]:+.3f},{st['end_roll'][1]:+.3f}) "
              f"pitch=({st['end_pitch'][0]:+.3f},{st['end_pitch'][1]:+.3f}) "
              f"knee=({st['end_knee'][0]:+.3f},{st['end_knee'][1]:+.3f})")
        print(f"[jump] settle {SETTLE} + pulse {PULSE} + tail {TAIL}...", flush=True)
        jm = test_jump(task, ckpt, hidden)
        print(f"  stand={jm['stand_z']} crouch_min={jm['min_crouch_z']} "
              f"crouch_depth={jm['crouch_depth']} peak={jm['peak_z']} jump={jm['jump_height']}m")
        print(f"  takeoff@{jm['takeoff_step']} vz={jm['takeoff_vz']} pose={jm['takeoff_pose']}")
        print(f"  hip_roll max: L={jm['max_roll_L']} R={jm['max_roll_R']} (abduction={jm['max_roll']})")
        print(f"  knee max: L={jm['max_knee_L']} R={jm['max_knee_R']} overextend={jm['knee_overextend']} "
              f"(limit ±{KNEE_LIMIT})")
        print(f"  air_steps={jm['air_steps']} phase_seq={jm['phase_seq']} terminated={jm['terminated']}")
        result[algo] = {"self_trigger": st, "jump": jm}

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
