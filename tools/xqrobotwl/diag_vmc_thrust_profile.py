#!/usr/bin/env python3
"""VMC 蹬伸力曲线诊断 — 记录一次跳跃全程的 L0/力/力矩, 找起跳瓶颈.

沿 verify_jump.py 的加载方式 (确定性策略 + 触发脉冲), 在每一步记录:
  fsm_state, jump_phase, base_z, vz, L0_actual, L0_target, force_L0 命令,
  hip/knee 实际力矩 (限幅后), 膝角, 轮地接触。

Usage:
    uv run tools/xqrobotwl/diag_vmc_thrust_profile.py \
        --task XqRobotWLJumpSRLVMC --checkpoint logs/.../model_3999.pt \
        [--jump_every 200] [--steps 600]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.xqrobotwl.verify_jump import load_actor, trained_env_overrides  # noqa: E402

ROWS = ["L0_target", "L0_act", "force_cmd", "tau_kneeL", "tau_kneeR", "kneeL", "kneeR"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--jump_every", type=int, default=200)
    p.add_argument("--settle", type=int, default=200, help="前 N 步不触发, 先站稳")
    p.add_argument("--hidden", default="512,512,256,128")
    args = p.parse_args()

    hidden = [int(x) for x in args.hidden.split(",")]
    from unilab.base import registry
    from unilab.training import ensure_registries

    ensure_registries()
    trained_ov = trained_env_overrides(args.checkpoint)
    if trained_ov is None:
        print("!! no run_config.json — 无法还原训练配置")
        return 1
    env = registry.make(args.task, num_envs=1, sim_backend="mujoco", env_cfg_override=trained_ov)
    obs_dim = env.obs_groups_spec["obs"]
    actor = load_actor(args.checkpoint, obs_dim, 8, hidden)
    env.init_state()

    vmc_cfg = env._vmc_cfg
    fb = float(getattr(env._jump_cfg, "feedback_gain", 0.15))
    scale_l0 = vmc_cfg.action_scale_l0
    l0_off = vmc_cfg.l0_offset
    l0_min, l0_max = vmc_cfg.l0_min, vmc_cfg.l0_max

    log = {k: [] for k in ROWS}
    log["step"] = []
    log["fsm"] = []
    log["phase"] = []
    log["z"] = []
    log["vz"] = []
    log["air"] = []
    max_z = 0.0
    jump_z = 0.0
    max_force = 0.0
    max_force_step = -1
    thrust_steps = 0

    def fk_L0(dof_pos):
        th1 = np.asarray(vmc_cfg.hip_sign) * dof_pos[:, [1, 4]] + vmc_cfg.c1
        th2 = np.asarray(vmc_cfg.knee_sign) * dof_pos[:, [2, 5]] + vmc_cfg.c2
        ex = vmc_cfg.offset + vmc_cfg.l1 * np.cos(th1) + vmc_cfg.l2 * np.cos(th1 + th2)
        ey = vmc_cfg.l1 * np.sin(th1) + vmc_cfg.l2 * np.sin(th1 + th2)
        return np.sqrt(ex**2 + ey**2), th1, th2

    settle = args.settle
    with torch.no_grad():
        for step in range(args.steps):
            trigger = (
                1.0 if (settle <= (step % args.jump_every) < settle + (args.jump_every // 2)) else 0.0
            )
            env.state.info["commands"][:, 4] = trigger
            obs = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
            action = actor(obs).numpy()
            state = env.step(action)

            dof_pos = env.get_dof_pos()
            dof_vel = env.get_dof_vel()
            base_z = float(np.asarray(env._backend.get_base_pos())[0, 2])
            linvel = env.get_local_linvel()
            vz = float(linvel[0, 2])
            contact = state.info.get("wheel_contact", np.zeros((1, 2)))
            air = 1.0 - float(np.mean(contact))
            fsm = int(env._fsm_state[0])
            phase = float(state.info.get("jump_phase", np.zeros(1))[0])

            # L0 target after step() blend + apply_action scaling/clip
            ref_phys = env._jump_leg_reference()[0]  # (2,) physical L0 ref
            blend = (ref_phys - l0_off) / scale_l0 + fb * action[0, [2, 5]]
            L0_tgt = np.clip(blend * scale_l0 + l0_off, l0_min, l0_max)
            L0_act, th1, th2 = fk_L0(dof_pos)
            L0_act = L0_act[0]

            # Force command (pre-clip): kp*(tgt-L0) - kd*L0dot + ff
            kp, kd, ff = env.get_l0_control_parameters()
            th1d = np.asarray(vmc_cfg.hip_sign) * dof_vel[:, [1, 4]]
            th2d = np.asarray(vmc_cfg.knee_sign) * dof_vel[:, [2, 5]]
            dt = 1e-3
            ex2 = vmc_cfg.offset + vmc_cfg.l1 * np.cos(th1 + th1d * dt) + vmc_cfg.l2 * np.cos(th1 + th2 + (th1d + th2d) * dt)
            ey2 = vmc_cfg.l1 * np.sin(th1 + th1d * dt) + vmc_cfg.l2 * np.sin(th1 + th2 + (th1d + th2d) * dt)
            L0p = np.sqrt(ex2**2 + ey2**2)
            L0dot = (L0p - L0_act[None, :]) / dt
            force = kp[0] * (L0_tgt - L0_act) - kd[0] * L0dot[0] + ff[0]
            force_mag = float(np.mean(force))

            tau = env._last_vmc_ctrl[0]
            if fsm == 1:
                thrust_steps += 1
                if force_mag > max_force:
                    max_force = force_mag
                    max_force_step = step
            max_z = max(max_z, base_z)

            log["step"].append(step)
            log["fsm"].append(fsm)
            log["phase"].append(phase)
            log["z"].append(base_z)
            log["vz"].append(vz)
            log["air"].append(air)
            log["L0_target"].append(L0_tgt.tolist())
            log["L0_act"].append(L0_act.tolist())
            log["force_cmd"].append(force.tolist())
            log["tau_kneeL"].append(float(tau[2]))
            log["tau_kneeR"].append(float(tau[6]))
            log["kneeL"].append(float(dof_pos[0, 2]))
            log["kneeR"].append(float(dof_pos[0, 5]))

            if state.terminated[0]:
                print(f"!! terminated at step {step}")
                break

    # 站立高度 = trigger off & 轮着地的 z 中位数
    standing = [log["z"][i] for i in range(len(log["z"])) if log["air"][i] < 0.01 and log["phase"][i] == 0]
    stand_z = float(np.median(standing)) if standing else 0.55
    jump_h = max_z - stand_z

    print(f"\n=== {args.task} thrust profile ===")
    print(f"站立 z={stand_z:.3f}  峰值 z={max_z:.3f}  跳高={jump_h:.3f} m")
    print(f"蹬伸相步数={thrust_steps}  最大力命令={max_force:.0f} N @ step {max_force_step}")
    print(f"\n蹬伸相 (fsm==1) 关键帧:")
    print(f"{'step':>5} {'z':>6} {'vz':>6} {'L0tgt':>7} {'L0act':>7} {'force':>7} {'tkneeL':>7} {'tkneeR':>7} {'kneeL':>6} {'kneeR':>6}")
    for i in range(len(log["step"])):
        if log["fsm"][i] == 1:
            tgt = log["L0_target"][i]
            act = log["L0_act"][i]
            fc = log["force_cmd"][i]
            print(
                f"{log['step'][i]:>5} {log['z'][i]:>6.3f} {log['vz'][i]:>6.2f} "
                f"{tgt[0]:>7.3f} {act[0]:>7.3f} {fc[0]:>7.0f} "
                f"{log['tau_kneeL'][i]:>7.1f} {log['tau_kneeR'][i]:>7.1f} "
                f"{log['kneeL'][i]:>6.2f} {log['kneeR'][i]:>6.2f}"
            )
    # 膝角最大值 (膝过伸检查)
    max_kneeL = max(max(k) for k in [log["kneeL"]])
    max_kneeR = max(max(k) for k in [log["kneeR"]])
    min_kneeL = min(min(k) for k in [log["kneeL"]])
    min_kneeR = min(min(k) for k in [log["kneeR"]])
    print(f"\n膝角范围: L=[{min_kneeL:.2f}, {max_kneeL:.2f}] R=[{min_kneeR:.2f}, {max_kneeR:.2f}] (极限 ±0.873)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
