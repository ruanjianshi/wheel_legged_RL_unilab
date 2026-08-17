#!/usr/bin/env python3
"""thrust_length 扫描 — 验证膝止位碰撞是否是跳高瓶颈.

用训练好的策略站定/下蹲/蹬伸, 覆盖 thrust_length 配置, 测跳高/膝超伸/峰值 vz。
决定性实验: 若 thrust_length 越短跳越高 (避免膝撞止位), 则止位碰撞就是瓶颈;
反之若越长越高, 则超伸蹬伸确实贡献推力。

Usage:
    uv run tools/xqrobotwl/diag_thrust_length_scan.py \
        --task XqRobotWLJumpSRLVMC --checkpoint logs/.../model_3999.pt
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


def run_one(env, actor, thrust_len: float, ff_scale: float, guard_start: float,
            brake_kd: float = 8.0, kd_l0: float | None = None,
            settle: int = 200, jump_len: int = 200, recovery_len: int = 150):
    """Run one episode with swept thrust params. Returns metrics dict."""
    vmc_cfg = env._vmc_cfg
    orig = (vmc_cfg.thrust_length, vmc_cfg.thrust_ff_scale,
            vmc_cfg.knee_guard_start, vmc_cfg.knee_brake_kd, vmc_cfg.kd_l0)
    vmc_cfg.thrust_length = thrust_len
    vmc_cfg.thrust_ff_scale = ff_scale
    vmc_cfg.knee_guard_start = guard_start
    vmc_cfg.knee_brake_kd = brake_kd
    if kd_l0 is not None:
        vmc_cfg.kd_l0 = kd_l0
    try:
        env.reset(np.array([0]))
        env.init_state()
        max_z = 0.0
        max_knee = 0.0
        max_vz = 0.0
        max_vx = 0.0
        standing_log = []
        recovery_gyro = []
        with torch.no_grad():
            total = settle + jump_len + recovery_len
            for step in range(total):
                trigger = 1.0 if (settle <= step < settle + jump_len // 2) else 0.0
                env.state.info["commands"][:, 4] = trigger
                obs = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
                action = actor(obs).numpy()
                state = env.step(action)
                base_z = float(np.asarray(env._backend.get_base_pos())[0, 2])
                linvel = env.get_local_linvel()[0]
                vz = float(linvel[2])
                gyro = env.get_gyro()[0]
                dof_pos = env.get_dof_pos()
                knee_max = float(max(abs(dof_pos[0, 2]), abs(dof_pos[0, 5])))
                contact = state.info.get("wheel_contact", np.zeros((1, 2)))
                air = 1.0 - float(np.mean(contact))
                if step < settle and air < 0.01:
                    standing_log.append(base_z)
                max_z = max(max_z, base_z)
                max_knee = max(max_knee, knee_max)
                max_vz = max(max_vz, vz)
                max_vx = max(max_vx, abs(float(linvel[0])))
                # 恢复期 (跳后 trigger off, 轮着地): 记录 |gyro|
                if step >= settle + jump_len and air < 0.01:
                    recovery_gyro.append(float(np.linalg.norm(gyro)))
                if state.terminated[0]:
                    break
        stand = float(np.median(standing_log)) if standing_log else 0.53
        rec_gyro = float(np.mean(recovery_gyro[-60:])) if recovery_gyro else -1.0
        return {
            "thrust_len": thrust_len,
            "ff_scale": ff_scale,
            "guard_start": guard_start,
            "jump": max_z - stand,
            "apex": max_z,
            "max_knee": max_knee,
            "max_vz": max_vz,
            "max_vx": max_vx,
            "rec_gyro": rec_gyro,
        }
    finally:
        vmc_cfg.thrust_length, vmc_cfg.thrust_ff_scale = orig[0], orig[1]
        vmc_cfg.knee_guard_start, vmc_cfg.knee_brake_kd = orig[2], orig[3]
        vmc_cfg.kd_l0 = orig[4]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hidden", default="512,512,256,128")
    p.add_argument("--ff_scales", default="1.0,2.0,3.5", help="thrust_ff_scale 扫描值")
    p.add_argument("--guard_starts", default="0.35,0.50", help="knee_guard_start 扫描值")
    p.add_argument("--brake_kds", default="0.0,8.0", help="knee_brake_kd 扫描值 (0=关闭刹车)")
    p.add_argument("--kd_l0s", default="", help="kd_l0 扫描值 (空=不扫)")
    args = p.parse_args()
    hidden = [int(x) for x in args.hidden.split(",")]

    from unilab.base import registry
    from unilab.training import ensure_registries

    ensure_registries()
    trained_ov = trained_env_overrides(args.checkpoint)
    env = registry.make(args.task, num_envs=1, sim_backend="mujoco", env_cfg_override=trained_ov)
    obs_dim = env.obs_groups_spec["obs"]
    actor = load_actor(args.checkpoint, obs_dim, 8, hidden)

    kd_l0s = [float(x) for x in args.kd_l0s.split(",")] if args.kd_l0s else [None]
    print(f"{'kd_l0':>6} {'跳高':>7} {'峰值z':>7} {'最大|膝|':>9} {'峰值vz':>7} {'峰值|vx|':>8} {'恢复|gyro|':>10}")
    for kd in kd_l0s:
        r = run_one(env, actor, 0.50, 3.5, 0.50, brake_kd=0.0, kd_l0=kd)
        kd_label = f"{kd:.0f}" if kd is not None else "base"
        print(
            f"{kd_label:>6} {r['jump']:>7.3f} {r['apex']:>7.3f} {r['max_knee']:>9.3f} "
            f"{r['max_vz']:>7.2f} {r['max_vx']:>8.2f} {r['rec_gyro']:>10.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
