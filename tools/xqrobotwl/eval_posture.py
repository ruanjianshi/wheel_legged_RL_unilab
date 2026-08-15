#!/usr/bin/env python3
"""姿态评估 — §7.0 稳定姿态 + 微动平衡 + 附录A 指标.

站立 (vx=0) 10s, 测: 站立高度 base_z (≈0.52±0.05), 直立度 up[2] (≈1),
水平漂移 (x/y 位移, <0.5m), yaw 累计 (<30°), 微动 linvel (无持续漂移)。

用法:
  uv run python tools/xqrobotwl/eval_posture.py --task walk_flat [--episodes 3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TASK_CONF = {
    "walk_flat": "conf/ppo/task/xqrobotwl_walk_flat/mujoco.yaml",
    "walk_rough": "conf/ppo/task/xqrobotwl_walk_rough/mujoco.yaml",
    "toe_walk": "conf/ppo/task/xqrobotwl_toe_walk_flat/mujoco.yaml",
    "jump": "conf/ppo/task/xqrobotwl_jump_flat/mujoco.yaml",
    "jump_srl": "conf/ppo/task/xqrobotwl_jump_srl_flat/mujoco.yaml",
    "jump_vmc": "conf/ppo/task/xqrobotwl_jump_vmc_flat/mujoco.yaml",
    "jump_srl_vmc": "conf/ppo/task/xqrobotwl_jump_srl_vmc_flat/mujoco.yaml",
    "backflip": "conf/ppo/task/xqrobotwl_backflip_flat/mujoco.yaml",
    "single_leg_flat": "conf/ppo/task/xqrobotwl_single_leg_flat/mujoco.yaml",
    "single_leg_move": "conf/ppo/task/xqrobotwl_single_leg_move/mujoco.yaml",
    "single_leg_unicycle": "conf/ppo/task/xqrobotwl_single_leg_unicycle/mujoco.yaml",
    "fall_recovery": "conf/cpo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml",
    "stairs": "conf/np3o/task/xqrobotwl_stairs/mujoco.yaml",
}


def build_env(task_key: str, num_envs: int = 1):
    import yaml

    from unilab.base import registry
    from unilab.base.registry import ensure_registries

    ensure_registries()
    cfg = yaml.safe_load(open(ROOT / TASK_CONF[task_key]))
    override: dict = {"reward_config": cfg["reward"], **cfg.get("env", {})}
    override["curriculum"] = {"enabled": False}
    override["domain_rand"] = {
        **override.get("domain_rand", {}),
        "randomize_init_yaw": False, "randomize_ground_friction": False,
        "randomize_kp": False, "randomize_kd": False, "random_com": False,
        "randomize_leg_length": False, "push_robots": False,
    }
    override["noise_config"] = {"level": 0.0}
    override["control_config"] = {**override.get("control_config", {}), "action_smoothing": 0.0}
    override["commands"] = {**override.get("commands", {}), "resampling_time": 0.0}
    override["max_episode_seconds"] = 1000.0
    env_sec = cfg.get("env", {})
    if "terrain_curriculum" in env_sec:
        override.setdefault("terrain_curriculum", {})["enabled"] = False
    if "terrain_scan" in env_sec:
        override.setdefault("terrain_scan", {})["enabled"] = False
    env = registry.make(cfg["training"]["task_name"], sim_backend="mujoco",
                        num_envs=num_envs, env_cfg_override=override)
    env.set_autoreset(False)
    return env, cfg


def latest_ckpt(task_key: str) -> Path:
    import yaml

    cfg = yaml.safe_load(open(ROOT / TASK_CONF[task_key]))
    task = cfg["training"]["task_name"]
    best: Path | None = None
    best_key = ("", -1)
    for algo in ("ppo", "cpo", "np3o"):
        base = ROOT / f"logs/rsl_rl_{algo}" / task
        if not base.exists():
            continue
        for ck in base.rglob("*.pt"):
            run = ck.parent.name
            if ck.name == "policy.pt":
                it = max((int(p.name.split("_")[1].split(".")[0]) for p in ck.parent.glob("model_*.pt")), default=0) - 1
            else:
                it = int(ck.name.split("_")[1].split(".")[0])
            if (run, it) > best_key:
                best, best_key = ck, (run, it)
    return best


def load_policy(task_key: str, ckpt: Path):
    from tools.xqrobotwl.verify_jump import load_actor

    env, _cfg = build_env(task_key, 1)
    obs_dim = int(env.obs_groups_spec["obs"])
    na = int(env._backend._model.nu)
    ckpt_obj = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    act_state = ckpt_obj.get("actor_state_dict", {})
    hidden = [act_state[f"mlp.{i}.weight"].shape[0] for i in range(0, 40, 2)
              if f"mlp.{i}.weight" in act_state][:-1]
    pol = load_actor(str(ckpt), obs_dim, na, hidden)
    if any(k.startswith("obs_normalizer.") for k in act_state):
        from rsl_rl.modules.normalization import EmpiricalNormalization
        normalizer = EmpiricalNormalization(obs_dim)
        nk = {k.replace("obs_normalizer.", ""): v for k, v in act_state.items()
              if k.startswith("obs_normalizer.")}
        normalizer.load_state_dict(nk)
        normalizer.eval()
        _raw = pol

        def pol(x, _r=_raw, _n=normalizer):
            return _r(_n(x))
    env.close()
    return pol


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASK_CONF))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--sim_time", type=float, default=10.0)
    args = ap.parse_args()

    ckpt = latest_ckpt(args.task)
    if ckpt is None:
        print(f"[{args.task}] ❌ 无 checkpoint"); return 1
    pol = load_policy(args.task, ckpt)
    env, _ = build_env(args.task, 1)
    env.init_state()
    cmd_dim = int(env.state.info["commands"].shape[1])
    dt = float(env._cfg.ctrl_dt)
    sim_steps = int(args.sim_time / dt)

    all_rec = []
    with torch.no_grad():
        for ep in range(args.episodes):
            env.reset(np.arange(env.num_envs, dtype=np.int32))
            x0, y0 = env._backend.get_base_pos()[0, :2]
            yaw0 = None
            z_rec, upz_rec, lin_rec, yaw_rec = [], [], [], []
            active = True
            for step in range(sim_steps):
                obs = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
                a = pol(obs).numpy()[0]
                cmd = np.zeros(cmd_dim)
                if cmd_dim >= 5:
                    cmd[4] = 0.518
                env.state.info["commands"][:] = np.tile(cmd, (env.num_envs, 1))
                env._update_commands(env.state.info)
                st = env.step(np.asarray(a)[None, :])
                backend = env._backend
                up = np.asarray(backend.get_sensor_data("upvector"), dtype=np.float64)[0]
                gyro = np.asarray(backend.get_sensor_data("gyro"), dtype=np.float64)[0]
                lin = np.asarray(backend.get_sensor_data("local_linvel"), dtype=np.float64)[0]
                base_z = float(backend.get_base_pos()[0, 2])
                z_rec.append(base_z); upz_rec.append(float(up[2]))
                lin_rec.append(float(np.hypot(lin[0], lin[1])))
                yaw_rec.append(float(gyro[2]))
                if yaw0 is None:
                    yaw0 = float(np.arctan2(backend.get_base_quat()[0][3] if False else 0, 1.0))
                if st.terminated[0] or st.truncated[0]:
                    active = False
                    break
            x1, y1 = env._backend.get_base_pos()[0, :2]
            drift = float(np.hypot(x1 - x0, y1 - y0))
            yaw_cum = float(np.sum(np.abs(np.asarray(yaw_rec))) * dt)  # 近似累计 (取绝对值)
            z_mean = float(np.mean(z_rec)) if z_rec else 0.0
            z_std = float(np.std(z_rec)) if z_rec else 0.0
            upz = float(np.mean(upz_rec)) if upz_rec else 0.0
            lin_mean = float(np.mean(lin_rec)) if lin_rec else 0.0
            all_rec.append((active, z_mean, z_std, upz, drift, yaw_cum, lin_mean))
            print(f"[{args.task}] ep{ep}: {'站住' if active else '倒'} z={z_mean:.3f}±{z_std:.3f} "
                  f"upz={upz:.3f} 漂移={drift:.2f}m yaw累计={yaw_cum:.1f}° 微动={lin_mean:.3f}m/s")

    ok = [r for r in all_rec if r[0]]
    if ok:
        z = np.mean([r[1] for r in ok]); zs = np.mean([r[2] for r in ok])
        up = np.mean([r[3] for r in ok]); dr = np.mean([r[4] for r in ok])
        yaw = np.mean([r[5] for r in ok]); mv = np.mean([r[6] for r in ok])
        h_ok = abs(z - 0.52) <= 0.05
        dr_ok = dr < 0.5
        yaw_ok = yaw < 30
        mv_ok = mv < 0.2
        print(f"[{args.task}] 姿态: z={z:.3f}[阈±0.05]{'✅' if h_ok else '❌'} upz={up:.3f} "
              f"漂移={dr:.2f}m[阈<0.5]{'✅' if dr_ok else '❌'} yaw={yaw:.1f}°[阈<30°]{'✅' if yaw_ok else '❌'} "
              f"微动={mv:.3f}[阈<0.2]{'✅' if mv_ok else '❌'}")
    else:
        print(f"[{args.task}] 无法稳定站立, 无姿态数据")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
