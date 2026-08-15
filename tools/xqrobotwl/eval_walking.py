#!/usr/bin/env python3
"""行走类任务 (walk_flat/walk_rough/toe_walk) 对称几何评估 — §7.2/7.3/7.4 对照.

30s 命令表 (站 → 前 → 后), 测: 存活率 / vx_rmse / base_z 稳定性 / gyro_rms。
walk_flat 额外测侧移 Vy (§7.2 达标 Vy>0.25)。

用法:
  uv run python tools/xqrobotwl/eval_walking.py --task walk_flat [--episodes 3]
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
}
# 命令表: [(时长s, vx, vy)]  — 站2s → 前10s → 后10s → 停8s
PROFILE = {"walk_flat": [(2, 0.0, 0.0), (10, 0.5, 0.0), (10, -0.5, 0.0), (8, 0.0, 0.0)],
           "walk_rough": [(2, 0.0, 0.0), (14, 0.4, 0.0), (14, 0.0, 0.0)],
           "toe_walk": [(2, 0.0, 0.0), (14, 0.4, 0.0), (14, 0.0, 0.0)]}
PROFILE_SIDE = [(2, 0.0, 0.0), (12, 0.3, 0.3), (12, 0.0, 0.0)]  # walk_flat 侧移段


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
    ap.add_argument("--side", action="store_true", help="walk_flat 额外侧移段")
    args = ap.parse_args()

    ckpt = latest_ckpt(args.task)
    if ckpt is None:
        print(f"[{args.task}] ❌ 无 checkpoint"); return 1
    pol = load_policy(args.task, ckpt)
    env, cfg = build_env(args.task, 1)
    env.init_state()
    cmd_dim = int(env.state.info["commands"].shape[1])
    dt = float(env._cfg.ctrl_dt)
    profile = PROFILE[args.task]
    if args.side and args.task == "walk_flat":
        profile = PROFILE_SIDE
    sim_time = sum(d for d, _, _ in profile)

    results = []
    with torch.no_grad():
        for ep in range(args.episodes):
            env.reset(np.arange(env.num_envs, dtype=np.int32))
            v_rec, vx_cmd_rec, base_z_rec, gyro_rec, active, steps = [], [], [], [], True, 0
            while steps * dt < sim_time and active:
                t = steps * dt
                acc = 0.0
                for dur, vx, vy in profile:
                    if t < acc + dur:
                        cmd_vx, cmd_vy = vx, vy
                        break
                    acc += dur
                obs = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
                a = pol(obs).numpy()[0]
                cmd = np.zeros(cmd_dim)
                cmd[0] = cmd_vx
                if cmd_dim >= 3:
                    cmd[1] = cmd_vy
                if cmd_dim >= 5:
                    cmd[4] = 0.518
                env.state.info["commands"][:] = np.tile(cmd, (env.num_envs, 1))
                st = env.step(np.asarray(a)[None, :])
                # 本体线速度 (local_linvel)
                lin = np.asarray(env._backend.get_sensor_data("local_linvel"), dtype=np.float64)[0]
                gyro = np.asarray(env._backend.get_sensor_data("gyro"), dtype=np.float64)[0]
                base_z = float(env._backend.get_base_pos()[0, 2])
                v_rec.append(float(lin[0]))
                base_z_rec.append(base_z)
                gyro_rec.append(float(gyro[1]))
                if args.side:
                    vx_cmd_rec.append(cmd_vy)  # 侧移段测 vy 追踪
                else:
                    vx_cmd_rec.append(cmd_vx)
                active = not bool(st.terminated[0]) and not bool(st.truncated[0])
                steps += 1
            surv = steps * dt >= sim_time - 1e-6
            # 稳态追踪 RMSE (排除命令段首 2s 瞬态)
            n = len(v_rec)
            if n > 200:
                v, c = np.asarray(v_rec), np.asarray(vx_cmd_rec)
                vx_rmse = float(np.sqrt(np.mean((v[-min(n - 200, 2000):] - c[-min(n - 200, 2000):]) ** 2)))
            else:
                vx_rmse = float("nan")
            results.append((surv, steps * dt, vx_rmse, float(np.mean(gyro_rec)), float(np.mean(base_z_rec))))
            print(f"[{args.task}] ep{ep}: {'存活' if surv else '跌倒'} {steps*dt:.0f}s/{sim_time:.0f}s "
                  f"vx_rmse={vx_rmse:.3f} gyro_rms={results[-1][3]:.2f} base_z={results[-1][4]:.3f}")
    rate = np.mean([r[0] for r in results])
    avg = np.mean([r[1] for r in results])
    print(f"[{args.task}] 存活率 {rate*100:.0f}%  平均时长 {avg:.1f}s  "
          f"vx_rmse均值 {np.mean([r[2] for r in results if not np.isnan(r[2])]):.3f}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
