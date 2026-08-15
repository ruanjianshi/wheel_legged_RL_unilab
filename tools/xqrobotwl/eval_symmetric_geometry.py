#!/usr/bin/env python3
"""对称几何下重跑 RL 任务模型评估 — 几何对称化后的影响检查.

每个任务构建对称几何 env, 加载最新 checkpoint, 跑固定指令 episodes, 报存活率。

用法:
  uv run python tools/xqrobotwl/eval_symmetric_geometry.py --task walk_flat
  uv run python tools/xqrobotwl/eval_symmetric_geometry.py --task walk_flat --episodes 5 --sim_time 10

任务 key → 配置: walk_flat/walk_rough/toe_walk/jump/jump_srl/jump_vmc/backflip/
                 single_leg_flat/single_leg_move/single_leg_unicycle/fall_recovery/stairs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import yaml  # noqa: E402

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
# 各任务默认前向指令 (vx) — 行走类用前向, 其余零命令
DEFAULT_VX = {
    "walk_flat": 0.5, "walk_rough": 0.4, "toe_walk": 0.4,
    "single_leg_move": 0.3, "single_leg_unicycle": 0.3, "stairs": 0.4,
}
CMD_DIM = {"walk_rough": 4, "jump_vmc": 4, "jump_srl_vmc": 4, "backflip": 4}


def build_env(task_key: str, num_envs: int = 1):
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
    # ★ 评估期间禁用 episode 截断 (否则 max_episode_seconds=10 会在 sim_time 边界误判跌倒)
    override["max_episode_seconds"] = 1000.0
    # ★ terrain 字段仅当任务配置声明时覆盖 (jump_vmc 等平地任务无 terrain_curriculum)
    env_sec = cfg.get("env", {})
    if "terrain_curriculum" in env_sec:
        override.setdefault("terrain_curriculum", {})["enabled"] = False
    if "terrain_scan" in env_sec:
        override.setdefault("terrain_scan", {})["enabled"] = False
    env = registry.make(cfg["training"]["task_name"], sim_backend="mujoco",
                        num_envs=num_envs, env_cfg_override=override)
    env.set_autoreset(False)
    return env, cfg


def _iter_of(path: Path) -> int:
    """model_N.pt → N; policy.pt → 一个很大的占位 (视为最完整)."""
    m = path.name
    if m == "policy.pt":
        return 10**9
    try:
        return int(m.split("_")[1].split(".")[0])
    except Exception:
        return -1


def latest_ckpt(task_key: str) -> Path:
    cfg = yaml.safe_load(open(ROOT / TASK_CONF[task_key]))
    task = cfg["training"]["task_name"]
    best: Path | None = None
    best_key = ("", -1)
    for algo in ("ppo", "cpo", "np3o"):
        base = ROOT / f"logs/rsl_rl_{algo}" / task
        if not base.exists():
            continue
        for ck in base.rglob("*.pt"):
            run = ck.parent.name  # 运行目录名 (时间戳, 可排序)
            if ck.name == "policy.pt":
                # policy.pt 视为该 run 最高训练迭代 (优先级低于 model_N.pt 同迭代)
                it = max((_iter_of(s) for s in ck.parent.glob("model_*.pt")), default=0) - 1
            else:
                it = _iter_of(ck)
            if (run, it) > best_key:
                best, best_key = ck, (run, it)
    return best


def hidden_dims(task_key: str) -> list[int]:
    cfg = yaml.safe_load(open(ROOT / TASK_CONF[task_key]))
    h = cfg.get("algo", {}).get("policy", {}).get("actor_hidden_dims")
    return h if h else [512, 512, 256, 128]


def main() -> int:
    ap = argparse.ArgumentParser(description="对称几何下重跑 RL 模型评估")
    ap.add_argument("--task", required=True, choices=list(TASK_CONF))
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--sim_time", type=float, default=10.0)
    ap.add_argument("--num_envs", type=int, default=1)
    args = ap.parse_args()

    from tools.xqrobotwl.verify_jump import load_actor

    ckpt = latest_ckpt(args.task)
    if ckpt is None:
        print(f"[{args.task}] ❌ 无 checkpoint"); return 1
    env, cfg = build_env(args.task, args.num_envs)
    na = int(env._backend._model.nu)  # 执行器数 (8)
    obs_dim = int(env.obs_groups_spec["obs"])
    # 兼容 dict (model_N.pt) 与 TorchScript (policy.pt) 两种格式
    ckpt_obj = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    if isinstance(ckpt_obj, dict):
        # ★ 从 checkpoint mlp 权重形状推导 hidden dims (比 config 可靠, stairs/np3o 无 algo.policy 段)
        act_state = ckpt_obj.get("actor_state_dict", {})
        hidden = [
            act_state[f"mlp.{i}.weight"].shape[0]
            for i in range(0, 40, 2)
            if f"mlp.{i}.weight" in act_state
        ][:-1]  # 末层是动作输出, 非 hidden
        if not hidden:
            hidden = hidden_dims(args.task)
        pol = load_actor(str(ckpt), obs_dim, na, hidden)
        # ★ obs_normalizer (walk_rough/stairs 用 empirical_normalization=true)
        if any(k.startswith("obs_normalizer.") for k in act_state):
            from rsl_rl.modules.normalization import EmpiricalNormalization

            normalizer = EmpiricalNormalization(obs_dim)
            nk = {
                k.replace("obs_normalizer.", ""): v
                for k, v in act_state.items()
                if k.startswith("obs_normalizer.")
            }
            normalizer.load_state_dict(nk)
            normalizer.eval()
            _raw = pol

            def pol(x, _r=_raw, _n=normalizer):  # noqa: E731
                return _r(_n(x))
    else:
        pol = ckpt_obj  # TorchScript 模块 (policy.pt 已内置 normalizer), 直接调用

    env.init_state()
    cmd_dim = int(env.state.info["commands"].shape[1])  # 动态取命令维度 (flat=5D/rough=4D...)
    vx = DEFAULT_VX.get(args.task, 0.0)
    dt = float(env._cfg.ctrl_dt)
    survivals = []
    with torch.no_grad():
        for ep in range(args.episodes):
            env.reset(np.arange(env.num_envs, dtype=np.int32))  # ★ 站姿复位, 有效 obs
            active = True
            steps = 0
            while steps * dt < args.sim_time and active:
                obs = torch.tensor(env.state.obs["obs"], dtype=torch.float32)
                a = pol(obs).numpy()[0]
                # 注入命令 (每步固定 vx); 5D 任务补站姿高度 (cmd[4]=0 会让机器人下沉)
                cmd = np.zeros(cmd_dim)
                cmd[0] = vx
                if cmd_dim >= 5:
                    cmd[4] = 0.518
                env.state.info["commands"][:] = np.tile(cmd, (env.num_envs, 1))
                st = env.step(np.asarray(a)[None, :])
                active = not bool(st.terminated[0]) and not bool(st.truncated[0])
                steps += 1
            survived = steps * dt >= args.sim_time - 1e-6  # ★ 跑满 sim_time = 存活
            survivals.append(1.0 if survived else 0.0)
            print(f"[{args.task}] ep{ep}: {'存活' if survived else '跌倒'} "
                  f"{steps*dt:.1f}s/{args.sim_time:.0f}s")
    rate = np.mean(survivals)
    print(f"[{args.task}] 存活率 {rate*100:.0f}% ({int(np.sum(survivals))}/{args.episodes}) ckpt={ckpt}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
