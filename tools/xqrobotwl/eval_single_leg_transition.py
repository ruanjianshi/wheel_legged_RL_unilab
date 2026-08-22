"""Evaluate standing -> triggered single-leg balance -> persistent hold.

The task starts from the normal two-wheel standing reset. ``sl_trigger`` is
kept off during the settle window and then latched on, matching the interactive
H-key behavior.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from unilab.base import registry
from unilab.base.registry import ensure_registries
from unilab.envs.locomotion.xqrobotwl import single_leg as single_leg_task

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "conf/ppo/task/xqrobotwl_single_leg_flat/mujoco.yaml"


class ActorMLP(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 8),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs)


def main() -> None:
    parser = argparse.ArgumentParser(description="评估站立按键触发的单腿平衡")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--settle-steps", type=int, default=100)
    parser.add_argument("--hold-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--support-roll-bias",
        type=float,
        default=0.0,
        help="诊断用：触发后叠加到支撑腿 hip-roll policy action",
    )
    parser.add_argument(
        "--fold-knee",
        type=float,
        default=None,
        help="诊断用：覆盖自由腿折膝目标 (rad)",
    )
    parser.add_argument(
        "--free-roll",
        type=float,
        default=None,
        help="诊断用：覆盖自由腿 hip-roll 配重目标 (rad)",
    )
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.fold_knee is not None:
        single_leg_task._FOLD_KNEE = float(args.fold_knee)
    if args.free_roll is not None:
        single_leg_task._FREE_LEG_ROLL_INIT = float(args.free_roll)

    ensure_registries()
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["reward"]["start_in_balance"] = False
    cfg["reward"]["sl_warmup_iters"] = 0
    cfg["env"]["commands"]["resampling_time"] = 0.0
    override = {"reward_config": cfg["reward"], **cfg["env"]}
    env = registry.make(
        "XqRobotWLSingleLegFlat",
        sim_backend="mujoco",
        num_envs=args.num_envs,
        env_cfg_override=override,
    )
    env.init_state()
    obs = env.state.obs

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    policy = ActorMLP(obs["obs"].shape[1])
    policy.load_state_dict(
        {
            key: value
            for key, value in checkpoint["actor_state_dict"].items()
            if key.startswith("mlp.")
        }
    )
    policy.eval()

    total_steps = args.settle_steps + args.hold_steps
    active = np.ones(args.num_envs, dtype=bool)
    first_done = np.full(args.num_envs, total_steps, dtype=np.int32)
    first_single = np.full(args.num_envs, -1, dtype=np.int32)
    balanced = np.zeros(args.num_envs, dtype=bool)
    single_samples = np.zeros(args.num_envs)
    support_contact_sum = np.zeros(args.num_envs)
    free_off_sum = np.zeros(args.num_envs)
    alignment_sum = np.zeros(args.num_envs)
    height_sum = np.zeros(args.num_envs)
    up_ref = np.array(
        [0.0, np.sin(single_leg_task._ROLL_REF_RAD), np.cos(single_leg_task._ROLL_REF_RAD)]
    )

    for step in range(total_steps):
        trigger = 1.0 if step >= args.settle_steps else 0.0
        env.state.info["commands"][:, :] = 0.0
        env.state.info["commands"][:, 4] = trigger
        with torch.no_grad():
            action = policy(torch.as_tensor(obs["obs"], dtype=torch.float32)).numpy()
        if trigger > 0.5:
            action[:, 3] += args.support_roll_bias
        state = env.step(action.astype(np.float64))
        fsm = np.asarray(state.info["fsm_state"])
        contact = np.asarray(state.info["wheel_contact"])
        gravity = np.asarray(env._backend.get_sensor_data(env._cfg.sensor.upvector))
        height = np.asarray(env._backend.get_base_pos())[:, 2]
        in_single = (fsm == 1) & active
        new_single = in_single & (first_single < 0)
        first_single[new_single] = step
        single_samples[in_single] += 1.0
        support_contact_sum[in_single] += contact[in_single, 1]
        free_off_sum[in_single] += 1.0 - contact[in_single, 0]
        alignment_sum[in_single] += gravity[in_single] @ up_ref
        height_sum[in_single] += height[in_single]
        balanced |= np.asarray(state.info["balance_completed"]) & active

        done = (state.terminated | state.truncated) & active
        first_done[done] = step + 1
        active[done] = False
        obs = state.obs

    count = np.maximum(single_samples, 1.0)
    reached = first_single >= 0
    print("=" * 64)
    print(f"checkpoint: {args.ckpt}")
    print(
        f"protocol: stand {args.settle_steps / 100:.1f}s -> H ON -> hold {args.hold_steps / 100:.1f}s"
    )
    print("=" * 64)
    print(f"站立后进入 FSM 单腿态: {reached.mean() * 100:.1f}%")
    print(f"达到严格连续平衡判据: {balanced.mean() * 100:.1f}%")
    print(f"完整跑完测试: {(first_done == total_steps).mean() * 100:.1f}%")
    print(f"平均存活: {first_done.mean() / 100:.2f}s / {total_steps / 100:.2f}s")
    print(f"单腿态支撑轮接触率: {(support_contact_sum / count).mean() * 100:.1f}%")
    print(f"单腿态自由轮离地率: {(free_off_sum / count).mean() * 100:.1f}%")
    print(f"单腿态姿态对齐: {(alignment_sum / count).mean():.3f}")
    print(f"单腿态平均高度: {(height_sum / count).mean():.3f}m")
    env.close()


if __name__ == "__main__":
    main()
