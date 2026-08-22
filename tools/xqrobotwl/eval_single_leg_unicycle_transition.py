"""Evaluate standing -> H-key -> sideways single-wheel balance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from tools.xqrobotwl.eval_single_leg_unicycle import ActorMLP  # noqa: E402
from unilab.base import registry  # noqa: E402
from unilab.training import ensure_registries  # noqa: E402

CONFIG = ROOT / "conf/ppo/task/xqrobotwl_single_leg_unicycle/mujoco.yaml"
UP_REF = np.array([0.0, -1.0, 0.0])
XVEC_REF = np.array([1.0, 0.0, 0.0])


def main() -> None:
    parser = argparse.ArgumentParser(description="评估站立按键切换独轮车式")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--num-envs", type=int, default=20)
    parser.add_argument("--settle-steps", type=int, default=100)
    parser.add_argument("--hold-steps", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ensure_registries()
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["reward"]["start_in_unicycle"] = False
    cfg["env"]["commands"]["resampling_time"] = 0.0
    override = {"reward_config": cfg["reward"], **cfg["env"]}
    env = registry.make(
        "XqRobotWLSingleLegUnicycle",
        sim_backend="mujoco",
        num_envs=args.num_envs,
        env_cfg_override=override,
    )
    try:
        env.init_state()
        checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        policy = ActorMLP(env.obs_groups_spec["obs"])
        policy.load_state_dict(
            {
                key: value
                for key, value in checkpoint["actor_state_dict"].items()
                if key.startswith("mlp.")
            }
        )
        policy.eval()

        total = args.settle_steps + args.hold_steps
        active = np.ones(args.num_envs, dtype=bool)
        first_done = np.full(args.num_envs, total, dtype=np.int32)
        reached = np.zeros(args.num_envs, dtype=bool)
        physical = np.zeros(args.num_envs, dtype=bool)
        hold_samples = np.zeros(args.num_envs)
        align_sum = np.zeros(args.num_envs)
        support_sum = np.zeros(args.num_envs)
        free_sum = np.zeros(args.num_envs)
        right_z_sum = np.zeros(args.num_envs)
        unicycle_dof_sum = np.zeros((args.num_envs, 6))
        max_right_z = np.full(args.num_envs, -np.inf)
        max_roll_align = np.full(args.num_envs, -np.inf)
        obs = env.state.obs

        for step in range(total):
            env.state.info["commands"][:, :] = 0.0
            env.state.info["commands"][:, 4] = float(step >= args.settle_steps)
            with torch.no_grad():
                action = policy(torch.as_tensor(obs["obs"], dtype=torch.float32)).numpy()
            state = env.step(action.astype(np.float64))
            obs = state.obs
            fsm = np.asarray(env._mode_state)
            up = np.asarray(env._backend.get_sensor_data("upvector"))
            basex = np.asarray(env._backend.get_sensor_data("basexvector"))
            left_z = np.asarray(env._backend.get_sensor_data("left_wheel_world_pos"))[:, 2]
            right_z = np.asarray(env._backend.get_sensor_data("right_wheel_world_pos"))[:, 2]
            max_right_z[active] = np.maximum(max_right_z[active], right_z[active])
            max_roll_align[active] = np.maximum(max_roll_align[active], up[active] @ UP_REF)
            goal = (
                (up @ UP_REF > 0.90)
                & (basex @ XVEC_REF > 0.90)
                & (left_z < 0.14)
                & (right_z > 0.16)
            )
            reached |= (fsm == 1) & active
            physical |= goal & active
            sample = (fsm == 1) & active
            hold_samples[sample] += 1.0
            align_sum[sample] += up[sample] @ UP_REF
            support_sum[sample] += (left_z[sample] < 0.14).astype(np.float64)
            free_sum[sample] += (right_z[sample] > 0.16).astype(np.float64)
            right_z_sum[sample] += right_z[sample]
            unicycle_dof_sum[sample] += env.get_dof_pos()[sample, :6]
            done = (state.terminated | state.truncated) & active
            first_done[done] = step + 1
            active[done] = False

        count = np.maximum(hold_samples, 1.0)
        print("=" * 68)
        print(f"checkpoint: {args.ckpt}")
        print(
            f"protocol: stand {args.settle_steps / 100:.1f}s -> H ON -> "
            f"hold {args.hold_steps / 100:.1f}s"
        )
        print("=" * 68)
        print(f"进入独轮 FSM: {reached.mean() * 100:.1f}%")
        print(f"达到真实独轮姿态: {physical.mean() * 100:.1f}%")
        print(f"完整跑完: {(first_done == total).mean() * 100:.1f}%")
        print(f"平均存活: {first_done.mean() / 100:.2f}s / {total / 100:.2f}s")
        print(f"独轮态姿态对齐: {(align_sum / count).mean():.3f}")
        print(f"左支撑轮贴地率: {(support_sum / count).mean() * 100:.1f}%")
        print(f"右自由轮离地率: {(free_sum / count).mean() * 100:.1f}%")
        print(f"独轮态右轮平均高度: {(right_z_sum / count).mean():.3f} m")
        print(f"独轮态腿角均值: {np.mean(unicycle_dof_sum / count[:, None], axis=0)}")
        print(f"过渡期右轮最大高度: {np.mean(max_right_z):.3f} m")
        print(f"过渡期最佳横滚对齐: {np.mean(max_roll_align):.3f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
