"""Expand a walk-flat PPO checkpoint for the single-leg FSM observations.

The single-leg actor/critic append 36 FSM/contact history features to the
walk-flat observations. Existing walk columns are copied exactly and the new
columns start at zero, so the initial policy preserves learned standing while
PPO learns to use the mode-specific features.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _copy_network(
    source: dict,
    target: dict,
    *,
    first_layer_key: str = "mlp.0.weight",
    source_frame: int = 0,
    target_frame: int = 0,
) -> None:
    for key, target_value in target.items():
        source_value = source.get(key)
        if not isinstance(source_value, torch.Tensor) or not isinstance(target_value, torch.Tensor):
            continue
        if key == first_layer_key:
            if source_value.shape[0] != target_value.shape[0]:
                raise ValueError(
                    f"incompatible first layer: {source_value.shape} -> {target_value.shape}"
                )
            if source_value.shape[1] > target_value.shape[1]:
                raise ValueError(
                    f"source observation is wider: {source_value.shape} -> {target_value.shape}"
                )
            target_value.zero_()
            if source_frame and target_frame:
                if source_value.shape[1] % source_frame != 0:
                    raise ValueError("source first layer is not divisible by source frame width")
                history = source_value.shape[1] // source_frame
                if target_value.shape[1] != history * target_frame:
                    raise ValueError(
                        "target first layer does not match history × target frame width"
                    )
                for frame in range(history):
                    source_slice = slice(frame * source_frame, (frame + 1) * source_frame)
                    target_slice = slice(frame * target_frame, frame * target_frame + source_frame)
                    target_value[:, target_slice].copy_(source_value[:, source_slice])
            else:
                target_value[:, : source_value.shape[1]].copy_(source_value)
        elif source_value.shape == target_value.shape and key != "distribution.std_param":
            target_value.copy_(source_value)


def main() -> None:
    parser = argparse.ArgumentParser(description="创建站立策略迁移的单腿 warm-start checkpoint")
    parser.add_argument("--walk", required=True, help="成熟 XqRobotWLWalkFlat checkpoint")
    parser.add_argument("--template", required=True, help="单腿任务 model_0 checkpoint")
    parser.add_argument("--out", required=True)
    parser.add_argument("--actor-source-frame", type=int, default=0)
    parser.add_argument("--actor-target-frame", type=int, default=0)
    parser.add_argument("--critic-source-frame", type=int, default=0)
    parser.add_argument("--critic-target-frame", type=int, default=0)
    args = parser.parse_args()

    walk = torch.load(args.walk, map_location="cpu", weights_only=False)
    output = torch.load(args.template, map_location="cpu", weights_only=False)
    _copy_network(
        walk["actor_state_dict"],
        output["actor_state_dict"],
        source_frame=args.actor_source_frame,
        target_frame=args.actor_target_frame,
    )
    _copy_network(
        walk["critic_state_dict"],
        output["critic_state_dict"],
        source_frame=args.critic_source_frame,
        target_frame=args.critic_target_frame,
    )
    output["iter"] = 0
    output["infos"] = {"warmstart_source": str(Path(args.walk).resolve())}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)
    print(f"warm-start checkpoint: {out_path}")


if __name__ == "__main__":
    main()
