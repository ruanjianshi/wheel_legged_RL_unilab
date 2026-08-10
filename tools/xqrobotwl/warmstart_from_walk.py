"""用 walk_flat 模型热启动 backflip 训练 — 继承平衡能力, 只学翻转

原理: backflip actor 输入 324 = walk 的 297 (基础 obs) + 27 (翻转特征)。
把 walk 权重填入前 297 列, 后 27 列置零 → 策略初始会平衡, 翻转特征初始中性。

用法:
  uv run tools/xqrobotwl/warmstart_from_walk.py
输出: logs/rsl_rl_ppo/XqRobotWLBackflipFlat/warmstart_from_walk/model_0.pt
然后 resume: algo.load_run=<该文件绝对路径>
"""

from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
WALK_CKPT = ROOT / "logs/rsl_rl_ppo/XqRobotWLWalkFlat/2026-07-23_19-29-36_mujoco/model_9999.pt"
# 模板: 用最新 backflip checkpoint 的结构 (critic/optimizer 等)
import glob

bf_runs = sorted(glob.glob(str(ROOT / "logs/rsl_rl_ppo/XqRobotWLBackflipFlat/*/model_*.pt")))
BF_CKPT = bf_runs[-1]
OUT = ROOT / "logs/rsl_rl_ppo/XqRobotWLBackflipFlat/warmstart_from_walk/model_0.pt"


def main() -> None:
    walk = torch.load(WALK_CKPT, map_location="cpu", weights_only=False)
    back = torch.load(BF_CKPT, map_location="cpu", weights_only=False)

    w_actor = walk["actor_state_dict"]
    b_actor = back["actor_state_dict"]

    # mlp.0.weight: (512, 297) → (512, 324), 前 297 列用 walk, 后 27 列置零
    w0 = w_actor["mlp.0.weight"]
    b0 = b_actor["mlp.0.weight"].clone()
    assert b0.shape[1] == w0.shape[1] + 27, f"expect 27 extra, got {b0.shape[1] - w0.shape[1]}"
    b0[:, : w0.shape[1]] = w0
    b0[:, w0.shape[1] :] = 0.0
    b_actor["mlp.0.weight"] = b0
    b_actor["mlp.0.bias"] = w_actor["mlp.0.bias"].clone()

    # 其余层 (512→512→256→128→8) 维度相同, 直接继承
    for k in [
        "mlp.2.weight",
        "mlp.2.bias",
        "mlp.4.weight",
        "mlp.4.bias",
        "mlp.6.weight",
        "mlp.6.bias",
        "mlp.8.weight",
        "mlp.8.bias",
    ]:
        b_actor[k] = w_actor[k].clone()

    # 重置迭代
    back["iteration"] = 0
    back["iter"] = 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(back, OUT)
    print(f"✅ 热启动 checkpoint: {OUT}")
    print(f"   actor mlp.0: {w0.shape} → {b0.shape} (后 27 列置零)")
    print("   继承 walk 全部隐藏层权重")
    print(f"   resume: algo.load_run={OUT}")


if __name__ == "__main__":
    main()
