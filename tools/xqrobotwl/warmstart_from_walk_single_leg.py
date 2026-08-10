"""用 walk_flat 模型热启动单腿平衡训练 — 继承两轮平衡俯仰能力, 只学横滚

原理: single_leg actor 输入 333 = walk 的 297 (基础 obs) + 36 (单腿特征:
fsm_state/timer/wheel_contact×2, 每帧4维 × 9帧历史)。
把 walk 权重填入前 297 列, 后 36 列置零 → 策略初始会两轮站立(俯仰平衡),
单腿特征初始中性, RL 只需叠加横滚主动平衡 + 折腿过渡。

用法:
  uv run tools/xqrobotwl/warmstart_from_walk_single_leg.py
输出: logs/rsl_rl_ppo/XqRobotWLSingleLegFlat/warmstart_from_walk/model_0.pt
然后 resume: algo.load_run=<该文件绝对路径>
"""

from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
WALK_CKPT = ROOT / "logs/rsl_rl_ppo/XqRobotWLWalkFlat/2026-07-23_19-29-36_mujoco/model_9999.pt"
# 模板: 用最新 single_leg checkpoint 的结构 (critic/optimizer 等)
import glob

sl_runs = sorted(glob.glob(str(ROOT / "logs/rsl_rl_ppo/XqRobotWLSingleLegFlat/*/model_*.pt")))
if not sl_runs:
    raise SystemExit("找不到 single_leg checkpoint, 请先跑一次训练生成模板")
SL_CKPT = sl_runs[-1]
OUT = ROOT / "logs/rsl_rl_ppo/XqRobotWLSingleLegFlat/warmstart_from_walk/model_0.pt"


def main() -> None:
    walk = torch.load(WALK_CKPT, map_location="cpu", weights_only=False)
    sl = torch.load(SL_CKPT, map_location="cpu", weights_only=False)

    w_actor = walk["actor_state_dict"]
    s_actor = sl["actor_state_dict"]

    # mlp.0.weight: (512, 297) → (512, 333), 前 297 列用 walk, 后 36 列置零
    w0 = w_actor["mlp.0.weight"]
    s0 = s_actor["mlp.0.weight"].clone()
    assert s0.shape[1] == w0.shape[1] + 36, f"expect 36 extra, got {s0.shape[1] - w0.shape[1]}"
    s0[:, : w0.shape[1]] = w0
    s0[:, w0.shape[1] :] = 0.0
    s_actor["mlp.0.weight"] = s0
    s_actor["mlp.0.bias"] = w_actor["mlp.0.bias"].clone()

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
        s_actor[k] = w_actor[k].clone()

    # 重置迭代
    sl["iteration"] = 0
    sl["iter"] = 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sl, OUT)
    print(f"✅ 热启动 checkpoint: {OUT}")
    print(f"   actor mlp.0: {w0.shape} → {s0.shape} (后 36 列置零)")
    print("   继承 walk 全部隐藏层权重 (两轮平衡俯仰能力)")
    print(f"   resume: algo.load_run={OUT}")


if __name__ == "__main__":
    main()
