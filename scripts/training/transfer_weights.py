"""跨任务权重迁移: 从 flat walk checkpoint 初始化 jump 训练模型"""
import argparse, sys, torch
from pathlib import Path

def transfer(src_path: str, dst_path: str, *, src_critic_dim: int, dst_critic_dim: int):
    src = torch.load(src_path, weights_only=False, map_location="cpu")
    dst = torch.load(dst_path, weights_only=False, map_location="cpu")

    # — actor —
    src_actor = src["actor_state_dict"]
    dst_actor = dst["actor_state_dict"]
    for key in dst_actor:
        if key not in src_actor:
            continue
        if "mlp.0.weight" in key:  # first layer: input dim differs
            src_w = src_actor[key]
            dst_w = dst_actor[key]
            common = min(src_w.shape[1], dst_w.shape[1])
            dst_w[:, :common] = src_w[:, :common]  # copy known features
            dst_actor[key] = dst_w
        elif "mlp.0.bias" in key:
            dst_actor[key] = src_actor[key].clone()
        else:
            dst_actor[key] = src_actor[key].clone()
    dst["actor_state_dict"] = dst_actor

    # — critic —
    src_critic = src["critic_state_dict"]
    dst_critic = dst["critic_state_dict"]
    for key in dst_critic:
        if key not in src_critic:
            continue
        if "mlp.0.weight" in key:
            src_w = src_critic[key]
            dst_w = dst_critic[key]
            common = min(src_w.shape[1], dst_w.shape[1])
            dst_w[:, :common] = src_w[:, :common]
            dst_critic[key] = dst_w
        elif "mlp.0.bias" in key:
            dst_critic[key] = src_critic[key].clone()
        else:
            dst_critic[key] = src_critic[key].clone()
    dst["critic_state_dict"] = dst_critic

    # Reset optimizer & iteration counter
    for key in ["optimizer_state_dict"]:
        if key in dst:
            del dst[key]
    dst["iter"] = 0

    # Save
    out_path = src_path + ".transfer_init.pt"
    torch.save(dst, out_path)
    print(f"Transferred weights saved to: {out_path}")
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source checkpoint (flat walk)")
    parser.add_argument("--dst", required=True, help="Destination checkpoint (jump init)")
    parser.add_argument("--src_critic_dim", type=int, default=324)
    parser.add_argument("--dst_critic_dim", type=int, default=342)
    args = parser.parse_args()
    transfer(args.src, args.dst, src_critic_dim=args.src_critic_dim, dst_critic_dim=args.dst_critic_dim)
