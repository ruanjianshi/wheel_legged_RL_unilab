#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch (.pt / TorchScript) 模型转 ONNX 工具脚本

本脚本用于将 TorchScript 格式的 RL 策略模型（.pt）转换为 ONNX 格式，
作为后续 onnx2rknn.py 转换的前置步骤。

使用前需确认：
- 模型输入维度（默认 270 = 45×6 帧堆叠）
- 模型输出维度（默认 12）
"""

import torch
import os
import sys

if __name__ == "__main__":
    # ======== 配置区 ========
    PT_MODEL_PATH = "/home/hightorque/sim2real_master-master/policy_htdw_4438_himloco.pt"
    OUT_DIR = "/home/hightorque/sim2real_master-master"
    ONNX_MODEL_PATH = os.path.join(OUT_DIR, "policy_htdw_4438_himloco.onnx")

    # 模型输入输出维度（与训练/config一致）
    NUM_SINGLE_OBS = 45
    FRAME_STACK = 6
    NUM_OBS = NUM_SINGLE_OBS * FRAME_STACK  # 270
    NUM_ACTIONS = 12
    # ======================

    if not os.path.exists(PT_MODEL_PATH):
        print(f"Error: model not found at {PT_MODEL_PATH}")
        sys.exit(1)

    print(f"--> Loading TorchScript model: {PT_MODEL_PATH}")
    device = torch.device("cpu")
    model = torch.jit.load(PT_MODEL_PATH, map_location=device)
    model.eval()
    print("done")

    # 构造 dummy 输入
    dummy_input = torch.randn(1, NUM_OBS)

    # 验证模型输出
    with torch.no_grad():
        output = model(dummy_input)
        out_dim = output.shape[-1] if hasattr(output, 'shape') else len(output)
        print(f"Model input shape: {dummy_input.shape}")
        print(f"Model output shape: {output.shape}")

    if out_dim != NUM_ACTIONS:
        print(f"Warning: expected output dim={NUM_ACTIONS}, got {out_dim}")

    print(f"--> Exporting ONNX model: {ONNX_MODEL_PATH}")
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_MODEL_PATH,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )
    print("done")

    # 验证 ONNX 模型
    print("--> Verifying ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(ONNX_MODEL_PATH)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX model verified: {onnx_model.graph.name}")
    except ImportError:
        print("onnx package not installed, skipping verification")
    except Exception as e:
        print(f"ONNX verification warning: {e}")

    print(f"ONNX model saved to: {ONNX_MODEL_PATH}")
