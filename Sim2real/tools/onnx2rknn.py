#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX模型转RKNN模型工具脚本

本脚本用于将ONNX格式的深度学习模型转换为Rockchip RKNN格式，
适用于Rockchip系列芯片（如RK3588s）的推理部署。

转换流程：
1. 创建RKNN对象
2. 加载ONNX模型
3. 构建RKNN模型
4. 导出并保存RKNN模型
5. 释放资源

使用前需配置：
- 修改ONNX模型路径（第34行）
- 修改输出目录路径（第49行）
- 根据目标平台调整配置（第32行）
"""

from rknn.api import RKNN
import os

if __name__ == "__main__":
    platform = "rk3588s"

    """step 1: create RKNN object"""
    rknn = RKNN()

    """step 2: load the .onnx model"""
    rknn.config(target_platform="rk3588s")
    print("--> Loading model")
    # 固定输入尺寸: batch=1, obs=270 (45×6帧堆叠)
    inputs = ["input"]
    input_size_list = [[1, 270]]
    ret = rknn.load_onnx("/home/hightorque/sim2real_master-master/policy_htdw_4438_himloco.onnx",
                          inputs=inputs, input_size_list=input_size_list)
    if ret != 0:
        print("load model failed")
        exit(ret)
    print("done")

    """step 3: building model"""
    print("-->Building model")
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print("build model failed")
        exit()
    print("done")

    """step 4: export and save the .rknn model"""
    OUT_DIR = "/home/hightorque/sim2real_master-master"
    RKNN_MODEL_PATH = "{}/policy_from_onnx.rknn".format(OUT_DIR)
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
    print("--> Export RKNN model: {}".format(RKNN_MODEL_PATH))
    ret = rknn.export_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        print("Export rknn model failed.")
        exit(ret)
    print("done")

    """step 5: release the model"""
    rknn.release()