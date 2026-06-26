#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boost 文件关节数据重排工具

本脚本用于重新排列 boost 文件中的关节数据顺序，主要用于 sim2real 项目中
机器人关节数据格式的转换。

功能说明：
- 解析 boost 格式的轨迹文件
- 将关节数据从 boost 文件顺序转换为电机控制顺序
- 映射关系：boost索引 -> 电机索引 = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6]
- 即：左右腿关节顺序反转（左腿6个关节倒序，右腿6个关节倒序）

关节定义（按boost文件顺序）：
0: l_hip_pitch_joint    (左髋俯仰)
1: l_hip_roll_joint     (左髋翻滚)
2: l_thigh_joint        (左大腿)
3: l_calf_joint         (左小腿)
4: l_ankle_pitch_joint  (左踝俯仰)
5: l_ankle_roll_joint   (左踝翻滚)
6: r_hip_pitch_joint    (右髋俯仰)
7: r_hip_roll_joint     (右髋翻滚)
8: r_thigh_joint        (右大腿)
9: r_calf_joint         (右小腿)
10: r_ankle_pitch_joint (右踝俯仰)
11: r_ankle_roll_joint  (右踝翻滚)

使用方式：
1. 修改第124-125行的输入输出文件路径
2. 运行脚本：python reorder_boost.py

输入：boost格式的轨迹文件（包含多组12维关节数据）
输出：重新排列后的boost文件
"""

import sys
import os
import re

def parse_boost_file(file_path):
    """解析boost文件，提取所有数据组"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 分割数据组
    parts = content.split(' 12 0 ')
    if len(parts) < 2:
        print("文件格式错误")
        return None, []
    
    # 第一部分是元数据
    metadata = parts[0]
    
    # 其余部分是数据组
    data_groups = []
    for i in range(1, len(parts)):
        data_part = parts[i].strip()
        if data_part:
            # 提取数值
            numbers = re.findall(r'-?\d+\.\d+e[+-]\d+', data_part)
            if len(numbers) == 12:
                data_group = [float(x) for x in numbers]
                data_groups.append(data_group)
    
    return metadata, data_groups

def reorder_data_group(original_data, map_index):
    """重新排列单个数据组"""
    reordered_data = [0.0] * 12
    for boost_idx, motor_idx in enumerate(map_index):
        reordered_data[motor_idx] = original_data[boost_idx]
    return reordered_data

def create_reordered_boost_file(original_file, new_file):
    """创建重新排列的boost文件"""
    
    # 映射关系：boost文件索引 -> 电机索引
    map_index = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6]
    
    # 关节名称（按boost文件顺序）
    joint_names = [
        "l_hip_pitch_joint", "l_hip_roll_joint", "l_thigh_joint", "l_calf_joint", "l_ankle_pitch_joint", "l_ankle_roll_joint",
        "r_hip_pitch_joint", "r_hip_roll_joint", "r_thigh_joint", "r_calf_joint", "r_ankle_pitch_joint", "r_ankle_roll_joint"
    ]
    
    # 解析原始文件
    metadata, data_groups = parse_boost_file(original_file)
    if metadata is None:
        return
    
    print(f"找到 {len(data_groups)} 组轨迹数据")
    
    # 重新排列所有数据组
    reordered_groups = []
    for i, group in enumerate(data_groups):
        print(f"\n第 {i+1} 组原始数据:")
        for j, (name, value) in enumerate(zip(joint_names, group)):
            print(f"  [{j}] {name}: {value}")
        
        reordered_group = reorder_data_group(group, map_index)
        reordered_groups.append(reordered_group)
        
        print(f"第 {i+1} 组重新排列后数据:")
        for motor_idx in range(12):
            joint_idx = map_index.index(motor_idx) if motor_idx in map_index else -1
            joint_name = joint_names[joint_idx] if joint_idx >= 0 else "unknown"
            print(f"  电机[{motor_idx}] {joint_name}: {reordered_group[motor_idx]}")
    
    # 创建新文件内容
    new_content = metadata + " 12 0 "
    for i, group in enumerate(reordered_groups):
        if i > 0:
            new_content += " 12 0 "
        new_content += " ".join([f"{x:.15e}" for x in group])
    
    # 写入新文件
    with open(new_file, 'w') as f:
        f.write(new_content)
    
    print(f"\n新文件已创建: {new_file}")

if __name__ == "__main__":
    original_file = "/home/hightorque/zy_workspace/sim2real_master/src/sim2real/way_point/way_point_backups_20250910153758.boost"
    new_file = "/home/hightorque/zy_workspace/sim2real_master/src/sim2real/way_point/way_point_backups_20250910153758_reorder.boost"
    
    if os.path.exists(original_file):
        create_reordered_boost_file(original_file, new_file)
    else:
        print(f"原始文件不存在: {original_file}")