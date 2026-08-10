# [09] play_interactive 窗口施力回灌 (Ctrl 拖动力真实作用于仿真)

**日期**: 2026-08-10
**来源**: 用户反馈 eval viewer 中双击选中 body 后 Ctrl+右键施力无效
**关联**: [[08_port_fall_recovery_branch]]

---

## 问题描述

`play_interactive.py` 的 MuJoCo 窗口里,双击选中机身后 Ctrl+右键拖动施加力,**机器人纹丝不动**。

## 根因分析

- 按键本身是对的 (mujoco_mpc GUI 文档: 双击选中, Ctrl+右键=力 / Ctrl+左键=力矩)
- 真正原因: 窗口的 `viz_data` 是**只读显示镜像** — 真实物理在 RL env 里 `advance()` 步进, 每帧代码把 env 状态 `mj_setState` 拷进 viz_data 再 `viewer.sync()`
- 窗口施力只写进镜像的 `xfrc_applied` (形状 `(nbody, 6)`, 实测 mjSTATE_FULLPHYSICS 不清零它), **但永远进不了 env 的真实仿真** → 无效

## 解决方案

`play_interactive.py` 新增 `_apply_viewer_forces()`:
1. 读 `viz_data.xfrc_applied` 找到被施力的 body
2. 按 body **名称**映射到 env 物理模型 (viewer 模型与 env 模型 body id 可能不同, 实测本任务恰好一致但按名映射更稳)
3. 经 `backend.apply_body_wrench()` 回灌进 `_pending_xfrc_applied`, 下一步 env step 时真实作用于物理

主循环 `advance()` 前调用。支持多 body 同时施力, 与 DR 推力/CPO 辅助力同一通道。

## 修改文件

- `scripts/play/play_interactive.py`: 新增 `_apply_viewer_forces()` + 主循环调用 + 启动提示

## 验证方法

- 单元验证: 对 `base_link` 模拟 viewer 施力 [50,0,0,0,0,0] → `_pending_xfrc_applied[6]=50` (env_id=1 ×6), 多 body 同时正确
- body 名称映射实测: viewer 模型与 env 模型 nbody 均为 10, base_link/wheel id 一致
- eval 冒烟: viewer 正常打开, 显示 "Force feedback ENABLED" 提示
- ruff format + lint 通过

## 后续计划

- [ ] 用户实际验证 (eval 中推机器人做扰动测试)
- [ ] 提交 (已提交 d8cbd7e)
