# [32] shell 目录按任务细分重组

**日期**: 2026-08-10
**来源**: 用户要求 "shell 整理, 让分类更加细一点"; 经 AskUserQuestion 确认按任务维度分类
**关联**: [[31_strip_repo_to_two_robots]]

---

## 问题描述

整理后的 shell 只有「机器人」一层 (xqrobotV2/xqrobotwl 平铺), 训练/评估/工具脚本
混在一起, 分类粒度不够细。

## 用户决策

按任务分类: `机器人/任务(flat|rough|jump|stairs|toe_walk|tools)/脚本`, 任务内
train/eval 脚本并存。

## 解决方案

新结构 `shell/<robot>/<task>/{train,eval}_*.sh`:
- xqrobotV2: `flat/` (ppo_flat), `rough/` (ppo_rough), `jump/` (ppo_jump_flat),
  `stairs/` (np3o_stairs), `toe_walk/` (ppo_toe_walk_flat), `tools/` (tensorboard,
  xqrobotV2_play, xqrobotV2_tb)
- xqrobotwl: `flat/` (ppo_flat), `rough/` (ppo_rough), `jump/` (ppo_jump_flat,
  ppo_jump_srl, ppo_jump_vmc, ppo_jump_srl_vmc, eval_jump_srl), `stairs/`
  (np3o_stairs), `toe_walk/` (ppo_toe_walk), `tools/` (export_onnx, play_policy,
  play_viser)
- 删除 3 个与新结构重复的通用训练入口: `shell/train/{ppo_mujoco,ppo_motrix,
  xqrobotV2_ppo}.sh`
- 通用工具从 `shell/eval/`、`shell/train/` 移入 `shell/<robot>/tools/`
- 更新所有脚本内部自引用注释 + 外部引用 (AGENTS.md, go.sh, jump_management/)

## 修改文件

- 28 个 shell 脚本 git mv 重新归类, 更新内部用法注释路径
- 删除: shell/train/{ppo_mujoco,ppo_motrix,xqrobotV2_ppo}.sh
- 移动: shell/eval/{export_onnx,play_policy,play_viser} → xqrobotwl/tools/
  shell/eval/{xqrobotV2_play,xqrobotV2_tb} → xqrobotV2/tools/
- 引用更新: AGENTS.md (shell 路径), go.sh, jump_management/{README,
  training/train_all.sh, train_ablation.sh, train_srl_full.sh}

## 验证

- 30 个脚本全部 `bash -n` 语法通过
- 无残留旧路径引用 (`shell/eval/`, `shell/train/`)
- 无空目录残留

## 后续计划

- [x] shell 按任务细分
- [x] 重复训练入口删除
- [x] 工具脚本归类到 tools/
- [x] 引用更新
- [ ] 提交 (不推送)
