# [34] 补全 xqrobotwl/jump 缺失的 VMC 验证脚本

**日期**: 2026-08-10
**来源**: 用户指出 shell/xqrobotwl/jump 少了两个验证脚本
**关联**: [[02_shell_reorganize_by_task]]

---

## 问题描述

shell 按任务细分后, `shell/xqrobotwl/jump/` 有 4 个训练脚本
(train_ppo_jump_flat/srl/vmc/srl_vmc) 但只有 2 个验证脚本
(eval_ppo_jump_flat/srl_flat), 缺 VMC 两个变体的验证脚本。

## 解决方案

参照 `eval_ppo_jump_srl_flat.sh` 模板创建两个验证脚本:
- `eval_ppo_jump_vmc.sh` → `--task xqrobotwl_jump_vmc_flat`
- `eval_ppo_jump_srl_vmc.sh` → `--task xqrobotwl_jump_srl_vmc_flat`

均支持 `--keyboard` (J=跳跃)、`policy|zero` 动作模式、`<run_id>` 参数。

## 修改文件

- 新建: `shell/xqrobotwl/jump/eval_ppo_jump_vmc.sh`,
  `shell/xqrobotwl/jump/eval_ppo_jump_srl_vmc.sh` (可执行)

## 验证

- `bash -n` 语法通过
- 任务名与训练脚本 env 注册名一致
- `shell/xqrobotwl/jump/` 现 4 train + 4 eval 一一对应

## 后续计划

- [x] 补全 vmc/srl_vmc 验证脚本
- [ ] 提交 (不推送)
