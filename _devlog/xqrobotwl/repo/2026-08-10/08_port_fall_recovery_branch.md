# [08] 移植 fall_recovery 分支 (后空翻/单腿平衡/跌倒恢复 + CPO)

**日期**: 2026-08-10
**来源**: 用户在另一台电脑基于同一仓库开发 `fall_recovery` 分支 (github.com/ruanjianshi/wheel_legged_RL_unilab),要求全部移植到本地
**关联**: [[04_launch_8_trainings_and_fix_path_obs_bugs]], [[07_thesis_folder_refactor_by_user_arch]]

---

## 分支关系

- 本地 main (a8eabbd) 与 `origin/fall_recovery` 共同祖先 = `38bcd16`
- fall_recovery 新增 2 提交: `b4b0d03` (跌倒恢复 FTSR+CPO + 后空翻/单腿平衡) + `62dd1f3` (backup/日志/技能)

## 移植内容

| 类别 | 内容 |
|------|------|
| 算法 | `src/unilab/algos/torch/cpo.py` (CPO 约束策略优化, 惩罚函数法, 继承 PPO) + `train_cpo.py` + `conf/cpo/` |
| 新 env | `backflip.py` / `fall_recovery.py` / `single_leg.py` / `single_leg_move.py` / `single_leg_unicycle.py` + 4 conf + `__init__.py` 13 注册 |
| 后端 | `apply_body_wrench` (mujoco 实现 / motrix 未支持) |
| 脚本 | `scripts/xqrobotwl/` 18 个 (feasibility/eval/render/warmstart) |
| shell | 8 个新脚本 → 归位到 backflip/fall_recovery/single_leg 子目录 + ROOT_DIR 3 级 |
| devlog | backflip(14)/single_leg(11)/fall_recovery(23) 60+ 条 + INDEX |
| 媒体 | `video/` 3038 文件 + `backup/` 补充 |

## 冲突解析

| 文件 | 处理 |
|------|------|
| `jump.py` | 保留本地 `anti_lazy` 奖励 + remote 删重复 `lean_forward` |
| `_devlog/INDEX.md` | 并集 (HEAD jump 条目 + remote backflip/fall_recovery 条目) |
| `play_smallHumanoid.py` | 保持删除 (机器人已移除) |
| `jump_management/` | 用户决定保持删除 |
| `.opencode/` | 用户决定跳过 |

## 现有任务零影响验证 (用户关切)

- 共享文件改动全部**行为中性** (见计划表): jump_srl 空格 / jump.py 重复key / xml 新增传感器 / backend 新增方法 / rsl_rl.py 门控转发 / play_interactive 门控 bypass
- remote 未动 `logs/` 与现有任务 conf; logs/ 在 .gitignore
- **8 个现有任务回归**: flat/rough/jump×4/stairs/toe_walk 全部 eval 冒烟通过, checkpoint 加载正常

## 验证结果

- ✅ 5 新 env + CPO import 通过
- ✅ 新任务端到端: backflip (PPO) + fall_recovery (CPO) 各跑 1 迭代成功
- ✅ 8 个现有任务 eval 回归通过
- ✅ doc_checks 19 passed (修复 CONTRIBUTING.md 引用 + 重新生成 support_matrix)
- ✅ ruff format + lint 通过
- ⚠️ `test_np_env.py::TestRewardSanitization` 1 例失败 — 历史遗留 (np_env.py 自 init 未改, 与移植无关)

## 提交记录 (未推送)

- `6ca6a3f` merge: 移植 fall_recovery
- `ca25dc0` refactor: 新 shell 脚本归位
- (doc 修复) fix: doc_checks + support matrix + format

## 后续计划

- [ ] 用户 review 合并 + 自行 commit (devlog/thesis 待提交项)
- [ ] 后空翻/单腿平衡/跌倒恢复 训练与验证 (用新 shell 脚本)
- [ ] 更新 thesis/ 专家状态 (experts/06/07/08)
