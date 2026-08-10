# [14] 八任务隔离审计 + README 更新 (反映 tools/ 迁移与八任务结构)

**日期**: 2026-08-10
**来源**: 用户指令 — "更新,然后检查,八个任务是否和开发规范一样,独立不干扰,可支持八个agent并行开发, README.md 也要更新"
**关联**: [[13_move_scripts_to_tools]] [[12_reorganize_per_updated_spec]]

---

## 审计结论: 八任务隔离 ✅ (与 CLAUDE.md §3 对齐)

对 8 个任务逐项核对 env / conf / shell / devlog / video / 任务脚本:

| # | 任务 | env | conf | shell | devlog | video |
|---|---|---|---|---|---|---|
| 1 | 平地滚动行走 | `joystick.py` | ppo/xqrobotwl_walk_flat | shell/xqrobotwl/flat | walk_flat | video/walk |
| 2 | 点足平地行走 | `toe_walk.py` | ppo/xqrobotwl_toe_walk_flat | toe_walk | toe_walk | video/toe_walk |
| 3 | 粗糙地形 | `rough.py` | ppo/xqrobotwl_walk_rough | rough | {walk_rough,rough} | video/rough |
| 4 | 跳跃 | `jump*.py`×5 | ppo/jump*_flat×5 | jump | jump | video/jump |
| 5 | 后空翻 | `backflip.py` | ppo/xqrobotwl_backflip_flat | backflip | backflip | video/backflip |
| 6 | 单腿平衡 | `single_leg*.py`×3 | ppo/single_leg*×3 | single_leg | single_leg | video/single_leg |
| 7 | 跌倒恢复 | `fall_recovery.py` | cpo/xqrobotwl_fall_recovery_flat | fall_recovery | fall_recovery | video/fall_recovery |
| 8 | 抬腿上台阶 | `stairs.py` | np3o/xqrobotwl_stairs | stairs | stairs | video/stairs |

**独立性核验点**:
- ✅ 每任务独立: env 文件 / conf 目录 / shell (train+eval) / devlog / video / 任务脚本
- ✅ registry 注册名唯一 (无重复 envcfg/env)
- ✅ 任务 yaml 无 cross-task `defaults` 耦合
- ✅ 训练日志隔离: `logs/rsl_rl_<algo>/<Task>/<timestamp>/` 每 run 独立
- ✅ 无裸类级可变属性 (dataclass `field(default_factory=…)` 安全模式)
- ✅ 并行训练为独立进程 (nohup), 无运行时状态竞争

**共享只读基座** (架构复用, 非可变状态): `joystick.py` (基座行走 env, 6 任务继承) /
`base.py` / `common/rewards.py` / 机器人 XML — 按 §3.2 不得被单 agent 独占修改。

**并行 agent 支持**: 8 agent 各自认领一任务 (独立工作区) 可行; 唯一约束是共享基座文件
并发编辑需协调 (一次只一个 agent 改 joystick.py 等共享文件)。

**规范与仓库的小出入** (均已知/无害):
- 规范 §3.1 把 stairs env/config 标"待建", 仓库已建成 (np3o 训练) — 仓库领先规范
- 规范 §3.1 视频列 `video/stair/`, 仓库用 `stairs/` 复数 (用户已拍板)
- shell 目录 `flat`/`rough` 对应任务 `walk_flat`/`walk_rough` (遗留别名, project_tree 有记录)

## README.md 更新内容

- 日期 → 2026-08-10; 项目树重画 (去 assess/notebook/go.sh, 加 tools/ thesis/, conf 按 algo×任务)
- 机器人任务表: XqRobotWL → 八大任务; XqRobotV2 标注旧版
- **新增 "XqRobotWL 八大任务" 隔离表** (env/conf/shell/devlog/video 6 列 + 并行 agent 说明)
- 移除已删除的 `assess/` 评估框架章节 → 替换为当前评估工具链 (eval_*/dump_pose_data/infer_pose_from_csv/render_*)
- 快速开始: shell 路径修正为 `shell/xqrobotwl/<task>/…`, TensorBoard 指向 XqRobotWLWalkFlat
- 清除死引用: assess/README.md, CONTRIBUTING.md (不存在)

## 验证方法

- `uv run pytest tests/scripts/test_check_docs.py` ✅ 19 passed
- grep 确认 README 无 `assess/ notebook/ go.sh shell/train_* CONTRIBUTING.md` 残留
- 八任务结构逐项比对 (env/conf/shell/devlog/video 均存在)

## 后续计划

- [ ] 提交 (用户自行 commit, 不推送)
