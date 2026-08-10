# [01] 仓库瘦身: 移除除 xqrobotwl/xqrobotV2 外所有机器人

**日期**: 2026-08-10
**来源**: 用户要求 "把除了我自己的两个机器人xqrobotwl,xqrobotV2,其他机器人都去掉,整理和分类好每个文件"
**关联**: [[30_paper_figs_nature_style]]

---

## 问题描述

仓库含 11 个其他机器人 (allegro_hand, g1, go1, go2, go2_arm, go2w, hfields,
sharpa_wave, smallHumanoidRobot, stewart, x2) 及其资产/env/配置/测试/脚本/文档,
与小论文 (武汉科技大学研究生, 仅用 xqrobotwl/xqrobotV2 轮腿机器人) 无关。

## 用户决策 (AskUserQuestion 确认)

- **删除方式**: 永久删除 (git `8d2c042` 已提交, 可 checkout 恢复)
- **Sim2real/**: 保留不动 (918MB 纯其他机器人 ROS 子项目)
- **tests/**: 改写共享框架测试到 xq + 删除机器人专属测试
- **docs/README/notebook**: 一并清理
- **算法入口** (appo/offpolicy/hora_distill/ppo_him/mlx_ppo): 保留入口脚本 + config.yaml 目录 (不含任务子目录), 运行时不可用

## 执行摘要 (依赖顺序)

### 1. 删除资产/env/conf
- `assets/robots/{allegro_hand,g1,go1,go2,go2_arm,go2w,hfields,sharpa_wave,smallHumanoidRobot,stewart,x2}`, `motions/{g1,x2}`, `objects/sharpa_cylinder`
- `envs/locomotion/{g1,go1,go2,go2_arm,go2w,smallHumanoidRobot}`, 整包 `manipulation/`, `motion_tracking/`
- `conf/{ppo,appo,offpolicy,hora_distill,ppo_him}/task/` 非 xq 任务全部删除; 保留算法 `config.yaml` (appo/offpolicy/hora/ppo_him 不含任务子目录, 运行时不可用)

### 2. 编辑共享 src
- `registry.py`: `_DEFAULT_REGISTRY_PACKAGES` → 仅 `unilab.envs.locomotion`
- `envs/locomotion/__init__.py`: 仅 `xqrobotV2, xqrobotwl`
- `demo.py`: 仅留 `teaser`; `cli.py` help 示例 → xq; `support_matrix.py` 任务列表 → xq; 删 `tools/pull_assets.py` + pyproject 入口
- `conf/ppo/config*.yaml` 默认任务 → `xqrobotwl_walk_flat/mujoco`

### 3. 测试 (改写共享 + 删专属, ~30 文件)
- **删专属**: g1/go1/go2/go2w/go2_arm/sharpa/stewart/motion_tracking 测试 + appo/offpolicy/hora/mlx 集成测试 + `test_train_script_configs.py` (整体废弃)
- **改写共享**: sim_backend/sim_backend_smoke/batch_env_jacobian/randomization/reward_override/config_system/locomotion_params/reward_injection/xml_utils/mjspec/training_helpers/seed_contract/sim2sim_resolver/rsl_rl_runner/reward_injection_integration/env_configs/support_matrix/mujoco_only_tooling/visualization_entrypoints 等
- `conftest.py`: 5 个机器人 reward 夹具 → 1 个 `default_xq_reward_config` (XqRobotWLRewardConfig)
- **关键发现**: xq env 的 `reward_config` 是 dataclass 且 `__init__` 强制要求; flat env 用 5D commands (VMC 任务), registry.make 直接构造需传 `commands`/`domain_rand` override (镜像 task YAML)
- **修复 HEAD 既有失败**: `_load_script` 脚本路径 (`scripts/training/`, `scripts/play/`); `test_train_scripts.py` 从 128 失败 → 全绿 (删除 appo/offpolicy/hora 用例 + 修复 monkeypatch 泄漏); `test_visualization_entrypoints.py` 路径 + 默认任务
- **预存失败 (与 strip 无关)**: `test_nan_guard.py` 5 个测试 + `test_np_env.py` 1 个 — 测试用 `caplog` 但 NanGuard 走 `print()` stderr, 初始提交即失败

### 4. 脚本/shell/benchmark
- 删 `scripts/manip_loco/{3个go2_arm}`, `scripts/motion/**`, `scripts/deploy/**`, `play_smallHumanoid.py`, `sharpa_collect_grasps.sh`
- `visualize_task_env.py` 默认任务 → `XqRobotWLWalkFlat`; shell TASK → xq; 删 `appo_mujoco.sh`/`sac_mujoco.sh`; 修 `go.sh` 路径
- `benchmark/core/task_names.py` 重写为 xq 映射 (保留公开 API); 删 6 个机器人 benchmark 脚本; 修 postprocess/mj_step/forward_reset 任务 id

### 5. 文档 (受 doc_checks 强制, ~50 文件)
- 删机器人专属 sphinx 页 + g1 图片 + notebook; `docs/references/` 加入 doc_checks 跳过 (外部文献笔记)
- `doc_checks.py`: 支持 file+line 引用 (`:21-50`) + markdown 链接跳过代码围栏 (修复预存误报)
- 重写 4-tasks/locomotion/api_reference/envs/DR 表/support_matrix/quick_demo/0-index/deployment 页到 xq
- 修正 tutorial 04 (删 SAC/TD3/APPO 段)、05、06、08; README/README_zh 机器人表 → 两个 xq
- `.github/ISSUE_TEMPLATE/work_item.yml` 下拉 → XqRobotWL/XqRobotV2

## 验证

- `ruff format` + `ruff check` 全绿
- `uv run pytest tests/ -q` → **787 passed, 5 failed** (5 个均为预存 nan_guard 失败, 与 strip 无关, 见上)
- 慢测试 `-m slow` → 80 passed (rsl_rl_runner 修 DR override 后)
- `doc_checks.collect_doc_errors()` → 0
- 注册表 import 冒烟: `ensure_registries()` + demo registry = {teaser}

## 后续计划

- [x] 全部 11 个其他机器人移除
- [x] tests/ 改写+删除完成
- [x] 脚本/shell/benchmark 清理
- [x] docs/README 清理 (doc_checks 0 错误)
- [ ] 预存 `test_nan_guard.py` 5 失败 (caplog vs print 不匹配) — 需单独修复 (非本任务范围)
- [ ] 提交 (不推送)
