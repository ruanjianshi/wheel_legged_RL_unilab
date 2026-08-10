# [10] 按 CLAUDE.md 开发规范整理项目 (补基建 + 归位脚本)

**日期**: 2026-08-10
**来源**: 用户要求基于自编 CLAUDE.md 开发规范整理项目;审计后按 B 档范围执行
**关联**: [[09_play_interactive_force_feedback]], [[02_shell_reorganize_by_task]], [[08_port_fall_recovery_branch]]

---

## 问题描述

CLAUDE.md (企业级闭环开发规范) 要求六件事: 开发有痕 / 有据 / 任务独立 / 项目树分类 / 跨平台 / 版本备份。
审计现状 (2 个探查代理) 发现**规范硬性要求但缺失**的基建 + 散落文件:

| 缺口 | 规范条款 |
|---|---|
| `docs/timeline/` 不存在 (需 8 任务时间线) | §2.5 |
| `scripts/xqrobotwl/dump_pose_data.py` 不存在 + `logs/pose_data/` 缺失 | §1.5.1 |
| scripts/ 根目录 12 个散落脚本未按任务归位 | §4.1 |
| 根目录 3 个 0 字节空日志 (train_cpo/np3o/rsl_rl.log) | §4.3 |
| 无项目树文档反映当前结构 | §4.3 |
| 每 1000 iter 训练监控无现成工具 | §1.2 |

## 修改了什么

1. **scripts/ 散落脚本归位** (§4.1): 8 个跳跃/VMC 脚本 → `scripts/xqrobotwl/`;4 个全仓库工具 → `scripts/tools/`(新建)。`git mv` 保留历史。
2. **补 `dump_pose_data.py`** (§1.5.1): 导出每步姿态 CSV → `logs/pose_data/`,26 列与规范 §1.5.2 反推示例兼容。
3. **补 `docs/timeline/`** (§2.5): README 模板 + 8 任务时间线 (从 devlog 提炼历史阶段)。
4. **补 `docs/project_tree.md`** (§4.3): 当前结构 + 任务隔离对照 + 命名约定 + 遗留 TODO。
5. **根目录清理**: 删 3 个 0 字节空日志 (train_cpo/np3o/rsl_rl.log);保留 `CLAUDE copy.md` 与 `picture/` (用户确认)。
6. **`video/` 补任务目录**: walk/jump/stairs/toe_walk 占位 (.gitkeep)。
7. **补 `monitor_training.sh`** (§1.2): 每 1000 iter 提醒 + 异常捕获。
8. **顺带修复移植遗留 (make format/lint 门禁)**: 5 个脚本 F841 未使用变量清理 + 12 处 docstring 用法 `uv run python scripts/` → `uv run scripts/` (卫生测试), 均纯删死代码/改注释, 无行为变化。

## 哪些文件

- **移动** (git mv, 12 个): `scripts/{compare_jump,diag_jump_trajectory,plot_jump_trajectory,record_jump_trajectory,verify_jump,verify_jump_trajectory,diag_vmc_thrust,calibrate_xqrobotwl_vmc}.py` → `scripts/xqrobotwl/`;`scripts/{analyze_offpolicy_trace,audit_sim2sim_contracts,generate_support_matrix,make_paper_figures}.py` → `scripts/tools/`
- **新增**: `scripts/xqrobotwl/dump_pose_data.py`、`scripts/xqrobotwl/__init__.py`、`scripts/tools/__init__.py`、`docs/timeline/{README,walk_flat,walk_rough,jump,stairs,toe_walk,backflip,fall_recovery,single_leg}.md`、`docs/project_tree.md`、`shell/xqrobotwl/tools/monitor_training.sh`、`video/{walk,jump,stairs,toe_walk}/.gitkeep`
- **引用修复** (移动脚本路径失效): `tests/scripts/test_train_scripts.py` (_load_script 加 tools 候选)、`scripts/xqrobotwl/verify_jump_trajectory.py` (ROOT 层级 2→3 + import 路径)、`README.md`、`src/unilab/training/sim2sim.py`、`docs/sphinx/AGENTS.md`、`docs/sphinx/source/{zh_CN,en}/3-deployment/2-sim_to_sim/2-owner_yaml_swap.md`、`docs/sphinx/source/zh_CN/5-reference/5-support_matrix.md`、`docs/sphinx/source/zh_CN/4-developer_guide/9-sim2sim_contract_status.md`、`CLAUDE copy.md`、`src/unilab/envs/locomotion/xqrobotwl/vmc.py`、`conf/ppo/task/xqrobotwl_jump_{vmc,srl_vmc}_flat/mujoco.yaml`

## 根因分析

- 12 个散落脚本原在 scripts/ 根,与 `scripts/<task>/` 分类约定不符 (跳跃诊断/paper 图属任务脚本,契约审计/支持矩阵属全仓库工具)
- 移动脚本内部 `ROOT = Path(__file__).parent.parent` 是**基于原位置**的 2 级假设,移入子目录后变 3 级,不改会找不到仓库根 (sys.path 插错)
- dump_pose_data 之前只写进规范文本、未实现 → doc_checks 前向引用失败是预期,补上即通过

## 验证方法

- `make format` (ruff) 覆盖新增/改动脚本 ✅
- `uv run pytest tests/scripts/test_train_scripts.py -x` 39 通过 (analyze_offpolicy_trace 归位后可加载) ✅
- `uv run pytest tests/scripts/test_check_docs.py -x` 19 通过 (doc_checks 门禁) ✅
- dump_pose_data 冒烟: 对现有 fall_recovery run 导出 50 步 CSV,26 列表头正确,§1.5.2 反推示例逐字可跑 ✅
- `bash -n` monitor 脚本 + 功能冒烟 (1000 iter 里程碑 + Traceback 捕获) ✅
- 全套 `uv run pytest`: **791 通过 / 5 失败** — 5 个全是 NanGuard 警告去重既有问题 (含已知 np_env 失败), 与本次改动无关; `make type` (mypy) 150 错全在 src/ 移植/既有文件, 本次脚本/文档改动 0 mypy 新增 ✅
- grep 确认无残留失效路径引用 ✅

## 后续计划

- [ ] 提交 (用户自行 commit, 不推送)
- [ ] **既有债务 (非本次引入, 另立任务)**: 5 个 NanGuard 测试失败 (警告去重) + `make type` 150 mypy 错 (移植 env + 既有 appo/np3o 算法)
- [ ] 时间线文档随开发实时追加 (docs/timeline/)
- [ ] 全量训练达标后: 渲染视频 + dump_pose_data 导出 + 版本备份 (规范 §2.4/§1.5/§6)
