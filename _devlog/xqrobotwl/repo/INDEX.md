# xqrobotwl/repo 开发索引

仓库/工程管理类日志 — 跨任务的整理、清理、脚本管理、批量训练启动等。
不隶属于某个具体训练任务 (walk/jump/stairs/...)。

## 2026-08-10

| 序号 | 标题 | 文件 |
|------|------|------|
| 01 | 仓库瘦身: 移除除 xqrobotwl/xqrobotV2 外所有机器人 | [→](2026-08-10/01_strip_repo_to_two_robots.md) |
| 02 | shell 目录按任务细分重组 | [→](2026-08-10/02_shell_reorganize_by_task.md) |
| 03 | 根目录清理: 删 AGENTS.md 只留 CLAUDE.md + 散落文件 | [→](2026-08-10/03_cleanup_root_agents_to_claude.md) |
| 04 | 启动 8 个 xqrobotwl 训练 + 修复 shell 路径与 stairs obs 维度 bug | [→](2026-08-10/04_launch_8_trainings_and_fix_path_obs_bugs.md) |
| 05 | 8 训练全部完成 + 8 验证脚本冒烟通过 + 大清理无影响评估 | [→](2026-08-10/05_verify_8_trained_policies.md) |
| 06 | 新建 thesis/ 论文开发指导中心 (框架图 + 开发文档模板) | [→](2026-08-10/06_thesis_dev_guide_folder.md) |
| 07 | thesis/ 按用户架构重构: 分训-整合两阶段 + 7专家 + 调度器 | [→](2026-08-10/07_thesis_folder_refactor_by_user_arch.md) |
| 08 | 移植 fall_recovery 分支: 后空翻/单腿平衡/跌倒恢复 + CPO | [→](2026-08-10/08_port_fall_recovery_branch.md) |
| 09 | play_interactive 窗口施力回灌: Ctrl 拖动力真实作用于仿真 | [→](2026-08-10/09_play_interactive_force_feedback.md) |
| 10 | 按 CLAUDE.md 规范整理: 时间线/姿态数据工具/脚本归位/项目树 | [→](2026-08-10/10_reorganize_per_claude_spec.md) |
| 11 | 修复 NanGuard 测试失败 + make type 门禁 (mypy/pyright) | [→](2026-08-10/11_fix_nan_guard_and_mypy_gate.md) |
| 12 | 按更新后的 CLAUDE.md (v2) 整理: 反推工具/rough 目录/开发框架模板/gitignore | [→](2026-08-10/12_reorganize_per_updated_spec.md) |
| 13 | 迁移 scripts/{tools,xqrobotwl} → tools/ + pose_data 保留两位小数 | [→](2026-08-10/13_move_scripts_to_tools.md) |
| 14 | 八任务隔离审计 ✅ + README 更新 (tools/ 迁移与八任务结构) | [→](2026-08-10/14_audit_8_tasks_isolation_update_readme.md) |
| 15 | 重构 _devlog/assess → 八任务完整评估体系 (engine/metrics/verify/report/pose/infer + 8 eval 模块) | [→](2026-08-10/15_rebuild_assess_eval_system.md) |
