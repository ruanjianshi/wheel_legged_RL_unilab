# [12] 按更新后的 CLAUDE.md (v2) 整理仓库增量

**日期**: 2026-08-10
**来源**: 用户更新开发规范 CLAUDE.md (新增 §0 汇报机制 / §1.3 反推映射表 / §1.5 数据优先三节 / §2.4 视角内渲染 / §7.0 通用评估+时长表 / §7.1 八任务总览 / 附录 A·C), 要求"阅读, 整理"
**关联**: [[10_reorganize_per_claude_spec]] [[11_fix_nan_guard_and_mypy_gate]]

---

## 差距审计结论 (新规范 vs 仓库现状)

对照更新后的规范逐条核对, **多数条款在上一轮整理 ([[10]]) 后已合规**:

| 已合规 (无需动) | 对应条款 |
|---|---|
| `scripts/training/train_cpo.py` 等训练入口 + `shell/xqrobotwl/tools/monitor_training.sh` | §1.2 |
| `render_recovery_video.py` `cam_tracking=True` + `render_trained_*` 用 `render_states_get_frames_tracking(tracking_env_idx=0)` | §2.4 视角内 |
| backup 结构含 `git_commit.txt` + `uncommitted.diff` + README + conf + src + shell | §6.2 |
| `dump_pose_data.py` 列名与 §1.5.1/§1.5.2 完全一致 | §1.5.1 |
| conf 任务目录 / scripts 三级结构 / devlog 结构 | §3.1 / §4.1 |

**真实缺口 (本次补齐)**:

| 缺口 | 条款 | 处置 |
|---|---|---|
| 无"读 CSV → 按反推表判姿态 → 统计时长/占比"工具 (仅规范 inline 示例) | §1.3 / §1.5.2 | 新增 `scripts/xqrobotwl/infer_pose_from_csv.py` |
| `video/rough/` 缺失 (不平坦地形) | §3.1 | 新建 + `.gitkeep` |
| §2.5 "开发框架" (目标/约束/方案/分阶段计划/风险) 无模板落点 | §2.5 | `docs/timeline/README.md` 增开发框架模板节 |
| `video/single_leg/_frames_*` 2.6GB ppm 中间帧已被 git 追踪 3000 文件 | §4.3 | `.gitignore` 加 `video/**/_frames_*/` (用户拍板: 只 ignore 不删) |

**用户拍板**: 帧目录只加 gitignore 本地保留; 楼梯视频目录保留 `stairs/` 复数 (与 env/config/timeline 一致, 规范表作 `stair/` 视为笔误)。

## 新增工具: `infer_pose_from_csv.py` (§1.3/§1.5.2)

- 输入: `dump_pose_data.py` 产出的 26 列姿态 CSV (`logs/pose_data/`)
- 逐行按 §1.3 反推映射表 + §1.5.2 示例链判定姿态, 优先级:
  `倒地 > 左右腿一前一后 > 髋外展/内收 > 左右高低腿 > 下蹲 > 伸腿/过高 > 前倾/后倾 > 左右倾斜 > 站立(可被 摇摆/轮子点地/转圈 覆盖) > 过渡`
- 转圈按**连续站立段**检测 (段内 yaw 累计漂移 > 1 rad 且左右轮速差 > 0.5 rad/s 才改判)
- 输出: 各姿态帧数/时长(s)/占比(%), 站立期微动指标 (mean|linvel_xy| / mean|gyro| / yaw 累计 / 轮子离地率), 恢复指标 (recover_completed + 首次恢复步)
- 纯 stdlib, `uv run python` 可跑 (不依赖 mujoco/torch); `--out` 追加 pose 列, `--json` 机器可读
- 阈值常量与 §1.3 表一致 (0.25/0.60/0.45/0.30/0.50/0.30/0.50/0.20/0.85/1.0)

## 修改的文件

- 新增 `scripts/xqrobotwl/infer_pose_from_csv.py`
- 新增 `video/rough/.gitkeep`
- `.gitignore`: `video/**/_frames_*/` (第 52 行)
- `video/README.md`: 任务→目录映射表 (§3.1), 中间帧说明
- `docs/timeline/README.md`: 开发框架模板节 (§2.5)
- `docs/project_tree.md`: rough 目录 / infer 工具 / gitignore 规则 / TODO 更新

## 验证方法

- `uv run ruff format` + `ruff check scripts/xqrobotwl/infer_pose_from_csv.py` ✅
- 真实数据: `logs/pose_data/fall_recovery_model_499_仰卧_supine.csv` (50 步早期模型, 46% 倒地/28% 前倾/14% 前后腿/12% 高低腿, 未恢复 → 合理)
- 合成 19 行覆盖全分支: 13 类单行分类全部正确 + 转圈段检测 (连续站立段 yaw 漂移 2 rad + 轮速差 4 → 6 帧改判) + `--out`/`--json` 正常
- `git check-ignore` 确认新规则匹配未来 `_frames_` 路径

## 遗留说明

- 已追踪的 3000 个 ppm 帧仍留在 git 历史 (用户选择本地保留不删); 若要瘦身需 git 历史重写, 本次不做
- `video/stair` vs `stairs`: 仓库统一 `stairs` 复数, 与 env/config/timeline 对齐
- 新任务启动时按 `docs/timeline/README.md` 的开发框架模板先行 (目标/约束/方案/分阶段/风险) 再开发

## 后续计划

- [ ] 提交 (用户自行 commit, 不推送)
- [ ] 各任务达标评估流程落实: `dump_pose_data` → `infer_pose_from_csv` 闭环 + §7.0 长时评估
