# 项目树 · 目录结构与规范对照

> 规范 CLAUDE.md §4.1/§4.2/§4.3: 构建文件不可乱放, 分类管理, 命名一看就懂, 定期清理并更新项目树文档。
> 本文档反映**当前**结构 (2026-08-10 整理后), 结构变化时同步更新。

> 规范更新增量 (2026-08-10 v2) 已落实项: §1.3 反推映射表 + §1.5.2 → 新增 `infer_pose_from_csv.py`;
> §3.1 → `video/rough/` 补齐 8 任务目录; §2.5 开发框架模板 → `docs/timeline/README.md`;
> §4.3 → `video/**/_frames_*/` 入 gitignore。§0 汇报机制 / §7.0 长时评估 / 附录 A/C 为流程规范, 见 CLAUDE.md。

## 顶层目录

```
wheel_legged_RL_unilab/
├── src/unilab/                      # 核心源码
│   ├── algos/                       # RL 算法 (ppo / np3o / cpo / offpolicy / …)
│   ├── base/                        # 基础框架 (registry / backend / env 基类)
│   ├── dr/                          # 域随机化 Domain Randomization
│   ├── envs/locomotion/xqrobotwl/   # xqrobotwl 任务环境 (每任务一个文件, 13 个注册)
│   ├── assets/robots/               # 机器人 XML (xqrobotwl/ / xqrobotV2/)
│   └── training/                    # 训练/推理公共组件
├── conf/                            # Hydra 配置 (任务隔离)
│   ├── ppo/task/xqrobotwl_<task>*/  # PPO 任务 (walk_flat / walk_rough / jump×4 / backflip / single_leg×3 / toe_walk)
│   ├── np3o/task/xqrobotwl_stairs/  # NP3O 任务
│   └── cpo/task/xqrobotwl_fall_recovery_flat/  # CPO 任务
├── tools/                           # 评估/渲染/工具脚本 (2026-08-10 由 scripts/{xqrobotwl,tools} 迁入, 含既有工具)
│   ├── xqrobotwl/                   # 任务脚本: eval_* / render_* / *_feasibility / warmstart_* / dump_pose_data / infer_pose_from_csv
│   ├── analyze_offpolicy_trace.py   # 全仓库工具 (平铺): 契约审计 / 支持矩阵 / 论文图 / 离线轨迹分析
│   ├── audit_sim2sim_contracts.py
│   ├── generate_support_matrix.py
│   ├── make_paper_figures.py
│   └── email/  mujoco/  pinocchio_traj/  xqrobotV2/  eval_flat_walk.py   # 既有工具
├── scripts/
│   ├── training/                    # 训练入口 train_*.py
│   ├── play/                        # 交互/回放入口 (play_interactive 施力回灌)
│   └── manip_loco/  completions/
├── shell/xqrobotwl/<task>/          # 启动脚本, 每任务 train_<algo>_<task>.sh + eval_<algo>_<task>.sh
│   └── tools/                       # monitor_training / export_onnx / play_policy / play_viser
├── logs/                            # 训练产物 (git 忽略)
│   ├── rsl_rl_<algo>/<Task>/<run>/  # 训练 run 目录 + checkpoint model_<iter>.pt
│   └── pose_data/                   # 姿态数据 CSV (dump_pose_data 导出, git 忽略)
├── video/<task>/                    # 结果视频 (8 任务目录已建: walk / toe_walk / rough / jump / backflip / single_leg / fall_recovery / stairs)
├── _devlog/<robot>/<task>/<algo>/<date>/<NN>_<slug>.md   # 开发日志
├── docs/
│   ├── references/                  # 参考文档 (论文/开源项目借鉴)
│   ├── timeline/                    # 开发进展时间线 (每任务一份)
│   ├── project_tree.md              # 本文档
│   └── sphinx/                      # 双语 Sphinx 站点
├── thesis/                          # 论文开发指导中心 (框架图 / 专家文档 / 调度 / 整合)
├── backup/<Robot><Task><Variant>/<task>_<algo>_v<N>/     # 版本备份 (开箱即跑)
├── benchmark/  picture/  Sim2real/  tests/              # 平台/素材/测试
└── CLAUDE.md                        # 开发规范 (本项目的 agent 规范)
```

## 任务隔离对照 (规范 §3.1)

每个任务独立: env 文件 + conf 目录 + shell 脚本 + devlog + video 目录。

| 任务 | env | conf | shell | devlog | video |
|------|-----|------|-------|--------|-------|
| 平地行走 | `joystick.py` | `conf/ppo/task/xqrobotwl_walk_flat/` | `shell/xqrobotwl/flat/` | `_devlog/xqrobotwl/walk_flat/ppo/` | `video/walk/` |
| 粗糙行走 | `rough.py` | `conf/ppo/task/xqrobotwl_walk_rough/` | `shell/xqrobotwl/rough/` | `_devlog/xqrobotwl/{walk_rough,rough}/ppo/` | `video/rough/` |
| 跳跃 | `jump*.py` (4 变体) | `conf/ppo/task/xqrobotwl_jump*_flat/` | `shell/xqrobotwl/jump/` | `_devlog/xqrobotwl/jump/ppo/` | `video/jump/` |
| 上楼梯 | `stairs.py` | `conf/np3o/task/xqrobotwl_stairs/` | `shell/xqrobotwl/stairs/` | `_devlog/xqrobotwl/stairs/np3o/` | `video/stairs/` |
| 点足行走 | `toe_walk.py` | `conf/ppo/task/xqrobotwl_toe_walk_flat/` | `shell/xqrobotwl/toe_walk/` | `_devlog/xqrobotwl/toe_walk/ppo/` | `video/toe_walk/` |
| 后空翻 | `backflip.py` | `conf/ppo/task/xqrobotwl_backflip_flat/` | `shell/xqrobotwl/backflip/` | `_devlog/xqrobotwl/backflip/ppo/` | `video/backflip/` |
| 跌倒恢复 | `fall_recovery.py` | `conf/cpo/task/xqrobotwl_fall_recovery_flat/` | `shell/xqrobotwl/fall_recovery/` | `_devlog/xqrobotwl/fall_recovery/ppo/` | `video/fall_recovery/` |
| 单腿平衡 | `single_leg*.py` (3 变体) | `conf/ppo/task/xqrobotwl_single_leg*/` | `shell/xqrobotwl/single_leg/` | `_devlog/xqrobotwl/single_leg/ppo/` | `video/single_leg/` |

> 注: video/ 8 任务目录均已建 (含 `.gitkeep`), 内容视频按达标进度产出; 楼梯目录用复数
> `stairs/` 与 env/config/timeline 命名一致 (规范 §3.1 表格作 `stair/`, 视为笔误不跟进)。

> 别名说明 (遗留, 暂不统一): shell 目录 `flat/`/`rough/` 对应任务 `walk_flat`/`walk_rough`;
> devlog 下 `rough/` 与 `walk_rough/` 并存; `single_leg_move`/`unicycle` 共享 `single_leg/ppo/` devlog。

## 命名约定 (规范 §4.2)

| 对象 | 规范 | 示例 |
|---|---|---|
| 任务环境文件 | `snake_case.py` | `fall_recovery.py` |
| 配置目录 | `<task>_<variant>_flat/` | `xqrobotwl_fall_recovery_flat/` |
| 训练 run 目录 | 时间戳 `<YYYY-MM-DD>_<HH-MM-SS>_mujoco` | `2026-08-09_15-14-54_mujoco/` |
| checkpoint | `model_<iter>.pt` | `model_4000.pt` |
| 评估脚本 | `eval_<task>.py` | `eval_fall_recovery.py` |
| 渲染脚本 | `render_<task>.py` | `render_recovery_video.py` |
| 视频 | `<日期>_<描述>.mp4` | `2026-08-07_恢复并稳定站立_model4000.mp4` |
| devlog | `<序号>_<slug>.md` | `23_delivery_v7_4pose.md` |
| 参考文档 | `<日期>_<主题>_<来源>.md` | `2026-08-09_force_guided_ftsr.md` |
| 时间线 | `docs/timeline/<task>.md` | `docs/timeline/fall_recovery.md` |
| 备份 | `<task>_<algo>_v<N>/` | `fall_recovery_cpo_v1/` |

## git 忽略规则 (规范 §4.3)

- `logs/`(含 `logs/pose_data/`)与 `*.log`: 训练产物/日志不进 git
- 渲染中间帧 `video/**/_frames_*/` (ppm, mp4 才是交付物)、`.mypy_cache`/`.pytest_cache`/`.ruff_cache`/`.venv`: 不进 git
- 大体积产物 (backup 权重) 按需管理, 不强制入 git

## 遗留 TODO (规范未闭合项)

- [ ] `video/` 8 任务目录已建但多为占位, 达标 checkpoint 需渲染视频 (规范 §2.4 视角内, `cam_tracking=True`)
- [ ] 确定性评估 `eval_<task>.py` 目前仅 fall_recovery / single_leg_move / single_leg_unicycle 有; 其余任务 shell eval 走 `play_interactive` 回放
- [ ] `docs/references/` 文件名尚未按 `<日期>_<主题>_<来源>.md` 重命名 (内容已达标)
- [ ] 新任务 (backflip / fall_recovery / single_leg) 全量训练中, 达标后按规范 §6 做版本备份
- [ ] 姿态数据闭环 `dump_pose_data` → `infer_pose_from_csv` 已通, 需在各任务达标评估流程中落实 (每 1000 iter 评估 §1.2)
- [ ] 长时评估规范 (§7.0: 站立≥10s / 行走≥30s / 动作≥10 次 / 恢复每姿态≥20ep) 与附录 A/C 指标需在各任务评估脚本中体现
