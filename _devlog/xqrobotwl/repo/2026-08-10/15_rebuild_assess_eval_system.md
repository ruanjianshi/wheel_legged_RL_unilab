# [15] 重构 _devlog/assess → 八任务完整评估体系

**日期**: 2026-08-10
**来源**: 用户指令 — "'_devlog/assess' 重构这个, 构建完整的评估体系"; 此前评估体系现状: 8 任务中仅
fall_recovery + single_leg 有指标化确定性 eval, 其余 6 任务只有 play_interactive 交互回放
**关联**: [[14_audit_8_tasks_isolation_update_readme]]

---

## 做了什么

把旧 `_devlog/assess/` (XqRobotV2 时代框架: 只注册 5 个 V2 任务 / 22 项通用论文指标 /
scenarios=decoupling/full/standing / 硬编码 env builder / 失效路径自举) 重构为
**按 CLAUDE.md §1.2-1.5 / §7.0 / §7.x / 附录 A 的八任务完整评估体系**:

| 文件 | 状态 | 说明 |
|---|---|---|
| `tasks.py` | 重构 | 8 个 xqrobotwl 任务注册表 (env 注册名/conf/log 根/algo/ctrl_dt/obs_dim) + §7.x 达标阈值 dict |
| `engine.py` | 新增 | 通用确定性 rollout 引擎: build_env(训练 run_config 优先还原) / load_policy(复用 verify_jump.load_actor) / run_cmd_scenario / run_episodes(active mask) / collect_step(26 列采集) |
| `metrics.py` | 重构 | StepSample/Trace 载体 + **附录 A 指标**(恢复率/最长连续站立/站立高度/水平漂移/yaw 累计/|gyro|/轮速差/轮子离地率) + 追踪/稳定/运动指标; 保留旧 16 项 |
| `scenarios.py` | 重构 | EvalScenario/EvalSuite + 行走套件 (decoupling/full/standing) |
| `verify.py` | 新增 | §7.x 阈值判定 → 逐项 ✅/❌ + 总体 |
| `report.py` | 新增 | stdout 摘要 + results/<task>/<session>/metrics.json + reports/<task>/eval.md |
| `pose.py` | 新增 | 姿态数据 CSV 导出 (§1.5.1, 26 列, 两位小数) → logs/pose_data/ |
| `infer.py` | 新增 | 姿态反推统计 (§1.3/§1.5.2, 复用 infer_pose_from_csv 判定) |
| `runner.py` | 重写 | 统一 CLI (修 ROOT/SRC 路径自举, 弃硬编码 env builder, 弃 5 个 V2 任务) |
| `eval/` 8 模块 | 新增 | walk_flat / toe_walk / walk_rough / jump / backflip / single_leg / fall_recovery / stairs, 各实现 §7.x |
| `README.md` | 重写 | 用法 + 对照规范 |
| `reporter.py` | 删除 | 被 report.py 取代 (flat_walk 专用, 已废弃) |

**复用**: `tools/xqrobotwl/verify_jump.load_actor` + `trained_env_overrides`(评估即训练配置),
`infer_pose_from_csv` 判定, `eval_fall_recovery._set_fixed_pose_provider`(固定倒地姿态),
旧 assess 的 16 项追踪/稳定指标。历史 results/plots/database 保留。

## 关键设计决策

1. **评估即训练配置**: build_env 优先从 ckpt 相邻 run_config.json 还原训练配置
   (trained_env_overrides), 无快照才回退 conf yaml — 否则 jump 等任务评估复现不出训练行为
   (实测: 用 conf yaml 时 jump air_frac=0, 用训练配置后跳到 0.35m/18% 腾空)
2. **站立判定拆分**: `upright_mask`(z∈[0.45,0.65]+up>0.85, 通用, walk env 不填充轮地接触)
   vs `standing_mask`(+双轮着地, 附录 A 精确版, fall_recovery 用)
3. **统一 runner + 每任务 eval 模块**: `uv run python _devlog/assess/runner.py -t <task> -r <run> -c <ckpt>`
   → 任务模块定义场景/指标/阈值; 8 个 agent 可并行跑不同 `-t`
4. **动作类任务触发对齐训练**: jump/backflip 触发周期取 `commands.resampling_time`(4s)

## 验证方法

- **合成数据单测**: metrics/verify/infer 用假 StepSample 全分支通过 (恢复率/最长站立/漂移/yaw/追踪/达标判定)
- **真实 checkpoint 冒烟**:
  - walk_flat model_9999: 端到端 (指标+达标判定+JSON+报告+姿态 CSV), 如实暴露该模型站立转圈问题 (gyro 4.9, yaw 累计大) → ❌ 未达标
  - fall_recovery model_499: run_episodes + 固定姿态, 恢复率 0 (早期模型未训练成) → 如实
  - jump model_9999 (08-07_19-54): 检测到真跳 (0.35m/18% 腾空/成功率 1.0) → 与 verify_jump 对照一致
- **门禁**: `make format` ✅ / `make type` ✅ (mypy 201 文件, pyright 0 错) / 全量 pytest **796 passed**
- 相关测试: test_check_docs / test_repo_hygiene / test_train_scripts 全绿

## 遗留说明

- jump/backflip 触发周期、stairs 上台阶判据、single_leg 三态判据为首次实现,
  待各任务真正达标模型出来后再校准阈值
- `survival_rate` 已从 jump 阈值移除 (§7.5 以成功率为主判据)
- 旧 exporter/plotter/recorder 保留 (独立可导入, 未接入新 runner; 趋势/对比/绘图为后续扩展点)
- 历史 `results/plots` 是 V2 时代数据, 保留存档

## 后续计划

- [ ] 提交 (用户自行 commit, 不推送)
- [ ] 接入 monitor_training.sh 每 1000 iter 评估建议 (§1.2): 提示改用 `runner.py -t <task>`
- [ ] 各任务达标后: 校准阈值 + 渲染视频 + pose 数据闭环产出
