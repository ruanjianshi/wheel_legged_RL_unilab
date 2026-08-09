# [01] 跌倒恢复方案改向 — FTSR (力引导 + 高度分阶段 + CPO 约束)

**日期**: 2026-08-07
**来源**: 用户指定论文 Hou et al. 2026 "Robust Fall Recovery for Armless Bipedal-Wheeled Robots via Force-Guided Learning" (FTSR), 要求按此实现
**关联**: 2026-08-06 的 [01_p1_feasibility](../2026-08-06/01_p1_feasibility.md) (已废弃) / [02_env_and_config](../2026-08-06/02_env_and_config.md) (已废弃)

---

## 问题描述

08-06 实现了"贴地后空翻"方案 (脚本化 FF 翻身 + RL 平衡)。用户指出应参照 FTSR 论文:
**无脚本轨迹, 纯学习恢复** — 策略从任意倒地姿态自己学起身, 恢复后能站住平衡。

## 根因分析 (论文 vs 旧方案)

| 维度 | FTSR 论文 | 08-06 旧方案 |
|------|----------|-------------|
| 恢复机制 | 纯学习 (trajectory-free) | 脚本化贴地后空翻 FF |
| 训练辅助 | 外部力 F + 力矩 T, 与高度相关, 作 CPO 约束 | 无 |
| 分阶段 | 按高度阈值 ru→rs→rw, 批次统计切换 | 固定 FSM 相位 |
| 初始姿态 | 4 种随机 (仰/俯/左/右) + 扰动 | 固定仰卧 |
| 核心消融 | 力课程=0%, 力引导=99.8%, 分阶段必需 | — |

## 解决方案

按 FTSR 重做 (用户确认范围: 核心 FTSR + 恢复后站住平衡 ru/rs, 不含行走 rw; 师生蒸馏留作后续):
1. **力引导**: env 训练期施加与高度相关的向上辅助力 F + 对齐直立力矩 T,
   同时作为 **CPO 约束** (C1=F, C2=T, d→0), 罚策略对辅助的依赖。
2. **高度分阶段奖励**: ru (上半身直立, h_cmd1=0.35) → rs (站起, h_cmd2=0.55),
   按批次高度统计 (|S1|>2/3N) 切换, 每阶段独立奖励集。
3. **多姿态复位**: 仰/俯/左/右 4 种倒地姿态 + 姿态/关节扰动。
4. **贴地终止**: base_z<0.16 持续 idle_ground_time(6s) 终止 (防死点);
   **恢复前 (has_recovered=False) 不按倾覆终止** (倒地是合法起始态)。
5. **观测 = 基础 297/324** (commands[4]=h_cmd), 热启动自 walk 直接复制。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/envs/locomotion/xqrobotwl/fall_recovery.py` | 全量重写: 删除脚本化 FF, 实现 FTSR |
| `src/unilab/base/backend/base.py` | + `apply_body_wrench` 接口 (力+力矩) |
| `src/unilab/base/backend/mujoco/backend.py` | + `apply_body_wrench` 实现 (xfrc_applied 6分量) |
| `src/unilab/base/backend/motrix/backend.py` | + `apply_body_wrench` stub (NotImplementedError) |
| `conf/ppo/task/xqrobotwl_fall_recovery_flat/mujoco.yaml` | 全量重写: FTSR 参数 + CPO 算法 |
| `scripts/xqrobotwl/warmstart_from_walk_fall_recovery.py` | 重写: 直接复制 walk (obs 相同, 无需填充) |

## 验证方法

`uv run mjpython scripts/xqrobotwl/smoke_fall_recovery.py` + CPO 训练冒烟

## 评估结果

- Smoke: obs 297/324 ✓, 多姿态复位 (up 分布 x/y) ✓, 力引导施加 (F≈38N, T≈5Nm) ✓, reward 有限 ✓
- 零动作下 base_z 保持 0.13-0.17 (38N 辅助 < 183N 自重, 需腿+策略配合 — 符合设计)
- **关键修复**: 初始终止条件 (tilt>55°) 把倒地起始态立刻杀掉 → 改 has_recovered 门控后
  episode 从 1 步 → 415 步

## 后续计划

- [ ] 训练: warmstart 自 walk → CPO 训练 → 确定性评估选模型
- [ ] 力参数调优 (Fmax 是否足够; 论文无参考值)
- [ ] 师生蒸馏 (特权教师, 论文消融 59%→99%)
- [ ] 行走阶段 (rw) + 俯卧/侧躺专项验证

## 关联日志

- FTSR 环境+配置: [02_cpo_port](02_cpo_port.md) / [03_fsr_env](03_fsr_env.md)
- 已废弃后空翻方案: 2026-08-06/01, 02
