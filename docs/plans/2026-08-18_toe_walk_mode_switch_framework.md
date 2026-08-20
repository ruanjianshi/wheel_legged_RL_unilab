# 开发框架 · 点足抬腿任务升级：站立 ⇄ 点足抬腿 双模式 + 按键切换 + 指令追踪

> 日期: 2026-08-18 | 状态: **方案待老板审阅** | 关联: [参考文档](../references/2026-08-18_mode_switching_multi_skill.md) | 时间线: [toe_walk](../timeline/toe_walk.md)
> 背景: 老板明确需求 — 默认站立姿态 → 键盘按键切换成点足抬腿模式 → 进行指令追踪（前进后退/侧向/转向）→ 按键切回站立。
> 现状: 现有 toe_walk 是"相位时钟持续抬腿"单模式, 无站立模式, 且最新模型已退化为单边抬腿转圈 (8-18 验证)。

## 一、目标

一个策略内实现两种行为模式, 按键实时切换:

1. **站立模式 (mode=0, 默认)**: 静止微动平衡 (对照 walk_flat 站立标准 §1.4), 可做小幅速度追踪
2. **点足抬腿模式 (mode=1)**: 左右交替点足抬腿 (对照 §7.3 + 8-18 交替/对称验证门槛), 支持前进/后退/侧移/转向指令追踪
3. **切换**: 任意时刻按键切换, 切换过程不跌倒 (0.5s 内), 1s 内进入新模式行为
4. 训练产出: 收敛模型 + 确定性评估 CSV + 渲染视频 + 版本备份, 全程 devlog 痕迹

## 二、约束

- 不破坏现有 8 任务隔离 (toe_walk 独立 env/conf/shell/devlog/video)
- 共享只读基座 (base.py / joystick.py / 机器人 XML) 不独占修改
- 训练前代码可通过 `make format/type` (ruff/mypy)
- 每次新开发设置 checkpoint / 备份可回退 (CLAUDE.md §9)

## 三、方案 (推荐: 单策略 + mode 命令通道, 参考 Uni-Match/so100/CaT)

### 3.1 命令与观测

| 项 | 现状 | 改动 |
|----|------|------|
| commands | 4D [vx,vy,vyaw,tsk] | **5D + mode** (末位 0/1) |
| obs | 相位 sin/cos 已含 | +mode 维 (直接拼在命令后) |
| 相位时钟 | 始终推进 | **保持连续不重置** (切换无硬跳变) |

### 3.2 奖励按 mode 门控 (env 内, 不拆双环境)

- **共项**: tracking_lin_vel / tracking_ang_vel / lin_vel_z / ang_vel_xy / base_height / orientation / joint_wheel_action_rate / leg_mirror / alive (站立与抬腿共享)
- **mode=0 站立**: 上述共项 + 微动平衡项 (stand_still 风格零指令不动, 可参考 walk_flat 已有配方), **关闭** 抬腿四项 (phase_swing_lift/knee_lift/knee_stance/stance_penalty)
- **mode=1 抬腿**: 共项 + 现行相位门控抬腿套件 (保持 07-28 配方: phase_swing_lift 30 / knee_lift 15 / action_scale 0.18)
- **新增对称性约束 (解决 8-18 根因)**: `lift_symmetry` 奖励 = -|L抬幅 - R抬幅| 按摆动相累计, 显式压制单侧塌缩; 可选交替门控 (摆动相到时器, CaT 式)

### 3.3 命令采样与课程

- DR provider: mode 按 **3~8s 随机驻留重采样** (覆盖任意切换时刻), 速度指令在各模式内按各自 vel_limit 采样
- 课程 (防多目标冲突, 历史教训: ramp 过陡即崩):
  - Stage 0-2k: 纯 mode=0 (站立)
  - Stage 2k-5k: 纯 mode=1 (抬腿) — 先复现交替步态
  - Stage 5k-10k: mode 随机切换混合 + 全指令
- max_iterations 10000, num_envs 1024 (与现行一致)

### 3.4 交互与评估 (交付老板)

- **play_interactive**: H 键 = mode 切换 (写 commands[4]); 复用现有 ↑/↓←/→/A/D 指令键 + legend 提示
- **确定性评估脚本** `verify_toe_walk_mode.py` (基于 8-18 验证工具扩展): 脚本化 mode 序列 — 站立3s → 切抬腿 → vx=0.2 前进3s → vy 侧移3s → vyaw 转向3s → 切回站立3s, 逐帧姿态 CSV + 交替/对称/追踪/高度指标
- **渲染视频**: 同一序列 render 到 `video/toe_walk/` (相机跟踪, 视角内)

### 3.5 验收标准

| 项 | 标准 |
|----|------|
| 站立 mode=0 | 零指令漂移<0.2m/s · \|gyro\|<1 · z≈0.52±0.05 · up_z>0.9 |
| 抬腿 mode=1 | 交替 PASS + 对称 PASS (8-18 门槛) · vx/vy/vyaw 追踪 RMSE 达标 |
| 切换 | 切换前后 0.5s 不倒 · 1s 内进入新模式行为 (姿态/接触统计) |
| 长时 | 站立≥10s / 抬腿行走≥30s |

## 四、分阶段计划

| 阶段 | 内容 | 产出 | 预计 |
|------|------|------|------|
| P0 (本次) | 参考文档 + 本框架, 老板审阅 | 2 份文档 ✅ (参考/框架) | 已交付 |
| P1 | env: commands 5D mode + 奖励门控 + 对称性奖励; conf 新任务目录 (保留旧 toe_walk 配置可回退) | 代码 + devlog 09 + checkpoint 点 A | 1 天 |
| P2 | DR 采样 + 课程 + play_interactive H 键 + legend | 代码 + devlog 10 | 0.5 天 |
| P3 | 训练 (1024 envs × 10k iter, ~6h) + 每 1k iter 监控评估 | run 目录 + 中间 devlog | 1 天 |
| P4 | verify_toe_walk_mode.py 确定性评估 + 姿态 CSV + 渲染视频 | 评估报告 + video/toe_walk/*.mp4 + devlog 11 | 0.5 天 |
| P5 | 达标确认 → 备份 toe_walk_mode_v1 (模型+conf+src+脚本+README) | backup/ + timeline 更新 | 0.5 天 |

> 每阶段结束: 更新 `docs/timeline/toe_walk.md` + `_devlog/xqrobotwl/toe_walk/ppo/INDEX.md`; 旧模型/配置保留可回退。

## 五、风险与对策

| 风险 | 概率/影响 | 对策 |
|------|-----------|------|
| 双模式同策略训练冲突 (站立↔抬腿互相拉崩) | 中/高 | 三阶段课程 + mode 门控奖励 + 对称性约束; 若冲突 → 降 mode 切换频率或分环境双策略兜底 |
| mode 切换时刻策略失稳 | 中/中 | 相位连续 + 切换不重置 + 训练覆盖任意时刻切换; 播放层滞回 |
| 抬腿行为重新退化 (8-18 已证) | 高/高 | lift_symmetry 奖励 + 交替门控 + 候选模型过验证门槛才放行; 这是本次硬性新增 |
| 训练时长/算力 | 中/低 | 每 1k iter 评估及时止损; 1024 envs 与历史一致 |

## 六、与论文/调度器关系 (备注)

本需求属于"第1阶段专家内行为调制" (专家 03 抬腿行走升级), 不引入全局调度器; thesis/scheduler 的"8 专家切换"仍是论文第 2 阶段独立课题, 本方案的 mode-conditioned 单策略可作为其"专家内多行为"的前置验证。

---
**待老板确认后开工 P1。** 确认点: ① 方案 (单策略+mode 通道) ② 是否需要我同时把"站立模式"做成独立可交付演示 (不含抬腿) ③ 训练 GPU 资源/时长窗口。