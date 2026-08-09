# 16 新增 PPO+VMC 与 SRL+VMC 两种跳跃算法 (论文 2×2 对比)

**日期**: 2026-08-06
**来源**: 小论文四种跳跃算法对比需求 — 已有单PPO/SRL(关节空间),需补齐 VMC(虚拟腿力矩控制)两个版本
**关联**: [01_phase_gated_rewards](2026-07-14/01_phase_gated_rewards.md), [08_wheeled_srl_framework_diag](2026-07-22/08_wheeled_srl_framework_diag.md)

---

## 问题描述

论文需要四种跳跃算法公平对比 (2×2: 关节空间 vs 虚拟腿VMC × 纯PPO vs SRL):

| | 关节空间 (位置控制) | 虚拟腿 VMC (力矩控制) |
|---|---|---|
| 纯 PPO | `XqRobotWLJumpFlat` ✅已有 | **`XqRobotWLJumpVMC`** 新建 |
| SRL (FSM前馈) | `XqRobotWLJumpSRLFlat` ✅已有 | **`XqRobotWLJumpSRLVMC`** 新建 |

## 根因分析 / 方案设计

参考项目 Wheel-Legged-Lab (Isaac Lab) 用 VMC 虚拟腿控制:策略输出虚拟腿角 theta0、腿长 L0、轮速,VMC 雅可比映射为关节力矩。xqrobotwl 需移植并适配:

1. **力矩级控制**:新建 `xqrobotwl_vmc.xml`(8 个 `<motor>` 作动器 + ctrlrange),用后端 `set_pre_step_control` 钩子(go2w 模式)在每物理子步前把策略控制转力矩。
2. **FK 标定**:xqrobotwl 关节轴与参考不同(膝关节在 qknee=0 已弯曲 ~96°, 腿无法完全伸直),数值标定 `scripts/calibrate_xqrobotwl_vmc.py` 扫描真实 MuJoCo 模型拟合 l1/l2/offset 与逐腿符号约定。
3. **8D 动作空间**(与纯 PPO 网络输出维一致,公平对比):`[roll_L, theta_L, L0_L, roll_R, theta_R, L0_R, wheel_L, wheel_R]`。hip_roll 单独 PD,theta/L0 走 VMC,轮速 PI。
4. **SRL+VMC**:复用 jump_srl 的 FSM 六状态,FSM 提供每阶段 L0 参考(下蹲→蹬伸→腾空收腿→落地压缩),策略输出残差(`final_L0 = L0_ref × feedback_gain + 残差`),并按阶段调 VMC 增益(蹬伸 kp×1.25/前馈×1.15、腾空前馈×0.55、落地 kd×2.5)。

## 修改文件

| 文件 | 修改内容 |
|------|---------|
| `src/unilab/assets/robots/xqrobotwl/xqrobotwl_vmc.xml` | 新建,8 个 motor 作动器 (腿 ±30 N·m、轮 ±10 N·m) |
| `src/unilab/assets/robots/xqrobotwl/scene_flat_vmc.xml` | 新建,include VMC robot + home keyframe |
| `scripts/calibrate_xqrobotwl_vmc.py` | 新建,数值标定 FK (l1=0.3005, l2=0.3007, hip_sign=[+1,-1], knee_sign=[-1,+1], c1=2.4103, c2=-1.679, l0_offset=0.3669),双腿 RMSE<3mm |
| `src/unilab/envs/locomotion/xqrobotwl/vmc.py` | 新建,VMC 核心 (FK、雅可比、roll PD、wheel PI) |
| `src/unilab/envs/locomotion/xqrobotwl/jump_vmc.py` | 新建,PPO+VMC 环境 (obs 369/468) |
| `src/unilab/envs/locomotion/xqrobotwl/jump_srl_vmc.py` | 新建,SRL+VMC 环境 (obs 387/486, FSM L0 参考+残差+阶段增益) |
| `src/unilab/envs/locomotion/xqrobotwl/__init__.py` | 导出两个新环境类 |
| `conf/ppo/task/xqrobotwl_jump_vmc_flat/mujoco.yaml` | 新建,PPO+VMC 训练配置 (num_envs=1024, max_iter=10000) |
| `conf/ppo/task/xqrobotwl_jump_srl_vmc_flat/mujoco.yaml` | 新建,SRL+VMC 训练配置 |
| `shell/xqrobotwl/train_ppo_jump_vmc.sh` | 新建,训练脚本 |
| `shell/xqrobotwl/train_ppo_jump_srl_vmc.sh` | 新建,训练脚本 |
| `tests/envs/locomotion/xqrobotwl/test_xqrobotwl_jump_vmc.py` | 新建,10 个测试 (FK 往返、力矩有限、env 冒烟、FSM 迁移) |

## 验证方法

1. 标定: `uv run python scripts/calibrate_xqrobotwl_vmc.py` → 左腿 0.30mm、右腿 0.29mm、统一约定双腿 <3mm RMSE
2. 测试: `uv run pytest tests/envs/locomotion/xqrobotwl/test_xqrobotwl_jump_vmc.py` → **10 passed**
3. FK 往返: 默认姿态 L0=0.3669=offset、theta0=0.0747=offset ✓
4. 短训冒烟 (16 env × 3 iter): PPO+VMC 与 SRL+VMC 均正常训练;SRL+VMC 出现 vertical_thrust>0 (FSM 驱动跳跃)
5. mypy/ruff: 新文件无错误 (仓库既有 jump.py/stairs.py 的 mypy 错误为历史遗留,未改动)

## 评估结果

obs 维度: PPO+VMC 369/468, SRL+VMC 387/486。力矩有限且均在 ctrlrange 内。零动作下机器人自然沉降(与纯 PPO 的"零动作振荡"一致,平衡需 RL 学习)。

## 后续计划

- [ ] 四版本正式训练 (用户分别运行训练脚本)
- [ ] 扩展 jump_management/evaluate runner 支持两个新任务,生成四版本对比表
- [ ] VMC 增益/前馈力可能需要按训练结果微调 (feedforward=80N 为初值)
- [ ] 参考项目经验: 机身高度 ≠ 轮端净空, 后续可加轮端净空奖励

## 关联日志

- [12_sign_bugs_entropy_fsm_fixes](2026-07-24/12_sign_bugs_entropy_fsm_fixes.md) (SRL 符号/FSM 修复)
- [11_srl_convergence](2026-07-22/11_srl_convergence.md) (SRL 收敛)
