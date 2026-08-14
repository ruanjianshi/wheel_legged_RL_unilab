# [35] 小论文重构: 单方法占位稿 → 四算法 2×2 对照研究 (基于真实项目开发)

**日期**: 2026-08-11
**来源**: 用户指令 — "'latex/Wheeled-SRL-Jumping' 是我的小论文框架, 基于我的项目开发重构完成这篇小论文撰写, 利用 nature-skills 指导"; 现状审计: 小论文是"单方法 Wheeled-SRL"占位稿 (97.2% 虚构数字/弹簧-质点SLIP假设/多处 [待补充])
**关联**: [[16_ppo_vmc_srl_vmc_algorithms]], [[24_final_four_algo_compare_figs]], [[26_final_compare_improved_vmc]], [[28_pure_ppo_reward_hacking_v10_v12]]

---

## 审计结论 (重构依据)

| 项 | 旧 main.tex (占位稿) | 真实项目 (devlog 16/19/24/26/28) |
|---|---|---|
| 框架 | 单方法 Wheeled-SRL (SLIP+RL) | **四算法 2×2 对照** (控制空间 × 参考轨迹) |
| 数字 | 97.2% 成功率/52% 训练缩减/跟踪<0.12m (虚构) | 实测: SRL 0.540m / air 21.7% / 存活100% |
| 方法 | 弹簧-质点+车轮动力学 SLIP 方程 (未实现) | 六状态 FSM 相位参考 (关节角/腿长查表) + VMC 雅可比力矩 |
| 观测/动作 | 35D / 8D (虚构) | 纯PPO 297 / SRL 315 / VMC 387; 动作 8D |

**用户拍板**: (1) 论文定位改为**四算法 2×2 对照研究**; (2) 实验数字**重跑 compare_jump 实测**保证可复现。

## 做了什么

1. **main.tex 全篇重写** (latex/Wheeled-SRL-Jumping/main.tex):
   - 标题: "融合SLIP-FSM相位参考与强化学习的两轮足机器人跳跃控制: 控制空间与参考轨迹的2×2对照研究"
   - 中英摘要/关键词重写为真实数据; 引言缺口聚焦"控制层 vs 参考轨迹"两维度
   - 方法: 任务定义+2×2框架 / 平台 (xqrobotwl 真实参数, MuJoCo) / 相位门控奖励 (含 launch_rise 反偷懒方程) / SLIP-FSM 六状态 / VMC 层 (FK标定+雅可比+阶段增益) / 网络训练
   - 实验: 协议 / 主对比表 / 训练曲线 / 纯PPO 奖励工程失败模式 (v10推而不蹲/v11推而不起/v12突破) / 跳跃轨迹+关节角
   - 讨论 (参考给协调与高度, VMC 补稳定性不补高度) / 结论 / 局限
2. **奖励相位校准**: landing_soft 实为"真实腾空落地段 + landing_timer 门控" (非 p≥30), 已按 jump.py 实际代码修正; 引用修正 §3.6/§3.7→§3.6
3. **参考文献核实与补全** (data/ref.bib):
   - WebSearch 核实: hu2026srl (arXiv:2606.18625 真实, 双足+四足, 六状态FSM+PPO加权融合 — 本项目 SRL 原型)、ATRos (arXiv:2510.09980)、FLORES (arXiv:2507.22345, RA-L 2026)、zeng2026jumping (arXiv:2602.21612)、patrizi2026rlmpc (RA-L 2026, arXiv:2603.10878) 均真实
   - 修正 4 条 bib 细节 (作者/期刊/arXiv号); 新增 schulman2017ppo / pratt2001vmc / rudin2022learning / todorov2012mujoco
4. **图件重生成** (latex/Wheeled-SRL-Jumping/figures/):
   - 修复 make_paper_figures.py ROOT 解析 bug (tools/ 位置多上一层 → /home/robot/xiaoq); 输出路径指向论文 figures/; 验证图 JSON 路径指向论文 data/
   - 新增 tools/make_framework_figure.py → 2×2 设计矩阵+数据流 framework.png/pdf
   - 新增 tools/render_robot_static.py → xqrobotwl_render.png (默认站姿渲染)
   - 修复 plot_jump_trajectory.py 路径 (jump_management 已删 → 论文 data/ + figures/); record_jump_trajectory.py 录制四算法单跳 NPZ → plot 生成 paper_fig_trajectory / paper_fig_jump_joints
   - make_paper_figures.py → paper_fig_training (2×2) / paper_fig_training_metrics (图4.1, 2×3)
5. **.latexmkrc 新增**: xelatex + biber + out/ 目录 (compile.sh 裸 latexmk 会走 pdflatex 报错)

## 关键数据 (compare_jump 重跑实测, 4速度×5集, 2026-08-11)

| 变体 | 跳高(m) | 腾空率 | 存活率 | 成功率 |
|------|--------|--------|--------|--------|
| **SRL** | **0.547** | **0.221** | **1.00** | 1.00 |
| PPO+VMC | 0.175 | 0.175 | 0.95 | 1.00 |
| VMC+SRL | 0.352 | 0.093 | 0.80 | 1.00 |
| 纯PPO (model_1000) | 0.264 | 0.086 | 0.35 | 1.00 |

与 devlog 28 (08-09 测量) 对比: 跳高高度稳定 (SRL 0.540→0.547 等), 存活率略降
(纯PPO 50→35%, VMC+SRL 90→80%, PPO+VMC 100→95%) — 评估含初始态随机性; 叙事更强:
"纯PPO 存活率仅 35%, VMC 提至 95%; SLIP-FSM 参考给高度与协调, VMC 补稳定性不补高度"。

数据存档: `latex/Wheeled-SRL-Jumping/data/four_algo_comparison.json`

## 验证方法

- 四算法最终 checkpoint 重跑 `tools/xqrobotwl/compare_jump.py` (4速度×5集, 与 devlog 28 相同协议):
  - SRL = XqRobotWLJumpSRLFlat/2026-08-06_01-16-20/model_9999
  - PPO+VMC = XqRobotWLJumpVMC/2026-08-09_01-21-12/model_9999
  - VMC+SRL = XqRobotWLJumpSRLVMC/2026-08-08_01-05-51/model_9999
  - 纯PPO = XqRobotWLJumpFlat/2026-08-09_01-21-11/model_1000 (最优早期 ckpt, 后期发散须如实说明)
- 图件全部生成 exit 0; 框架图/轨迹图/训练指标图目视核验 (中文字体 SimHei 正常)
- 论文 PDF 编译验证 (latexmk -xelatex + biber, .latexmkrc 新增): **成功, 14 页, 无未定义引用/缺失图**
- 门禁: ruff format/check 全绿; mypy src/unilab + pyright (仅 include src/unilab, tools/ 不检查) 不受影响; test_repo_hygiene + test_check_docs **24 passed**
- 论文图 6 张全部生成: framework (2×2矩阵+流水线) / xqrobotwl_render / paper_fig_validation (腾空+存活柱状) / paper_fig_training_metrics (图4.1 2×3) / paper_fig_trajectory / paper_fig_jump_joints

## 交付物

| 文件 | 说明 |
|---|---|
| `latex/Wheeled-SRL-Jumping/main.tex` | 小论文全篇重写 (2×2 对照研究) |
| `latex/Wheeled-SRL-Jumping/data/ref.bib` | 25 条参考文献 (4 条修正 + 4 条新增, 均已核实真实) |
| `latex/Wheeled-SRL-Jumping/data/four_algo_comparison.json` | 四算法实测数据存档 (2026-08-11) |
| `latex/Wheeled-SRL-Jumping/data/jump_traj_*.npz` | 四算法单跳轨迹数据 |
| `latex/Wheeled-SRL-Jumping/figures/*.png/pdf` | 6 张论文图 |
| `latex/Wheeled-SRL-Jumping/.latexmkrc` | xelatex+biber 编译配置 |
| `latex/Wheeled-SRL-Jumping/out/main.pdf` | 编译产物 (14 页) |
| `tools/make_framework_figure.py` / `tools/render_robot_static.py` | 新增论文图脚本 |
| `tools/make_paper_figures.py` / `tools/xqrobotwl/plot_jump_trajectory.py` | 修复路径 (jump_management 已删) |

## 后续计划

- [x] 论文图 caption 数字按实测微调 (已完成)
- [ ] 备份: latex/ 论文目录版本备份 (§6)
- [ ] 提交 (用户自行 commit)
- [ ] 按审稿意见迭代
