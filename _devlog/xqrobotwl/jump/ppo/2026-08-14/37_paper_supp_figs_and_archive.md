# 37 论文补充图 5 张 + 图片归档规范 picture/paper/[任务]/[内容]/[版式]/[版本]

## 日期
2026-08-14

## 来源
用户提供推荐补充图清单 (主图 5 张), 并要求论文数据/图片统一归档到
`picture/paper/[任务]/[内容]/[版式]/[版本]`。

## 修改了什么
1. **扩展轨迹采集** `tools/xqrobotwl/verify_jump_trajectory.py`:
   - 新增 3 个字段: `hip_roll` (髋滚转), `base_euler` (机身 ZYX 欧拉角 [roll,pitch,yaw]),
     `linvel` (机身本地系线速度) — 复用 dump_pose_data 的 `_quat_to_euler` / `_world_to_local`
   - DOF 索引补 hip_roll: `DOF = {"hip_roll": [0,3], "hip": [1,4], "knee": [2,5]}`
   - 输出目录 `jump_management/results/` 补建, 重新采集 4 算法轨迹, 峰值与论文一致
     (SRL 0.545m / PPO 0.255m 终止 / PPO+VMC 0.197m / SRL+VMC 0.330m)
2. **新建** `tools/make_paper_supp_figs.py` — 按 nature-draw v2.0 规范出 5 张补充图:
   - Fig.1 `fig_jump_flat_training_overview_2x2_v2` — 训练全景: 奖励/存活(回合)/动作std/value loss
   - Fig.2 `fig_jump_flat_final_perf_2x1_v2` — 最终性能: 跳跃高度 + 最终平均奖励 (柱状)
   - Fig.3 `fig_jump_flat_traj_vel_2x1_v2` — 跳跃轨迹: 机身高度 + 前向速度
   - Fig.4 `fig_jump_flat_joints_2x3_v2` — 关节角: 髋pitch/膝/髋roll 左右腿 (SRL vs PPO)
   - Fig.5 `fig_jump_flat_reward_split_2x2_v2` — 奖励分项拆解 (4 算法各面板, term 排序)
3. **图片归档** `picture/paper/jump/`:
   - 9 张图 (5 主图 + 5 补充图, 同名 PDF+PNG) 按 `[内容]/[版式]/[版本]` 归档
   - framework (1x2), robot 渲染 (1x1), data/ 数据文件 (json+4 npz+bib)

## 哪些文件
- 修改: `tools/xqrobotwl/verify_jump_trajectory.py` (3 字段 + hip_roll 索引 + 输出目录)
- 新建: `tools/make_paper_supp_figs.py`
- 生成: `latex/Wheeled-SRL-Jumping/figures/fig_jump_flat_{training_overview,final_perf_2x1,traj_vel,joints_2x3,reward_split}_*.{pdf,png}`
- 数据: `latex/Wheeled-SRL-Jumping/data/jump_traj_*.npz` 更新 (含新字段) + `jump_management/results/`
- 归档: `picture/paper/jump/` (图 + 数据)

## 训练后效果 (视觉验证)
- Fig.1: SRL 全程领先, 纯PPO 后期发散 (奖励骤降/回合骤减/loss 激增), 支撑"无参考发散"
- Fig.2: 跳高柱 SRL 0.547 最高; 最终奖励 SRL 最高, 纯PPO 虽末段均值尚可但方差大
- Fig.3: 高度+前向速度双面板, SRL 速度峰 1.67m/s 最高, PPO 终止带 × 标记
- Fig.4: SRL 膝角先屈后伸时序结构 vs PPO 缺乏, 髋roll 微小 (对称性)
- Fig.5: SRL 各 reward term 均值排序, crouch/landing 高分解释"为什么好"

## 参数调整好坏
- 补充图命名遵循 nature-draw §九: `fig_[步态]_[任务]_[内容]_[版式]_[版本]`
- 奖励分项排除 vertical_thrust/height_progress (PPO 系用 launch_rise, 无此 tag), 避免缺项误导
- 每算法末 500 iter EMA 均值作最终值, 比末点更稳 (纯PPO 末段噪声大)

## 根因分析
用户推荐清单中 Fig.3 (前向速度) / Fig.4 (髋roll) / 姿态稳定性 (pitch/roll) 需要现有 npz
未采集的速度/欧拉角/髋roll 数据 → 扩展采集脚本重跑。

## 验证方法
1. 逐张 Read 检查 PNG 渲染 (配色/数值/相位色带/时序结构正确)
2. 采集峰值与 four_algo_comparison.json 对照一致 (0.545≈0.547, 0.197≈0.175, 0.330≈0.352)
3. 新字段数值合理: SRL linvel_x 峰 1.67m/s, pitch -0.47~0.09, hip_roll 微小
4. 归档目录 `picture/paper/jump/` 结构符合 `[任务]/[内容]/[版式]/[版本]`

## 后续补充 (用户决策: 全部替换为主图5张)
- main.tex 更新 (用户确认"全部替换为主图5张"):
  - fig:validation 4.1: final_perf 3×1 → **final_perf 2×1** (跳高+最终平均奖励)
  - fig:training_metrics 4.2: training 2×3 → **training_overview 2×2** (奖励/回合/std/value loss)
  - 新增 §4.4 奖励分项拆解: **reward_split 2×2** (fig:reward_split 4.3)
  - fig:joints 4.4: joints 2×2 → **joints 2×3** (髋pitch/膝/髋roll, SRL vs PPO)
  - fig:trajectory 4.5: traj 2×2 → **traj_vel 2×1** (高度+前向速度)
  - 正文同步: 训练曲线段补 value loss 说明, 轨迹段补速度/髋roll 描述, 新增奖励分项小节
- 编译: latexmk -xelatex exit=0, 7 图全部嵌入, 14 页

## 后续计划
- 多 seed 箱线图不可行 (每算法仅 1 seed) — 已在汇报中说明
- 姿态稳定性 (pitch/roll) 图可用新 base_euler 字段快速补 (2×1)
- 13 任务重训仍在后台 (task #115)

## 关联日志
- [[36_paper_figs_v2_nature_draw]] 主图 v2.0 重出 (IBM 配色)
- [[30_paper_figs_nature_style]] 上一版论文图规范
