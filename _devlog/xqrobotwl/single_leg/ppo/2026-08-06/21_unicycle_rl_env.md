---
日期: 2026-08-06
来源: 改进结构单轮平衡 RL env 实现 (devlog 20 结构验证通过后, 用户选 RL 路线)
问题描述: 为改进结构 (横躺+左腿支撑+直配重+腿kp300+轮扭矩源) 建独立 RL env 并从零训练.
解决方案/实现:
  1) **XML 加 2 个观测传感器** (xqrobotwl.xml, 不影响物理):
     - `<framexaxis name="basexvector">`: base 长轴向量 — 关键! 横躺姿态下 pitch
       (绕轮轴=世界y) 在 up 向量不可见 (up∥轮轴), 必须用长轴测
     - `<framepos name="left_wheel_world_pos">`: 左轮世界坐标 (离地检测)
  2) **新 env `single_leg_unicycle.py`** (注册 XqRobotWLSingleLegUnicycle, 继承 WalkFlat):
     - __init__ 运行时改执行器: 腿 kp=300 kv=10 (gainprm[0]=300, biasprm[1]=-300,
       biasprm[2]=-10), 轮=扭矩源 (gainprm[0]=1, biasprm[2]=0) — 膝防塌 + 倒立摆正确物理
     - apply_action: 钉住支撑腿 (L_roll=1.49,L_pitch=-0.1,L_knee=-0.2,R_knee=0,R_wheel=0),
       RL 控 [R_roll(+0.5rad), R_pitch(+0.5rad), L_wheel(扭矩×8, ±20Nm)]
     - obs 帧 33→36D (加 basex 3D), critic 39D; reset 横躺 quat Rx90 + base_z=0.6776
     - 终止: 左轮离地>2cm / up或basex对齐<0.80 / base_z<0.30
     - 奖励: upright(up+xvec对齐,×10) + wheel_down(×3) + counterweight_pose(-0.5)
       + damping(-0.5) + action_rate(-0.1) + alive(1)
  3) **配置** `conf/ppo/task/xqrobotwl_single_leg_unicycle/mujoco.yaml`: 第一阶段只保持
     (vx=0), num_envs=1024, max_iterations=2万 (实际先跑 5000 观察)
  4) **踩坑**: get_body_pos_w 需要 add_body_sensors (重), 改用 framepos sensor;
     reset 时 _compute_obs 收到子集 batch, basex 须按 linvel.shape[0] 切片
  5) **奖励设计迭代** (训练实测):
     - v1 无漂移惩罚: 平台 2.75s (episode 275 步), 可能漂移倾覆
     - v2 线速度惩罚 (lin_vel -0.8): 平台 1.2s **反更低** — 倒立摆平衡必须让
       轮子滚动产生速度, 罚速度阻止了平衡动作
     - v3 位置漂移惩罚 (drift -0.3 × base位移²): 允许轮子微动, 只罚净漂移
  6) **验证**: smoke test (执行器参数✓ obs shape 324/351✓ 左轮贴地✓ 30步不崩);
     训练冒烟 39 iter: reward 4.19, episode 58步(0.58s) — env 信号有效
  7) **训练结果 (v3, 位置drift惩罚)**: model_1000 eval 平均 **4.1s 单轮平衡**
     (PD 基线 0.86s 的 4.8 倍), 轮贴地 100%, pitch/roll 对齐 0.98/0.99.
     失败模式: 缓慢耦合 tilt 漂移 (~4s 倾到37°阈值) — 平衡机制正常, 补偿不足.
     ⚠️ eval 脚本 bug: ep_len 对已 done env 继续累加 (env自动reset), 修复为
     active mask 冻结. 之前的"600/600"是假象.
  8) 训练 v3 继续至 5000 iter (monitor checkpoint 事件).
修改文件:
  - src/unilab/assets/robots/xqrobotwl/xqrobotwl.xml (+basexvector, +left_wheel_world_pos)
  - src/unilab/envs/locomotion/xqrobotwl/single_leg_unicycle.py (新建)
  - src/unilab/envs/locomotion/xqrobotwl/__init__.py (+import)
  - conf/ppo/task/xqrobotwl_single_leg_unicycle/mujoco.yaml (新建)
验证方法: smoke test + 40 iter 训练冒烟 (reward/episode 有效)
后续计划: 5000 iter 全量训练 → 监控 reward/episode_length → 评估 + 渲染视频
关联日志: 20 (结构搜索, 物理可行), 19 (kp bug), 18 (v4评估)
---
