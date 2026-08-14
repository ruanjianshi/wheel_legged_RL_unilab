"""MPC×SAC 融合控制分支 — 高层 SAC 决策 + 低层 MPC 执行 (AugMPC 层次化移植).

独立任务轨 (自包含): 自带线性 MPC (mpc/) + 紧凑 SAC (sac/),
只读复用共享基础设施 common/ (§3.2). 命名 = 融合算法对: MPC + SAC。
"""
