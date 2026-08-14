# 经典控制轨 (LQR + MPC)

两轮足机器人 xqrobotwl 的经典(模型基)平衡控制器 — 独立任务轨, 不修改任何 RL 文件。
代码在 `scripts/classic_control/`, **算法配置在 `conf/classic_control/`**(结构与 RL 一致)。

## 配置结构 (对齐 RL)

```
conf/classic_control/
├── config.yaml                    # ★ 算法配置: 矢状面权重/MPC/偏航/腿控/命令表
└── task/
    ├── xqrobotwl_walk_flat.yaml   # 任务 env 映射 + 机器人常数 (站姿/腿目标/VMC/限位)
    └── xqrobotwl_walk_rough.yaml  # rough 覆盖 (仅 env 差异, 复用 robot 常数)
```

- 控制器参数经 `config.py::load_config()` 加载 → `default_params()` → `SagittalConfig`
- CLI 覆盖合并: `SagittalConfig(**{**default_params(task), **overrides})`
- 调参只需改 `conf/classic_control/config.yaml`, 无需动代码
- ★ 站姿/腿目标为**左右镜像对称** (几何对称化后, 见 common devlog #03); 平衡全阶段达标

## 能力 (LQR 与 MPC 均已达成 P1-P3)

| 阶段 | 能力 | LQR | MPC |
|---|---|---|---|
| P1 | 稳定平衡 | 15s 存活 ✅ | 15s 存活, gyro 0.227 ✅ |
| P2 | 指令控制 | vx RMSE 0.080 ✅ | vx RMSE 0.093 ✅ (积分增广+参考偏置) |
| P3 | 腿长控制 | 高度 0.48-0.55 追踪 ✅ | height_err 0.039 ✅ |
| P4 | 地形自适应 | rough 20s (对称前) / 现 0% ⚠️ | 与 LQR 并列 ⚠️ (rough 过难, 遗留) |

MPC 说明: 线性 MPC (Hildreth QP, 求解 <1ms), 黑箱模型 `logs/classic/mpc_plant_bb.npz`
(LQR 闭环+探针辨识, `identify_plant.py`), LQR 末端代价 + 控制变化率惩罚。
模型来源优先级: CLI A_d/B_d → config `model_file` → 解析 (α,β,τ)。
P1 用解析模型 (黑箱 P1 漂移不达标); P2+ 用黑箱 (速度跟踪准)。

评估 (确定性, 5 ep): `bash shell/xqrobotwl/classic_control/eval_classic.sh mpc all`

## 交互控制 (MuJoCo 键盘)

```bash
bash shell/xqrobotwl/classic_control/play_classic.sh              # 平地
bash shell/xqrobotwl/classic_control/play_classic.sh rough        # 粗糙地形
```
键盘: ↑/↓ 前进后退 · ←/→ 转向 · Q/E 腿长 · Enter 停止 · Space 暂停

## 评估

```bash
bash shell/xqrobotwl/classic_control/eval_classic.sh lqr 1      # LQR P1 平衡
bash shell/xqrobotwl/classic_control/eval_classic.sh lqr all    # LQR 全阶段
uv run mjpython scripts/classic_control/balance_lqr.py --phase 2 --sim_time 40
```

## 渲染

```bash
uv run mjpython scripts/classic_control/render_balance_demo.py --demo success_env  # P1
uv run mjpython scripts/classic_control/render_balance_demo.py --demo p2|p3|p4
```

## 关键根因 (开发记录)

控制方向曾反了 (前倾需后驱, 曾用前驱 → 正反馈失败)。修复后 LQR 平衡达成。
详见 `_devlog/xqrobotwl/classic_control/` #02-#06。

## 限制 (如实)

- MPC (线性, Hildreth QP): 植物模型不确定 (轮打滑+腿耦合) → 未稳定平衡。
- P3 高度范围受限: kp=60 位置腿 + 腿长范围 → 仅 0.48-0.55。
- P2 高速转向 (vyaw≥0.3): 差速削弱平衡权威 → 建议低速转向。
