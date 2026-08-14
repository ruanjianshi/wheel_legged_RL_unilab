# 07 目录重构: 代码迁到 scripts/, 算法配置分离到 conf/ (对齐 RL)

## 日期
2026-08-14

## 来源
用户要求(老板):
1. 把 `tools/xqrobotwl/classic_control` 移到 `scripts/` 目录;
2. 把算法配置从 `config.py` 分离, 在 `conf/` 构建配置文件, **保持和 RL 算法类似**。

## 修改了什么

### 1. 代码迁移: tools/ → scripts/
`tools/xqrobotwl/classic_control/`(17 文件)整体移到 `scripts/classic_control/`。
- 全部模块导入改写: `tools.xqrobotwl.classic_control.*` → `scripts.classic_control.*`
- ROOT 深度修正: `parents[3]` → `parents[2]`(目录深了一层)
- shell 脚本 + 各文件 docstring 里的路径同步更新

### 2. 算法配置分离: config.py → conf/classic_control/
新增配置目录(结构对齐 RL `conf/<algo>/config.yaml + task/`):

```
conf/classic_control/
├── config.yaml                    # 算法配置: sagittal/mpc/yaw/commands/leg_control/phases
└── task/
    ├── xqrobotwl_walk_flat.yaml   # env 映射 + 机器人常数(站姿/腿目标/VMC/限位)
    └── xqrobotwl_walk_rough.yaml  # rough 覆盖(仅 env, 复用 robot 常数)
```

`scripts/classic_control/config.py` 改为 **YAML 加载器**(手动 deep-merge, 不依赖 hydra):
- `load_config(task_key)` → 合并 base + task
- `default_params(task_key)` → 扁平参数字典
- `build_cfg_and_overrides(task_key, overrides)` → `(SagittalConfig, merged)`, 分离 dataclass 字段与 smoothing/sign 等控制器字段
- 模块常数(STANDING_ANGLES/VMC/LEG_TARGETS...)改从 YAML 读取, 接口不变
- `PhaseCommands.from_config()` 替代 `PhaseCommands()`
- rollout 的 env 映射(env name / command_dim / mujoco.yaml 路径)改从任务 YAML 读取

## 哪些文件
- 迁移: `tools/xqrobotwl/classic_control/*` → `scripts/classic_control/*`(17 文件)
- 新增: `conf/classic_control/config.yaml`, `conf/classic_control/task/xqrobotwl_walk_{flat,rough}.yaml`
- 改写: `scripts/classic_control/config.py`(加载器), `controller.py`(build_cfg_and_overrides),
  `run.py`(PhaseCommands.from_config), `rollout.py`(env 映射取自配置)
- 更新: `shell/xqrobotwl/classic_control/*.sh`, `scripts/classic_control/README.md`,
  `docs/timeline/classic_control.md`, `_devlog/xqrobotwl/INDEX.md`

## 训练后效果(无辅助确定性评估)

重构不改任何控制逻辑/参数, 数值与迁移前一致。冒烟验证 LQR P1:
- 5s 存活, `tilt_rms=0.110`, `gyro_rms=0.366 ✅`, `linvel_xy_mean=0.066 ✅`(5s 时长下 `stand_hold<10s` 仅因 sim 短)

## 参数调整好坏
- 无参数调整。YAML 中的默认值 = 原 `config.py` dataclass 默认值(逐项核对: q_theta=100, q_v=80, q_z=5, mpc_horizon=20, wheel_smoothing=0.85, leg_balance_kp=-0.3 等全部一致)。
- **顺带修复一个潜在 bug**: 旧代码 `SagittalConfig(**params)` 若 CLI 传 `--sign`(非 dataclass 字段)会 TypeError; 新 `build_cfg_and_overrides` 过滤非字段键, `--sign` 覆盖可正常生效。

## 根因分析
- 之前配置与代码耦合在 `config.py`, 不符合项目 §4 分类规范; 用户要求与 RL 算法配置对齐 → 分离到 `conf/`。
- `tools/` 是任务脚本目录(§4.1), 经典控制是算法轨, 放 `scripts/` 更符合语义。

## 验证方法
1. `uv run python -c "...config 加载/合并/覆盖..."` — YAML 各 section 数值逐项比对通过
2. 4 阶段控制器构造 + act 冒烟(空动力学)— 输出正常
3. `uv run python scripts/classic_control/balance_lqr.py --phase 1 --sim_time 5` — 平衡保持 ✅
4. 门禁: ruff format + check 通过; mypy(src) 通过; pyright 0 errors(4 warnings 为既有可选依赖缺失, 非本次引入)

## 后续计划
- 完整门禁 `make format` / `make type` / `make test-all`
- P1/P3 全时长评估复跑确认指标与迁移前一致
- 交互 `play_classic.sh` 让用户确认无回归

## 关联日志
- [[06_mpc_attempt]] 上一日志
