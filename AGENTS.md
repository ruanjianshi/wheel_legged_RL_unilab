# UniLab Agent Principles

**Always use `uv run`, not python**.

UniLab 是一个 **高性能、模块化、contract 驱动** 的 RL infrastructure 仓库。

---

## AI 自循环开发系统 (Self-Loop)

AI 开发遵循闭环迭代：**训练 → 监控 → 评估 → 日志 → 分析 → 改进 → 自检 → 再训练**，循环直到达成目标。达成后自动发送邮件报告。

```
┌─────────────────────────────────────────────────────────────────┐
│                     AI Self-Loop 流程图                          │
│                                                                 │
│   用户设定目标                                                    │
│      │                                                          │
│      ▼                                                          │
│   ┌──────┐    监控     ┌──────────┐                             │
│   │ 训练  │───定期───→│ 训练曲线   │                             │
│   │      │   检查     │ TensorBoard│                             │
│   └──┬───┘           └────┬─────┘                              │
│      │                    │                                     │
│      │   达到检查点        │ 未达标                               │
│      │   (每 1000 iter)   │ (继续训练)                         │
│      ▼                    │                                     │
│   ┌──────────┐            │                                     │
│   │ 评估策略  │←───────────┘                                     │
│   │ assess   │                                                  │
│   └────┬─────┘                                                  │
│        │                                                        │
│        ▼                                                        │
│   ┌──────────┐                                                   │
│   │ 写开发日志 │── _devlog/<task>/<algo>/<date>/<n>_<slug>.md   │
│   │ _devlog  │                                                  │
│   └────┬─────┘                                                  │
│        │                                                        │
│        ▼                                                        │
│   ┌──────────┐                                                   │
│   │ 分析结论  │                                                   │
│   │          │   查: tracking 误差, 稳定性, 对称性, 高度, 步态    │
│   └────┬─────┘                                                  │
│        │                                                        │
│        ▼                                                        │
│   ┌──────────┐   达标?                                            │
│   │ 判断目标  │──Yes──→ ┌──────────┐ → ┌──────────┐              │
│   │          │         │ 发送邮件   │   │ ✅ 完成   │              │
│   └────┬─────┘         │ to 我     │   └──────────┘              │
│        │ No            └──────────┘                              │
│        ▼                                                        │
│   ┌──────────┐                                                   │
│   │ 改进方案  │   改: 超参/奖励权重/课程/地形/域随机化/网络结构     │
│   │          │   原则: 每次只改一个变量, 对照实验                   │
│   └────┬─────┘                                                  │
│        │                                                        │
│        ▼                                                        │
│   ┌──────────┐                                                   │
│   │ 自检验证  │   跑: make format, make type, pytest              │
│   │          │   原则: 不通过不进入训练                            │
│   └────┬─────┘                                                  │
│        │                                                        │
│        ▼                                                        │
│   ┌──────┐                                                       │
│   │ 再训练 │ → 回到监控, 循环                                      │
│   └──────┘                                                       │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 1: 训练启动

```bash
# 单 GPU 训练 (后台运行 + 日志)
setsid bash -c 'CUDA_VISIBLE_DEVICES=0 uv run train --algo ppo --task xqrobotV2_walk_flat --sim mujoco' \
  &>/tmp/flat_train.log & disown

# 或使用便利脚本
CUDA_VISIBLE_DEVICES=0 bash shell/xqrobotV2/flat/train_ppo_flat.sh

# 查看训练日志
tail -f /tmp/flat_train.log
```

**训练启动检查清单**:
- [ ] 确认 GPU 空闲 (`nvidia-smi`)
- [ ] 确认训练输出目录已创建 (`logs/rsl_rl_ppo/XqRobotV2WalkFlat/<timestamp>/`)
- [ ] 确认 TensorBoard 可访问 (`uv run tensorboard --logdir logs/`)
- [ ] 写入出发日志到 `_devlog/`

### Phase 2: 定期监控

AI 应**主动**每隔指定 iter 检查训练状态，不等待用户指示。

```bash
# TensorBoard 启动 (后台)
bash shell/xqrobotV2/tools/tensorboard.sh flat 8080  # → http://localhost:8080

# 查看最新训练日志摘要
tail -100 /tmp/flat_train.log | grep -E "iter|mean_reward|mean_episode_length|action_std|tracking"

# 列出可用 checkpoint
ls logs/rsl_rl_ppo/XqRobotV2WalkFlat/<run>/model_*.pt | sort -t_ -k2 -n

# 查看最新 iter
grep -oP 'iter: \d+' /tmp/flat_train.log | tail -1
```

**监控指标正常/异常判断**:

| 指标 | 正常趋势 | 异常信号 | 处理 |
|------|----------|----------|------|
| `mean_reward` | 持续上升 | 停滞 >500 iter 不增 | 调 lr 或奖励权重 |
| `mean_episode_length` | 持续增长到 max | 突然下降 | 检查 DR 或课程 |
| `action_std` | 缓慢下降 (0.5→0.05) | 快速塌缩到 <0.01 | entropy_coef 太低 |
| `tracking_lin_vel` | 持续上升 | 波动剧烈, 不收敛 | 检查奖励 scale |
| `base_height` | 收敛到 target±0.05 | 持续偏移 | 检查 reward_scale |
| `orientation_error` | < -0.5 (reward high) | > -5.0 (机器人倾斜) | 检查终止条件 |
| `mean_loss` | 平稳 < 0.1 | 爆炸 > 10 | 梯度裁剪/学习率 |

**AI 应在训练异常时主动介入**：分析 TensorBoard 曲线，提出改进方案。

### Phase 3: 策略评估

每个达到里程碑的 checkpoint (每 1000 iter 或训练结束) 都须评估。

```bash
# 基础评估 (decoupling 6 场景)
uv run assess/runner.py -t <task> -a <algo> -r <run> -c <iter>

# 全量评估 + 绘图 + CSV + 报告
uv run assess/runner.py -t <task> -a <algo> -r <run> -c <iter> \
    -s full --plot --csv --report

# 跨 checkpoint 趋势
uv run assess/runner.py -t <task> -a <algo> -r <run> \
    --trend --ckpts 1000,2000,3000,4000,5000 -s decoupling

# 跨模型对比
uv run assess/runner.py --cmp \
    results/<task>/<algo>/<s1>/metrics.json \
    results/<task>/<algo>/<s2>/metrics.json --plot
```

**task 与 env 自动适配**: `assess/runner.py` 根据 `-t` 自动选择对应环境。
- `flat_walk` → 平地环境
- `rough_walk` → 粗糙地形环境
- `toe_walk` → 脚趾行走环境

### Phase 4: 开发日志

每次 Phase 3 评估完成后，**必须**写入分析日志。详见 [AI 开发日志纪律](#ai-开发日志纪律)。

通过分析报告，得出本次循环"结论"。关键不是命令跑完，而是要提炼出**量化结论**：
实验是否起效？Vx/Vy 解耦了没？跟上一个 checkpoint 比有没有进步？等等。

### Phase 5: 判断目标

根据评估结果和用户设定的目标判断是否完成：

| 目标 | 判断标准 | 指标 |
|------|----------|------|
| 行走 | Vx/Vy 跟踪误差 < 0.1, 存活 ≥ 95% | vx_rmse, survival_rate |
| 稳定 | base_height_rmse < 0.05, max_tilt < 15° | base_height_rmse, max_tilt |
| 解耦 | Vy crosstalk < 0.05 (仅 Vx 指令时) | vy_xtalk |
| 后退 | Vx=-0.3 时 AvgVx < -0.25 | avg_velocity_x |
| 侧移 | Vy=±0.3 时 AvgVy > 0.25 | avg_velocity_y |
| 能效 | COT < 2.0 | cot_xy |

**达成目标 → Phase 5.5 发邮件报告，然后停止循环。未达成 → 进入 Phase 6。**

### Phase 5.5: 发送最终报告

任务达标后，自动生成 HTML/文本报告并发送邮件给开发者。

```bash
# 环境变量 (密码不暴露在命令行)
export UNILAB_SMTP_USER=qfantastic@2925.com
export UNILAB_SMTP_PASS=<密码>

uv run tools/email/report.py -t <task> -a <algo> -r <run> -c <ckpt>

# 一键 (评估 + 发邮件)
UNILAB_SMTP_USER=qfantastic@2925.com UNILAB_SMTP_PASS=<密码> \
  bash tools/email/send.sh <task> <algo> <run> <ckpt>
```

报告内容: 任务名、算法、训练参数、分场景评估数据、综合指标、自动结论（✅/⚠️/❌）。

该工具也支持 `--preview` 仅预览不发送，或 `--to` 指定其他收件人。

### Phase 6: 改进方案

分析 Phase 3 的评估报告，识别瓶颈，提出改进。

**改进原则**:
1. **单一变量**: 每次只改一个条件，以确定因果关系
2. **对照实验**: 保留旧实验作为基线，新实验直接对比
3. **可逆**: 所有改动通过 YAML 配置，不硬编码
4. **量化**: 每次改动必须提出预期指标变化

**改进手段优先级** (从低风险到高风险):
1. 调整奖励权重 (reward.scales) — 最低风险
2. 调整课程参数 (curriculum.*) — 低风险
3. 调整 PPO 超参 (entropy_coef, lr, noise_std) — 中风险
4. 调整命令范围 (vel_limit) — 中风险
5. 调整地形配置 (terrain proportions) — 中风险
6. 调整网络结构 (hidden_dims) — 高风险
7. 调整观测空间 (obs groups) — 高风险
8. 调整机械结构 (XML, keyframe) — 最高风险

### Phase 7: 自检验证

每次代码/配置修改后，进入 Phase 7。

```bash
# 格式检查
make format     # ruff format

# 类型检查
make type       # mypy

# 单元测试
make test       # pytest

# 全量检查
make test-all   # format + type + test
```

**门禁规则**: `make test-all` 未通过 → 禁止启动训练，回退修改。

通过后写入改进日志到 `_devlog/`，然后进入 Phase 1 → 启动新一轮训练。

### Phase 8: 完整循环示例

```
Iter 0:     出发日志: "启动 flat_walk PPO 训练, entropy=0.002, hips=-0.1/+0.1"
Iter 1000:  评估 → action_std=0.40 太高 → 调 entropy=0.01 → 自检通过 → 再训练
Iter 3000:  评估 → Vx tracking=0.85 → Vy 串扰=0.65 → 分析: hip 不对称
           → 改 hips 为对称外展 → 自检通过 → 再训练
Iter 4000:  评估 → Vy 串扰=0.04 ✅ → Vx tracking=0.95 → 后退弱
           → 增加后退命令比例 → 自检通过 → 再训练
Iter 5000:  评估 → Vx RMSE=0.04, Vy 串扰<0.06, 存活 95% ✅
           → 完成日志: "flat_walk 达成目标"
           → 发送邮件报告 → qfantastic@2925.com
```

---

## AI 编码行为规范

减少 LLM 常见编码错误。对于简单任务可按判断放宽。

### 1. 先想再写

**不假设、不隐藏困惑、暴露权衡。**

动手前：
- 明确说出你的假设。不确定就问。
- 如果一个需求有多种理解，全部列出——不要默默选定一种。
- 如果有更简单的做法，指出来。
- 如果有地方不清楚，**停下来**，说出困惑点。

### 2. 简洁优先

**最少代码解决问题。不做猜测性开发。**

- 不写用户没要求的功能。
- 不为单次使用建抽象层。
- 不为了"灵活性"或"可配置性"做额外工作。
- 不处理不可能发生的异常场景。
- 如果写了 200 行可以精简到 50 行，重写。

自问：「资深工程师会觉得这段代码过度设计吗？」如果会，简化。

### 3. 精准修改

**只改必须改的。只清理自己造成的残留。**

修改已有代码时：
- **不要**"优化"旁边的代码、注释、格式。
- **不要**重构没坏的东西。
- 匹配现有代码风格，即使你会写得更漂亮。
- 如果注意到无关的死代码，口头提一句——不要删除。

当你的修改产生孤立依赖：
- 删除**你的修改**导致的无用 import / 变量 / 函数。
- 不删除之前就存在的死代码，除非被要求。

**检验标准**：每一行变更都能追溯到用户的需求。

### 4. 目标驱动

**定义成功标准。循环直到验证通过。**

把任务转化为可验证的目标：
- "加校验" → "先写非法输入的测试，然后让它通过"
- "修 bug" → "先写复现的测试，然后让它通过"
- "重构 X" → "确保测试前后都通过"

强成功标准让你自主循环推进。弱标准（"能用就行"）需要不断追问。

---

## Core Principles

1. **Contract first**: 不为了一次通过绕过 env / backend / runner contract。
2. **Fix at owner layer**: `scripts/` 只组装流程，不承载长期业务规则。
3. **Config first**: task / reward / backend 优先通过 Hydra + registry 表达。
4. **Backend isolation**: MuJoCo / Motrix 差异留在 backend 适配层和配置层。
5. **Evidence only**: support claim 只写仓库里已有的注册、配置、测试或 benchmark 事实。
6. **Validate near risk**: 在最接近风险的边界补验证，不只跑顶层命令。
7. **Cold-path asset access only**: asset/XML/model metadata 只允许在 init / materialization / cache 等低频路径处理；热路径不能解析 asset，也不能靠 `getattr` / `hasattr` 探测 backend 私有能力。
8. **Self-Loop Mandate**: AI 必须主动推进训练→评估→分析→改进 闭环，不等待用户逐条指令。

---

## High-Risk Areas

| 区域 | 不可破坏的不变量 |
|------|----------------|
| Env  | `NpEnvState.obs` 必须是 dict；`reset()` 返回 `(obs_dict, info_dict)`；`obs_groups_spec` 影响 wrapper 和 learner 维度。 |
| Config / Reward | reward 通过 Hydra 注入；后端切换必须通过 `task=<task>/<backend>` 选择 owner YAML，`training.sim_backend` 只是 owner YAML 的身份字段，不能单独 override 来切后端。算法超参数直接走 YAML compose，不经 Python 层解释。 |
| Backend | backend-specific 逻辑留在 backend / env 适配层，不向训练脚本扩散。env 层只能调用 `SimBackend`（`base.py`）中已声明的方法；若某方法只在 MuJoCo 或 Motrix 中存在，必须先将其加入 `SimBackend` 抽象接口（可抛 `NotImplementedError`），禁止直接在 env 里调用 backend 子类的私有方法（即"功能泄漏/feature leakage"）。新增 backend 专有能力时，需同步更新 `SimBackend`。 |
| Asset / Metadata | `ASSETS_ROOT_PATH`、`model_file`、XML / asset 元数据只允许在 init / materialization / cache 等低频路径访问；`step/reset/domain randomization` 等热路径不得解析 asset 或基于 asset 元数据做运行时分支。 |
| Asset / XML structure | `<keyframe>` 必须放在 task-level XML（`scene_*.xml` 或 `locomotion_task.xml` 等 fragment），**禁止放进 robot.xml**。robot.xml 是纯机器人描述（body / joint / actuator / sensor），跟 task / 场景无关；keyframe 是 task 起始姿态，属于场景或 task 资源。motrix 后端需要 keyframe 时通过 `scene.fragment_files` 引用 fragment XML。 |
| Async | 不绕开 runner lifecycle，也不另起 collector / learner 同步协议。 |
| Sim2Sim 契约 | 跨后端 play 时，影响策略 I/O / 网络结构的字段必须跨后端一致；不一致即 `CrossBackendIncompatibleError`。详见下方 Sim2Sim 章节。 |

---

## Sim2Sim 跨后端配置契约

`src/unilab/training/sim2sim.py` 按 dotted path 维护三类字段：

- **DENYLIST**（差异即 `CrossBackendIncompatibleError`）：`algo.obs_groups`、`env.control_config.action_scale`、`algo.policy.actor_hidden_dims` / `critic_hidden_dims`、`algo.empirical_normalization` / `algo.obs_normalization`、`env.sampling_mode`。`env.*` 子集对**任一方向**的不对称出现也 fail-closed；`algo` 专属字段目标缺省时按设计跳过（跨算法合法）。
- **WARNING_LIST**：`reward.*`、`env.control_config.simulate_action_latency`、`env.ctrl_dt`。
- **ALLOWLIST**（自由覆盖）：`training.sim_backend`、`env.scene`、`training.play_steps`、`env.domain_rand`、`env.noise_config`、`env.commands.vel_limit`。

训练时 `ExperimentTracker.start()` 把上述字段写入 `run_config.json` 的 `contract_snapshot`（不改 checkpoint 格式，旧 run 无 snapshot 时 fallback + warning）；五个 play 入口在建 env 前调用 `resolve_sim2sim_config` 校验，并用 `policy_load_dim_guard` 包裹 checkpoint 加载以把维度不匹配的隐晦报错重抛为显式诊断。设 `training.sim2sim_strict=false` 可把 DENYLIST 差异降级为 warning（默认 `true`）。DENYLIST 字段应通过 task 的 `base.yaml` 共享（范例：`conf/ppo/task/g1_walk_flat/{base,mujoco,motrix}.yaml`）；跨后端契约审计见 `scripts/audit_sim2sim_contracts.py`。

---

## Pointers

- PPO: `scripts/training/train_rsl_rl.py`
- MLX PPO: `scripts/training/train_mlx_ppo.py`
- APPO: `scripts/training/train_appo.py`
- SAC / TD3: `scripts/training/train_offpolicy.py`
- HIM-PPO: `scripts/training/train_him_ppo.py`
- env contract: `src/unilab/base/np_env.py`
- backend contract: `src/unilab/base/backend/base.py`
- training run helpers: `src/unilab/training/run.py`
- visualization helpers: `src/unilab/visualization/`
- env shared numeric helpers: `src/unilab/envs/common/rotation.py`, `src/unilab/envs/common/math.py`
- MLX rotation helpers: `src/unilab/algos/mlx/common/rotation.py`
- config schema: `src/unilab/structured_configs.py`
- async runner: `src/unilab/ipc/async_runner.py`
- sim2sim 跨后端契约: `src/unilab/training/sim2sim.py`
- **Policy assessment**: `assess/runner.py` — 评估 .pt 模型，按 `task/algo/session` 分类输出
- **Assessment docs**: `assess/README.md` — 指标定义、场景集、使用说明
- **DevLog**: `_devlog/README.md` — 开发日志规范，AI 必须自记录每次变更
- **Email report**: `tools/email/report.py` — 自循环最终报告生成 & 邮件发送

### 便利脚本

| 脚本 | 功能 | 用法 |
|------|------|------|
| `shell/xqrobotV2/flat/train_ppo_flat.sh` | Flat Walk 训练 | `CUDA_VISIBLE_DEVICES=0 bash shell/xqrobotV2/flat/train_ppo_flat.sh` |
| `shell/xqrobotV2/rough/train_ppo_rough.sh` | Rough Walk 训练 | `CUDA_VISIBLE_DEVICES=1 bash shell/xqrobotV2/rough/train_ppo_rough.sh` |
| `shell/xqrobotV2/jump/train_ppo_jump_flat.sh` | Jump 训练 | `CUDA_VISIBLE_DEVICES=1 bash shell/xqrobotV2/jump/train_ppo_jump_flat.sh` |
| `shell/xqrobotV2/toe_walk/train_ppo_toe_walk_flat.sh` | Toe Walk 训练 | `CUDA_VISIBLE_DEVICES=1 bash shell/xqrobotV2/toe_walk/train_ppo_toe_walk_flat.sh` |
| `shell/xqrobotV2/flat/eval_ppo_flat.sh` | 键盘控制评估 | `bash shell/xqrobotV2/flat/eval_ppo_flat.sh --keyboard` |
| `shell/xqrobotV2/tools/tensorboard.sh` | TensorBoard | `bash shell/xqrobotV2/tools/tensorboard.sh flat 8080` |

---

## AI 开发日志纪律

**每次修改代码/超参/架构后，AI 必须写入 `_devlog/`**：

1. 新建日志文件 `_devlog/<task>/<algo>/<YYYY-MM-DD>/<序号>_<slug>.md`
2. 使用 `_devlog/TEMPLATE.md` 模板
3. 更新对应 `INDEX.md`
4. 更新全局 `_devlog/INDEX.md`

**不得跳过**：代码变更、超参调优、bug 修复、评估结论、架构调整。  
**可跳过**：纯查询（无修改）、仅运行命令、临时调试。

在完成代码/配置修改后，主动提示用户"需要写日志"，并执行写入。

### 日志章节要求

| 章节 | 必填 | 内容 |
|------|------|------|
| **日期** | ✅ | YYYY-MM-DD |
| **来源** | ✅ | 触发原因 |
| **问题描述** | ✅ | 具体现象、数据指标、错误日志 |
| **根因分析** | ✅ | 为什么会发生，影响范围 |
| **解决方案** | ✅ | 具体修改内容 |
| **修改文件** | ✅ | 绝对路径 + 行号 + 改动内容摘要 |
| **验证方法** | ✅ | 如何确认修复有效 |
| **评估结果** | 条件 | 若涉及策略修改，必须附 assess 评估数据 |
| **后续计划** | ✅ | 遗留问题、下一步方向 |
| **关联日志** | 条件 | 链接到相关的前序/后续日志 |

---

## GitHub CLI (gh) 速查

### Issue 查看
```bash
gh issue view <number>
gh api repos/<owner>/<repo>/issues/<number> --jq '.body'
```

### PR 创建与管理
```bash
gh pr create --title "标题" --body "内容" --base main
gh pr list
gh pr view
```

### PR Gate

创建或更新 PR 前必须满足：

1. 最终提交已经完成，且 `git status --short --branch` 确认工作树干净。
2. 最终提交已经通过 `make test-all`。
3. 如果用户明确说明已经跑过 `make test-all`，不要重复跑；但必须在 PR body 的 Validation 里记录 `make test-all` 已完成。
4. 如果 `make test-all` 未通过且用户没有明确 override，不要创建或更新 PR。

### CI 工作流查看
```bash
gh run list
gh run list --workflow=<workflow-name>
gh run view <run-id>
gh run list --status=failure
```

### 常用组合
```bash
gh api repos/unilabsim/UniLab/issues/174 --jq '.title, .body'
git push -u origin fix/issue-174-mlx-ppo-config-alignment
gh pr create --title "fix: xxx" --body "Fixes #174" --base main
```

---

## Context

- 项目架构: `README.md`
- 架构标准与验证详情：[docs/sphinx/source/zh_CN/4-developer_guide/0-index.md](docs/sphinx/source/zh_CN/4-developer_guide/0-index.md)
- 协作流程与 PR 规范：[docs/sphinx/source/zh_CN/4-developer_guide/5-contributing_workflow.md](docs/sphinx/source/zh_CN/4-developer_guide/5-contributing_workflow.md)
- 开发者入口（环境、命令、提交规范）：[CONTRIBUTING.md](CONTRIBUTING.md)
- 文档本地构建与发布到 UniLab-doc：[docs/sphinx/README.md#本地发布到-unilab-doc](docs/sphinx/README.md#本地发布到-unilab-doc)
