# [13] 迁移 scripts/{tools,xqrobotwl} → tools/ + pose_data 保留两位小数

**日期**: 2026-08-10
**来源**: 用户指令 — "这两个文件夹放到 tools 这里去" (scripts/tools 与 scripts/xqrobotwl), "logs/pose_data 保留两位小数即可"
**关联**: [[12_reorganize_per_updated_spec]]

---

## 做了什么

### 1. 目录迁移 (git mv 保留历史)
- `scripts/xqrobotwl/` (28 个任务脚本) → **`tools/xqrobotwl/`**
- `scripts/tools/` 的 4 个全仓库工具 **平铺到既有 `tools/` 根**:
  `analyze_offpolicy_trace.py / audit_sim2sim_contracts.py / generate_support_matrix.py / make_paper_figures.py`
  + `__init__.py` → `tools/__init__.py`
- **平铺决策**: 根目录 `tools/` 已存在 (email/mujoco/pinocchio_traj/xqrobotV2/eval_flat_walk.py),
  若按字面迁会得到 `tools/tools/` 冗余嵌套, 故把"全仓库工具"并入 `tools/` 根, 任务脚本独立成 `tools/xqrobotwl/`
- `scripts/` 保留 `training/ play/ manip_loco/ completions/ __init__.py` 不动

### 2. 引用更新面 (全仓零残留)
| 类别 | 文件 |
|---|---|
| Python import (2) | `tools/xqrobotwl/{dump_pose_data,verify_jump_trajectory}.py`: `from scripts.xqrobotwl` → `from tools.xqrobotwl` |
| 迁移脚本内部 docstring (32 处) | `tools/xqrobotwl/*.py` + `tools/*.py`: `scripts/xqrobotwl/` → `tools/xqrobotwl/`, `scripts/audit*` → `tools/audit*` |
| src (5) | `sim2sim.py` / `vmc.py`×2 / `single_leg.py` / `backflip.py` |
| shell (1) | `monitor_training.sh` |
| conf (2) | `jump_vmc_flat` + `jump_srl_vmc_flat` 的 mujoco.yaml (calibrate 脚本注释) |
| tests (2) | `doc_checks.py` / `test_train_scripts.py` (加载候选加 `_ROOT_DIR/tools/`) |
| 文档 (8) | `CLAUDE.md` (含 §4.1 树) / `README.md` / `docs/sphinx/{AGENTS.md, 3 篇}` / `docs/CLAUDE copy.md` / `video/README.md` / `docs/project_tree.md` |

### 3. pose_data 保留两位小数
- `dump_pose_data.py` 写 CSV 时 float 列 `round(v, 2)`, step/接触/恢复锁存整数列不受影响
- 已存在的样例 `logs/pose_data/fall_recovery_model_499_仰卧_supine.csv` 同步圆整

## 验证方法

- `make format` ✅ / `make type` ✅ (mypy 201 文件, pyright 0 错)
- 全量 `uv run pytest`: **796 passed / 0 failed**
- 导入冒烟: `from tools.xqrobotwl.dump_pose_data` / `from tools.audit_sim2sim_contracts` 等均 OK
- infer 工具在两位小数 CSV 上分类正常 (阈值粗粒度, 边界帧微动属预期)
- 全仓 grep 确认 `scripts/xqrobotwl` / `scripts/tools` 零残留

## 遗留说明

- `backup/` 快照内 README/src 仍引用旧路径 `scripts/xqrobotwl/...` — **历史快照不动**
- `scripts/` 目录仍在 (training/play/manip_loco), 未迁移
- 规范 CLAUDE.md 已同步新结构 (§1.2/§1.5.1/§2.3/§2.4/§3.1/§4.1)

## 后续计划

- [ ] 提交 (用户自行 commit, 不推送)
