# video — 渲染视频归档

MuJoCo 渲染出的步态/训练回放视频，按机器人步态分类管理。
规范 CLAUDE.md §2.4: 每个评估通过的 checkpoint 渲染视频, 且**机器人始终在视角范围内**
(相机跟踪 `cam_tracking=True` / `render_states_get_frames_tracking`)。

## 目录规范

```
video/
  <任务>/                          # 对应 CLAUDE.md §3.1 任务隔离表
    <YYYY-MM-DD>_<序号>_<描述>_<关键参数>.mp4
```

## 任务 → 目录映射 (§3.1)

| 任务 | 目录 |
|---|---|
| 平地滚动行走 | `walk/` |
| 点足平地行走 | `toe_walk/` |
| 不平坦地形行走 | `rough/` |
| 平地跳跃 | `jump/` |
| 平地后空翻 | `backflip/` |
| 单腿平衡 (三态) | `single_leg/` |
| 跌倒恢复 | `fall_recovery/` |
| 抬腿上台阶 | `stairs/` (复数, 与 env `stairs.py` / conf `xqrobotwl_stairs` / timeline `stairs.md` 一致) |

> 注: 规范表格写作 `stair/` 单数, 仓库统一用复数 `stairs/` 以与 env/config/timeline 命名对齐。

文件名约定（保证可复现）：
- **日期**：`YYYY-MM-DD`（渲染/训练当天）
- **序号**：同日多条时递增（01, 02...）
- **描述**：一句话说明（如 `开环可行`, `训练成功`, `消融对比`, `参数扫描`）
- **关键参数**：影响结果的主要参数（如 `lean06_ch-025`, `W45`），不追求全量

## 生成方式

`backflip_feasibility.py` 支持 `--render`，可直接归档：

```bash
uv run python tools/xqrobotwl/backflip_feasibility.py \
  --W 0 --launch_lean 0.6 --tlaunch 0.25 --crouch_hip -0.25 \
  --render video/backflip/2026-08-04_01_开环可行_360°落地3.8°.mp4
```

`--render` 传一个**目录**或**mp4 文件路径**均可：
- 传目录 → 目录内生成 `backflip.mp4`
- 传 `xxx.mp4` → 直接写入该文件

## 现有内容

- `backflip/2026-08-04_01_开环可行_360°落地3.8°_lean06_ch-025.mp4` — 开环脚本后空翻，恰好 360°，落地倾角 3.8°，腾空 0.41s（P1 物理可行性验证）
- `single_leg/` — 2026-08-06 单腿平衡早期探索视频（开环/结构改进/RL 训练回放）+ `_frames_*` 中间帧目录（ppm, 已 gitignore, 不追踪）
- 其余任务目录待达标后产出对应 `render_*.py` 视频

## 中间帧说明

渲染过程产生的 `_frames_*` ppm 帧目录是中间产物, 不入库（`.gitignore` 已含
`video/**/_frames_*/`), 交付以 mp4 为准。
