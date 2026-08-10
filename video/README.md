# video — 渲染视频归档

MuJoCo 渲染出的步态/训练回放视频，按机器人步态分类管理。

## 目录规范

```
video/
  <步态>/                          # 如 backflip, jump, walk_flat, rough
    <YYYY-MM-DD>_<序号>_<描述>_<关键参数>.mp4
```

文件名约定（保证可复现）：
- **日期**：`YYYY-MM-DD`（渲染/训练当天）
- **序号**：同日多条时递增（01, 02...）
- **描述**：一句话说明（如 `开环可行`, `训练成功`, `消融对比`, `参数扫描`）
- **关键参数**：影响结果的主要参数（如 `lean06_ch-025`, `W45`），不追求全量

## 生成方式

`backflip_feasibility.py` 支持 `--render`，可直接归档：

```bash
uv run python scripts/xqrobotwl/backflip_feasibility.py \
  --W 0 --launch_lean 0.6 --tlaunch 0.25 --crouch_hip -0.25 \
  --render video/backflip/2026-08-04_01_开环可行_360°落地3.8°.mp4
```

`--render` 传一个**目录**或**mp4 文件路径**均可：
- 传目录 → 目录内生成 `backflip.mp4`
- 传 `xxx.mp4` → 直接写入该文件

## 现有内容

- `backflip/2026-08-04_01_开环可行_360°落地3.8°_lean06_ch-025.mp4` — 开环脚本后空翻，恰好 360°，落地倾角 3.8°，腾空 0.41s（P1 物理可行性验证）
