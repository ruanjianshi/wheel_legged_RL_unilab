# 36 论文图 v2.0 重出 — 严格遵循 nature-draw.md 规范 (Wheeled-SRL 小论文)

## 日期
2026-08-14

## 来源
用户指定: 按 `nature-draw.md`(用户手写的 Nature 期刊级绘图规范 v2.0)为小论文
`latex/Wheeled-SRL-Jumping` 重出全部数据图。旧图(v1)由 `make_paper_figures.py` +
`make_framework_figure.py` + `plot_jump_trajectory.py` 生成, 用的是 **Okabe-Ito 配色 +
scienceplots 样式**, 与用户新规范不符。

## 修改了什么
新建统一出图脚本 `tools/make_paper_figures_v2.py`, 严格按 nature-draw.md v2.0 规范
生成 5 张图 (PDF 矢量 + PNG 600dpi), 并更新 `main.tex` 引用为 PDF。

### 规范落实情况 (nature-draw.md)
| 规范条目 | 落实 |
|---|---|
| 尺寸 | 按本论文实际 `\columnwidth≈6.3in` 渲染 (A4 去 2.5cm 边距), 字体字号精确 |
| 字体 | 无衬线 Liberation Sans (Helvetica/Arial 回退); 轴标签 9pt / 刻度 7pt / 图例 7pt; `mathtext=stixsans`; `pdf.fonttype=42` |
| 线条 | 数据 1.5pt; 坐标轴 0.7pt 仅左下可见+外移 5pt; 刻度向内; 网格仅 y 轴 `#ADB5BD` α=0.2 |
| 配色 | **IBM 调色板**: PPO `#4263EB` / SRL `#40C057` / PPO+VMC `#FA5252` / SRL+VMC `#7950F2` |
| 训练曲线 | EMA α=0.02 平滑主曲线 (无原始淡线); 降采样 ≤1200 点; x 轴 ×10³ 单位 |
| 子图编号 | (a)/(b)… 左上角粗体 9pt |
| 图例 | 无边框, 图下居中, ncol=4, 按最终性能降序 (SRL→SRL+VMC→PPO→PPO+VMC) |
| 命名 | `fig_[步态]_[任务]_[内容]_[版式]_[版本]` |
| 输出 | PDF 矢量 + PNG 600dpi (framework 300dpi 因含大量中文, 投稿用 PDF) |

### 生成的图 (latex/Wheeled-SRL-Jumping/figures/)
| 新文件 | 内容 | 对应论文图 |
|---|---|---|
| `fig_jump_flat_training_2x3_v2.pdf/png` | 训练指标 2×3: 平均奖励/回合长度/跳高奖励/软着陆奖励/动作std/FPS | 图 4.2 (fig:training_metrics) |
| `fig_jump_flat_final_perf_3x1_v2.pdf/png` | 最终性能 3 栏柱状: 跳高/腾空率/存活率 | 图 4.1 (fig:validation) |
| `fig_jump_flat_traj_2x2_v2.pdf/png` | 跳跃高度轨迹 2×2 + FSM 相位色带 | 图 4.4 (fig:trajectory) |
| `fig_jump_flat_joints_2x2_v2.pdf/png` | 腿部关节角 2×2 (膝先屈后伸时序) | 图 4.3 (fig:joints) |
| `framework.pdf/png` | 2×2 对照设计 + 控制流水线 (IBM 配色 + SimHei 中文) | 图 3.1 (fig:framework) |

### main.tex 更新
- 4 处数据图引用: `paper_fig_*.png` → `fig_jump_flat_*.pdf` (矢量图)
- framework: `framework.png` → `framework.pdf`
- fig:validation 宽度 `0.62\columnwidth` → `\columnwidth` (3 栏柱状需更宽), caption 改为
  "(a)跳高 (b)腾空率 (c)存活率"
- xqrobotwl_render.png 保留 (机器人渲染图, 非数据图)

## 哪些文件
- 新建: `tools/make_paper_figures_v2.py`
- 更新: `latex/Wheeled-SRL-Jumping/main.tex` (5 处 includegraphics + 1 处 caption)
- 生成: `latex/Wheeled-SRL-Jumping/figures/fig_jump_flat_*.pdf/png` (8 个) + `framework.pdf/png`
- 备份: 旧图 14 个 → `backup/paper_figs_v1_okabe_ito/`
- 清理: figures/ 下旧 paper_fig_*.png/pdf、eval_*.png、framework.drawio* 移除 (已备份)

## 训练后效果 (视觉验证)
- **training_metrics 2×3**: EMA 平滑后 SRL(绿)全程最高、纯PPO(蓝)后期发散, 与论文
  §4.3 文字完全一致; 图例按性能降序, 颜色唯一对应
- **final_perf 3×1**: 三指标柱状图, 数值标注精确到 3 位小数 (h=0.547/0.352/0.264/0.175,
  air=0.221/0.093/0.086/0.175, surv=1.00/0.80/0.35/0.95), 与表 4.1 一致
- **traj 2×2**: SRL/SRL+VMC 相位色带清晰 (crouch→thrust→flight→landing), 纯PPO 无
  FSM 相位全灰, 体现"相位分明 vs 含混"
- **joints 2×2**: SRL 膝角先负向屈膝再爆发伸展的时序结构可见
- **framework**: 中文字体 SimHei 正常嵌入 (pdffonts 验证, 非豆腐块), IBM 配色四格
- 论文编译: `latexmk -xelatex` exit=0, 无图形缺失警告, 6 个图均正常嵌入

## 参数调整好坏
- 配色: Okabe-Ito(旧) → **IBM 调色板**(新)。SRL 用绿色突出最优方法, PPO 蓝色作基准,
  红/紫区分两个 VMC 变体。IBM 色板明度差异大, 灰度打印也可区分 (符合规范 §四)
- EMA: 0.8 TensorBoard 风格(旧) → **α=0.02**(新)。α=0.02 更平滑、趋势更干净, 突出
  SRL 收敛与 PPO 发散对比
- 图例位置: 图内(旧) → 图下居中无边框(新), 避免遮挡曲线

## 根因分析
旧脚本不符合用户手写的 nature-draw.md v2.0 规范 (Okabe-Ito 配色 / scienceplots 样式 /
EMA 0.8 / 命名不一致), 需统一重出。

## 验证方法
1. 逐张 Read 检查 PNG 渲染 (数据图肉眼确认: 配色/平滑/编号/图例正确)
2. framework 中文: pdffonts 确认 SimHei 嵌入 (非豆腐块)
3. `latexmk -xelatex main.tex` 编译通过 exit=0, grep 图形警告 = 0
4. 数值与 four_algo_comparison.json / 表 tab:baseline 逐项核对一致

## 后续计划
- 旧图已备份 `backup/paper_figs_v1_okabe_ito/`, 如需回退可恢复
- 若审稿要求图例中英文、加 std 误差棒 (需多 seed), 脚本已预留 plot_band 能力
- 13 任务重训仍在后台 (task #115), 与本次出图无关

## 关联日志
- [[30_paper_figs_nature_style]] 上一版论文图规范 (Okabe-Ito)
- [[35_rewrite_small_paper_4algo]] 小论文重构 (2×2 对照)
