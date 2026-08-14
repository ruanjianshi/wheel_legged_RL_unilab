
注：论文的数据和图片放置/home/robot/xiaoq/wheel_legged_RL_unilab/picture/paper，按照八个任务分别片放在/home/robot/xiaoq/wheel_legged_RL_unilab/picture/paper/[任务]/[内容]/[版式]/[版本]
# 强化学习科研绘图规范 v2.1

> 基于 Nature Master Style Guide + Okabe-Ito 色盲安全调色板
> 适用于: 跳跃机器人、轮腿机器人等强化学习科研绘图

---

## 目录

1. [尺寸规范](#一尺寸规范)
2. [字体规范](#二字体规范)
3. [线条与坐标轴](#三线条与坐标轴)
4. [配色规范](#四配色规范)
5. [训练曲线标准画法](#五训练曲线标准画法)
6. [子图编号与图例](#六子图编号与图例)
7. [输出规范](#七输出规范)
8. [布局模板](#八布局模板)
9. [文件命名规范](#九文件命名规范)
10. [算法对比图专题](#十算法对比图专题)
11. [必出图清单](#十一必出图清单)
12. [出图工作流](#十二出图工作流)
13. [LaTeX 插入模板](#十三latex-插入模板)
14. [常见问题](#十四常见问题)

---

## 一、尺寸规范

| 项目 | mm | inch |
|------|----|------|
| 单栏图宽度 | 88.9 | 3.5 |
| 双栏跨栏宽度 | 184 | 7.25 |
| 推荐宽高比 | — | 1 : 0.618 (黄金比) |
| 最大高度 | 230 | 9.06 |

```python
COL_WIDTH = 3.5
FULL_WIDTH = 7.25
GOLDEN_RATIO = (5**0.5 - 1) / 2
```

---

## 二、字体规范

**Nature 要求: 全部使用无衬线字体**

| 元素 | 字号 (pt) | 字体 |
|------|-----------|------|
| 轴标签 | 9 | Helvetica / Arial |
| 正文/刻度 | 7-8 | Helvetica / Arial |
| 子图编号 (a)/(b) | 9 | 粗体 |
| 数学公式 | — | STIX |
| 图例 | 7 | Helvetica / Arial |

```python
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans"],
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "mathtext.fontset": "stix",
})
```

---

## 三、线条与坐标轴

| 元素 | 线宽 (pt) | 样式 |
|------|-----------|------|
| 主曲线 (EMA) | 1.4 | 实线 |
| 原始数据 | 0.4 | 同色，alpha=0.22 |
| 坐标轴 spine | 0.7 | 仅左下可见，向内偏移 5pt |
| 刻度线 (主) | 0.7 | 长度 3.5 pt，向内 |
| 网格线 | 0.3 | **仅 y 轴**，淡灰 `#CCCCCC`，alpha=0.25 |

```python
mpl.rcParams.update({
    "lines.linewidth": 1.4,
    "axes.linewidth": 0.7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.linewidth": 0.3,
    "grid.alpha": 0.25,
    "grid.color": "#CCCCCC",
})

# 隐藏右上 + 向内偏移
for ax in axes.flat:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_position(("outward", 5))
    ax.spines["bottom"].set_position(("outward", 5))
```

---

## 四、配色规范

### Okabe-Ito 色盲安全调色板 (Nature 官方推荐)

**v2.1 选定方案 — 最经典、最安全、审稿人最熟悉**

| 顺序 | 算法 | HEX | 颜色 | 用途 |
|------|------|-----|------|------|
| 1 | PPO | `#0072B2` | 🔵 深蓝 | 基准算法 |
| 2 | CPO | `#D55E00` | 🟠 橙 | 方法 1 |
| 3 | NP3O | `#009E73` | 🟢 绿 | 方法 2 |
| 4 | SAC | `#CC79A7` | 🟣 紫 | 方法 3 |
| 5 | Baseline | `#56B4E9` | 🔵 浅蓝 | 基线 |
| 6 | Ablation | `#F0E442` | 🟡 黄 | 消融项 |

```python
ALGO_COLORS = {
    "PPO": "#0072B2",      # 蓝
    "CPO": "#D55E00",      # 橙
    "NP3O": "#009E73",     # 绿
    "SAC": "#CC79A7",      # 紫
    "Baseline": "#56B4E9", # 浅蓝
}

# 统一实线，仅颜色区分 (v2.1)
ALGO_LINESTYLES = {k: "-" for k in ALGO_COLORS}
```

### 为什么选这套？
- ✅ **色盲安全** — 红绿色盲、蓝黄色盲都能区分
- ✅ **灰度可分** — 黑白打印明度有明显差异
- ✅ **顶会通用** — Nature/Science/Cell 标配
- ✅ **不刺眼** — 饱和度适中，长时间看不累

---

## 五、训练曲线标准画法

### v2.1 科研标准: 淡原始 + EMA 平滑主曲线

**为什么要保留原始线？**
- 科研诚实：展示数据的真实波动
- 审稿人能直接看到噪声水平
- EMA 只是辅助看趋势，原始数据才是真实结果

**参数规范**

| 元素 | 线宽 | alpha | zorder |
|------|------|-------|--------|
| 原始数据 | 0.4 pt | 0.22 | 1 (底层) |
| EMA 主曲线 | 1.4 pt | 1.0 | 3 (顶层) |
| EMA α | 0.02 | — | — |
| 降采样 | ≤ 1200 点 | — | — |

```python
def ema(values, alpha=0.02):
    out = np.empty_like(values, dtype=np.float64)
    acc = float(values[0])
    for i, v in enumerate(values):
        acc = (1-alpha)*acc + alpha*float(v)
        out[i] = acc
    return out

def plot_curve(ax, steps, values, label, algo, alpha_smooth=0.02,
               show_raw=True, raw_alpha=0.22, raw_lw=0.4):
    color = ALGO_COLORS[algo]
    s, v = downsample(steps, values)
    
    # 淡原始线
    if show_raw:
        ax.plot(s, v, color=color, linewidth=raw_lw,
                alpha=raw_alpha, zorder=1)
    
    # EMA 主曲线
    s2, v2 = downsample(steps, ema(values, alpha_smooth))
    ax.plot(s2, v2, label=label, color=color,
            linewidth=1.4, zorder=3)
```

### 多 seed 时的画法

**均值曲线 + std 填充带** (顶会标准不确定性表达)

| 元素 | 参数 |
|------|------|
| 均值线宽 | 1.4 pt |
| std 填充带 alpha | 0.15 |
| 填充带描边 | 无 (linewidth=0) |

```python
def plot_band(ax, steps, runs, label, algo, alpha_smooth=0.02):
    color = ALGO_COLORS[algo]
    smoothed = [ema(r, alpha_smooth) for r in runs]
    stack = np.vstack(smoothed)
    mean, std = stack.mean(axis=0), stack.std(axis=0)
    
    ax.fill_between(steps, mean-std, mean+std,
                    color=color, alpha=0.15, linewidth=0, zorder=1)
    ax.plot(steps, mean, label=label, color=color,
            linewidth=1.4, zorder=3)
```

---

## 六、子图编号与图例

**子图编号**
```python
def panel_label(ax, text, dx=0.0, dy=1.02):
    ax.text(dx, dy, text, transform=ax.transAxes,
            fontsize=9, fontweight="bold")
```
- 位置: 左上角 (0, 1.02)，axes 坐标系
- 格式: `(a)`, `(b)`

**图例**
- 无边框
- 多子图: 整图下方居中，ncol=算法数
- 单子图: 图内 (upper left / lower right / best)

```python
def shared_legend(fig, ax, ncol=3, y=-0.12):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, y), frameon=False)
```

---

## 七、输出规范

| 格式 | DPI | 用途 |
|------|-----|------|
| PDF | 矢量 | 论文投稿 |
| PNG | 600 | 预览 / 网页 |

```python
def save(fig, out_dir, name):
    fig.savefig(f"{out_dir}/{name}.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(f"{out_dir}/{name}.png", format="png",
                dpi=600, bbox_inches="tight")
    plt.close(fig)
```

> 必须: `pdf.fonttype = 42`

---

## 八、布局模板

### 2×1 单栏主图
```python
fig, axes = make_axes(1, 2)
fig.tight_layout(pad=0.6, rect=(0, 0.18, 1, 1))
```

### 2×2 双栏全景图
```python
fig, axes = make_axes(2, 2)
fig.tight_layout(pad=0.6, rect=(0, 0.1, 1, 1))
```

### 1×1 单指标图
```python
fig, axes = make_axes(1, 1)
ax.legend(loc="best")
```

---

## 九、文件命名规范

### 核心格式
```
fig_[步态]_[任务]_[内容]_[版式]_[版本].pdf/png
```

### 字段说明

| 字段 | 说明 | 常用值 |
|------|------|--------|
| `[步态]` | 运动模式 | `jump`, `walk`, `stand`, `fall`, `stair` |
| `[任务]` | 场景/地形 | `flat`, `rough`, `slope`, `stair`, `push` |
| `[内容]` | 图表内容 | 见缩写表 |
| `[版式]` | 子图布局 | `1x1`, `2x1`, `2x2`, `2x3` |
| `[版本]` | 可选 | `v2`, `nature`, `appendix` |

### 内容缩写

| 缩写 | 含义 |
|------|------|
| `training` | 训练全景 |
| `surv_reward` | 存活+奖励 |
| `reward` | 累计奖励 |
| `ep_len` | 存活步长 |
| `traj` | 轨迹 |
| `joints` | 关节角 |
| `final_perf` | 最终性能柱状图 |
| `ablation` | 消融实验 |

### 示例
```
fig_jump_flat_training_2x2_v2.pdf
fig_jump_flat_surv_reward_2x1_v2.pdf
fig_jump_flat_final_perf_1x1.pdf
```

> 原则: 从左到右，范围从大到小 — 步态→任务→内容→版式→版本

---

## 十、算法对比图专题

### 10.1 训练过程对比 (线图)

| 要素 | 规范 |
|------|------|
| 图类型 | 折线图 (EMA + 淡原始) |
| 多 seed | 均值曲线 + std 填充带 (alpha=0.15) |
| x 轴 | Training iteration / Environment steps (k 单位) |
| 配色 | Okabe-Ito 固定颜色 |
| 顺序 | 按最终性能从高到低 (图例顺序一致) |

**注意**:
- ✅ x 轴范围对齐 (取最小 max iter)
- ✅ 至少 3 个 seed，推荐 5 个
- ✅ caption 注明 "EMA α=0.02"
- ❌ 不要只画最好的一条

---

### 10.2 最终性能对比 (柱状图)

| 要素 | 规范 |
|------|------|
| 图类型 | 柱状图 + 误差棒 |
| 误差棒 | std 或 95% CI，capsize=4 |
| 统计标记 | * p<0.05, ** p<0.01, *** p<0.001 |
| 柱子宽度 | 0.6-0.7 |
| 配色 | 同训练曲线 (一致性) |

```python
algos = ["PPO", "CPO", "NP3O"]
means = [950, 920, 880]
stds = [30, 45, 50]
x = np.arange(len(algos))

ax.bar(x, means, 0.6, color=[ALGO_COLORS[a] for a in algos],
       yerr=stds, capsize=4, error_kw={"elinewidth": 0.8})
ax.set_xticks(x)
ax.set_xticklabels(algos)
```

---

### 10.3 多条件分组柱状图

不同速度 / 不同地形 / 不同扰动下的性能对比:

```python
conditions = ["0.5 m/s", "1.0 m/s", "1.5 m/s"]
algos = ["PPO", "CPO", "NP3O"]
data = np.random.rand(3, 3)  # [condition, algo]

x = np.arange(len(conditions))
width = 0.25
for i, algo in enumerate(algos):
    ax.bar(x + (i-1)*width, data[:, i], width,
           color=ALGO_COLORS[algo], label=algo)
ax.set_xticks(x)
ax.set_xticklabels(conditions)
```

---

### 10.4 消融实验 (横向柱状图)

```python
methods = ["Full", "w/o reward A", "w/o module B"]
values = [100, 75, 60]
colors = ["#0072B2", "#D55E00", "#D55E00"]

y = np.arange(len(methods))
ax.barh(y, values, color=colors, height=0.6)
ax.set_yticks(y)
ax.set_yticklabels(methods)
ax.axvline(values[0], color="#0072B2", linestyle="--", alpha=0.5)
```

---

### 10.5 审稿人常见质疑

| 问题 | 解决 |
|------|------|
| "只跑了一个 seed?" | 至少 3-5 seed，画 std 带 |
| "为什么不用统计检验?" | 柱状图加显著性标记 |
| "x 轴对不齐?" | 统一用环境步数，不是 iters |
| "平滑参数是多少?" | caption 写 EMA α 值 |

---

## 十一、必出图清单

### 第一优先级 — 主图必备 (4-5 张)

| # | 图名 | 版式 | 文件名 | 说明 |
|---|------|------|--------|------|
| 1 | 训练曲线全景 | 2×2 | `fig_jump_flat_training_2x2_v2.pdf` | 奖励+存活+跳跃高度+策略std |
| 2 | 存活+奖励 | 2×1 | `fig_jump_flat_surv_reward_2x1_v2.pdf` | 核心训练对比 |
| 3 | 最终性能柱状图 | 2×1 | `fig_jump_flat_final_perf_2x1.pdf` | 奖励+跳跃高度均值±std |
| 4 | 典型跳跃轨迹 | 2×1 | `fig_jump_flat_traj_2x1.pdf` | 高度+速度，标腾空段 |
| 5 | 消融实验 | 1×1 | `fig_jump_flat_ablation_1x1.pdf` | 各组件贡献 |

### 第二优先级 — 补充图 (2-3 张)

| # | 图名 | 版式 | 说明 |
|---|------|------|------|
| 6 | 关节角曲线 | 2×3 | 6 个腿关节 |
| 7 | 不同速度性能 | 1×1 | 速度 vs 跟踪误差 |
| 8 | 姿态稳定性 | 2×1 | roll/pitch |

### 第三优先级 — 附录

| # | 图名 | 说明 |
|---|------|------|
| 9 | Loss 曲线 | Value/Policy loss |
| 10 | Seed 鲁棒性箱线图 | 5-10 seed 分布 |
| 11 | 样本效率对比 | 达到同性能所需步数 |
| 12 | 超参数敏感性 | 学习率/熵系数影响 |

---

## 十二、出图工作流

```
1. 跑训练 → 出训练曲线 (确认收敛)
   ↓
2. 训完 → 出最终性能柱状图 + 统计检验
   ↓
3. 导出 rollout → 出轨迹/关节曲线
   ↓
4. 多地形/多速度 → 出泛化图
   ↓
5. 消融实验 → 出 ablation 图
```

> 💡 主图讲故事逻辑: 「收敛快 → 性能好 → 轨迹美 → 组件有用」

---

## 十三、LaTeX 插入模板

```latex
% 单栏图
\begin{figure}[t]
    \centering
    \includegraphics[width=\linewidth]{figures/fig_jump_flat_surv_reward_2x1_v2.pdf}
    \caption{Comparison of episode length and return.
             Light traces indicate raw data; dark lines indicate
             exponential moving average ($\alpha = 0.02$).
             Best viewed in color.}
    \label{fig:survival}
\end{figure}

% 双栏图
\begin{figure*}
    \centering
    \includegraphics[width=0.95\textwidth]{figures/fig_jump_flat_training_2x2_v2.pdf}
    \caption{Training overview. (a) Return. (b) Episode length.
             (c) Jump height reward. (d) Policy std.}
    \label{fig:training}
\end{figure*}

% 柱状图
\begin{figure}[t]
    \centering
    \includegraphics[width=0.8\linewidth]{figures/fig_jump_flat_final_perf_1x1.pdf}
    \caption{Final performance (mean $\pm$ std over 5 seeds).
             $^*p<0.05$, $^{**}p<0.01$, two-tailed t-test.}
    \label{fig:final_perf}
\end{figure}
```

---

## 十四、常见问题

| 问题 | 解决 |
|------|------|
| PDF 字体嵌入失败 | `pdf.fonttype = 42` |
| 图例被截断 | `bbox_inches="tight"` |
| 曲线太密集 | 降采样到 1000-1200 点 |
| 原始线太淡/太浓 | 调 `raw_alpha` (0.15-0.3) |
| 主曲线不够突出 | 调 `linewidth` (1.2-1.6) |
| 网格太显眼 | 调 `grid.alpha` (0.15-0.25) |
| 中英文混排 | 加 Noto Sans CJK SC |

---

---

**版本**: v2.1 (Okabe-Ito 配色 + 科研标准曲线画法)
**最后更新**: 2026-08-14
**适用**: 轮腿机器人 / 跳跃机器人强化学习科研绘图
