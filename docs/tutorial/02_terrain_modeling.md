# 02. 地形建模

## 目录

- [地形系统架构](#地形系统架构)
- [7 种地形类型详解](#7-种地形类型详解)
- [地形生成器配置](#地形生成器配置)
- [预设地形配置](#预设地形配置)
- [在环境中启用地形](#在环境中启用地形)
- [自定义地形](#自定义地形)

---

## 地形系统架构

**核心文件**: `src/unilab/terrains/`

```
terrains/
├── terrain_generator.py       # 主生成器: TerrainGeneratorCfg, GeneratedTerrain
├── heightfield_terrains.py    # 7 种子地形类型实现
├── config.py                  # 预设配置 (ROUGH_TERRAINS_CFG, STAIRS_TERRAINS_CFG)
└── utils.py                   # 双线性插值, 平地检测
```

### 数据流

```
TerrainGeneratorCfg (配置)
  │  size, num_rows, num_cols, sub_terrains{...}
  ▼
TerrainGenerator.generate()
  │  遍历 num_rows × num_cols 网格
  │  每个格子随机/课程选择一个 SubTerrainCfg
  │  调用 sub_terrain.function(difficulty, rng) → TerrainOutput
  │  将子地形拼接到全局高度场
  ▼
GeneratedTerrain
  ├── heights_yx: np.ndarray          # (rows, cols) 高度值
  ├── horizontal_scale: float         # 每像素的物理宽度 (m)
  ├── z_min / z_max: float            # 高度范围
  ├── base_thickness: float           # 底部厚度
  ├── terrain_origins: np.ndarray     # 每个格子的世界坐标原点
  │
  └── to_uint16() → PNG → MuJoCo hfield
```

---

## 7 种地形类型详解

**文件**: `src/unilab/terrains/heightfield_terrains.py`

### 1. HfFlatTerrainCfg — 平地

```python
@dataclass
class HfFlatTerrainCfg(SubTerrainCfg):
    horizontal_scale: float = 0.05
    vertical_scale: float = 0.005
    base_thickness_ratio: float = 0.0

    def function(self, difficulty, rng):
        # ↑ difficulty 参数被忽略
        noise = np.zeros((width_pixels, length_pixels))
        origin = np.array([self.size[0] / 2, self.size[1] / 2, 0.0])
        return TerrainOutput(origin=origin, noise=noise)
```

**参数**: `size` — 从父类 SubTerrainCfg 继承，控制格子尺寸。

### 2. & 3. HfPyramidStairsTerrainCfg / HfInvertedPyramidStairsTerrainCfg — 台阶

```python
@dataclass
class HfPyramidStairsTerrainCfg(SubTerrainCfg):
    step_height_range: tuple[float, float]   # 台阶高度范围 [0.02, 0.10]
    step_width: float                        # 每级台阶宽度 (m)
    platform_width: float = 1.0              # 顶端平台宽度
    border_width: float = 0.0                # 边界宽度
    holes: bool = False                      # 是否在角落挖坑
    pit_depth: float = 5.0
    horizontal_scale: float = 0.05
    vertical_scale: float = 0.005

    def function(self, difficulty, rng):
        # difficulty ∈ [0, 1] 控制台阶高度
        step_height = step_height_range[0] + difficulty * (range[1] - range[0])
        step_units = int(round(step_height / self.vertical_scale))
        # 生成同心方环, 每环升高 step_units
        for k in range(n_steps):
            noise[lo_x:hi_x, lo_y:hi_y] = (k + 1) * step_units
        # 如果 holes=True: 在四个对角挖深坑
```

**关键参数**: `difficulty ∈ [0, 1]` 线性映射到 `step_height_range`, 台阶逐级升高。

`HfInvertedPyramidStairsTerrainCfg` 继承自正向版，`step_units` 取负值，从边缘往中心逐级 **下降**，形成坑状结构。

### 4. & 5. HfPyramidSlopedTerrainCfg — 斜坡

```python
@dataclass
class HfPyramidSlopedTerrainCfg(SubTerrainCfg):
    slope_range: tuple[float, float]     # 坡度范围 (弧度)
    platform_width: float = 1.0          # 中间平台宽度
    inverted: bool = False               # False=山峰, True=盆地
    border_width: float = 0.0
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005

    def function(self, difficulty, rng):
        slope = slope_range[0] + difficulty * (range[1] - range[0])
        # 生成从边缘到中心按斜率渐变的高度场
        # inverted=False → 中心最高 (金字塔)
        # inverted=True  → 中心最低 (碗形)
```

### 6. HfRandomUniformTerrainCfg — 随机粗糙

```python
@dataclass
class HfRandomUniformTerrainCfg(SubTerrainCfg):
    noise_range: tuple[float, float]     # 粗糙度范围 [0.02, 0.10]
    noise_step: float = 0.005            # 下采样粒度
    downsampled_scale: float | None = None
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    border_width: float = 0.0

    def function(self, difficulty, rng):
        # 在低分辨率网格上随机采样高度
        # 然后双线性上采样到全分辨率
        # difficulty 控制噪声幅度
```

### 7. HfWaveTerrainCfg — 波浪地形

```python
@dataclass
class HfWaveTerrainCfg(SubTerrainCfg):
    amplitude_range: tuple[float, float]   # 振幅范围
    num_waves: int = 1                     # 波浪数量
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    border_width: float = 0.0

    def function(self, difficulty, rng):
        amplitude = amplitude_range[0] + difficulty * (range[1] - range[0])
        # 正弦+余弦波叠加:
        # hf_raw = amplitude * (cos(yy * wave_number) + sin(xx * wave_number))
```

---

## 地形生成器配置

**文件**: `src/unilab/terrains/terrain_generator.py`

### TerrainGeneratorCfg

```python
@dataclass
class TerrainGeneratorCfg:
    seed: int | None = None
    curriculum: bool = False                # True=按行难度排序, False=随机采样

    # 格子尺寸
    size: tuple[float, float]               # 每个子地形的物理尺寸 (m)
    num_rows: int = 1                       # 行数
    num_cols: int = 1                       # 列数 (curriculum 模式下为难度级别数)

    # 分辨率
    horizontal_scale: float = 0.05          # 每像素 = 5cm (默认)
    vertical_scale: float = 0.005           # 高度量化步长 = 5mm (默认)

    # 边界
    border_width: float = 0.0               # 四周额外边界宽度

    # 子地形
    sub_terrains: dict[str, SubTerrainCfg]  # {"type_name": config, ...}

    difficulty_range: tuple[float, float] = (0.0, 1.0)  # 难度映射范围
    add_lights: bool = False                # 是否添加灯光
```

### 两种生成模式

**随机模式** (`curriculum=False`):
```python
# 每个格子独立按比例随机选择子地形
proportions = [cfg.proportion for cfg in sub_terrains]
for each cell:
    sub_type = np_rng.choice(sub_terrains, p=proportions/normalized)
    difficulty = np_rng.uniform(*difficulty_range)
```

**课程模式** (`curriculum=True`):
```python
# 一列对应一个子地形类型，行数对应难度级别
for sub_col in range(num_cols):         # 遍历地形类型
    for sub_row in range(num_rows):     # 遍历难度级别
        difficulty = (sub_row + noise) / num_rows  # 0→1 线性递增
```

---

## 预设地形配置

**文件**: `src/unilab/terrains/config.py`

### Rough 配置（随机模式）

```python
ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),          # 每格 8m × 8m
    border_width=20.0,        # 20m 外边距 (给机器人留跑出空间)
    num_rows=10,
    num_cols=20,              # 10×20 = 200 个格子
    horizontal_scale=0.1,     # 10cm 分辨率
    sub_terrains={
        "flat":              flat(proportion=0.20),
        "pyramid_stairs":    pyramid_stairs(proportion=0.15, step_height_range=(0.0, 0.1)),
        "pyramid_stairs_inv": pyramid_stairs_inv(proportion=0.15, step_height_range=(0.0, 0.1)),
        "hf_pyramid_slope":  hf_pyramid_slope(proportion=0.05, slope_range=(0.0, 0.15)),
        "hf_pyramid_slope_inv": hf_pyramid_slope_inv(proportion=0.05, slope_range=(0.0, 0.15)),
        "random_rough":      random_rough(proportion=0.30, noise_range=(0.0, 0.05)),
        "wave_terrain":      wave_terrain(proportion=0.10, amplitude_range=(0.0, 0.05)),
    },
)
```

### Stairs 配置（课程模式）

```python
STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=20.0,
    num_rows=10, num_cols=4,
    curriculum=True,          # ★ 课程模式
    sub_terrains={
        "flat":              flat(proportion=0.25),
        "easy_stairs":       pyramid_stairs(step_height_range=(0.02, 0.05), step_width=0.40),
        "moderate_stairs":   pyramid_stairs(step_height_range=(0.05, 0.08), step_width=0.35),
        "challenging_stairs": pyramid_stairs(step_height_range=(0.08, 0.10), step_width=0.30),
    },
)
```

### 地形预设装饰器

```python
@terrain_preset  # 注册为预设定制器
def flat(**overrides):
    return HfFlatTerrainCfg(**overrides)

# 使用: flat(proportion=0.2)  → 覆盖 proportion
#       flat(size=(4, 4))     → 覆盖 size
```

---

## 在环境中启用地形

### 粗糙地形环境配置

**文件**: `src/unilab/envs/locomotion/xqrobotV2/rough.py`

```python
@envcfg("XqRobotV2WalkRough")
@dataclass
class XqRobotV2WalkRoughCfg(XqRobotV2WalkFlatCfg):
    scene: SceneCfg = field(default_factory=lambda: SceneCfg(
        model_file=str(ASSETS_ROOT_PATH / "robots" / "xqrobotV2" / "xqrobotV2.xml"),
        fragment_files=[
            str(ASSETS_ROOT_PATH / "robots" / "xqrobotV2" / "locomotion_task.xml"),
        ],
        terrain=TerrainSceneCfg(
            generator=XqRobotRoughTerrainCfg(),  # TerrainGeneratorCfg 实例
            hfield_name="terrain_hfield",         # MuJoCo heightfield 名
            geom_name="floor",                    # 生成的碰撞几何体名
        ),
    ))
```

### 关键关系

```
SceneCfg.terrain.generator ≠ None
  ↓
MuJoCo 后端自动:
  1. 调用 TerrainGenerator.generate() → GeneratedTerrain
  2. 将 heights_yx 写入 PNG 纹理
  3. 生成 MJCF 代码:
     <hfield name="terrain_hfield" file="terrain.png" size="..."/>
     <geom name="floor" type="hfield" hfield="terrain_hfield"/>
  4. 将生成的 MJCF 合并到 robot.xml
```

**重要**: 如果 `SceneCfg.terrain` 为 `None`，后端降级为纯平坦地面（平面 `geom type="plane"`）。

---

## 自定义地形

### 最小示例

```python
from src.unilab.terrains.heightfield_terrains import HfRandomUniformTerrainCfg
from src.unilab.terrains.terrain_generator import TerrainGeneratorCfg

my_terrain = TerrainGeneratorCfg(
    size=(4.0, 4.0),          # 每个子地形 4m × 4m
    border_width=5.0,         # 5m 边界
    num_rows=4,               # 4 行
    num_cols=4,               # 4 列 = 16 个格子
    horizontal_scale=0.05,   # 5cm 分辨率
    sub_terrains={
        "flat":       flat(proportion=0.3),
        "rough":      random_rough(proportion=0.7, noise_range=(0.0, 0.08)),
    },
    # curriculum=False 默认 → 随机模式
)
```

### 在评估中用小地形

评估时只需要 1 个环境，不需要 200 个格子的地形。可以通过覆盖配置来缩小：

```python
# assess/runner.py 中的 rough env builder
cfg.scene.terrain.generator.num_rows = 4     # 减少行数
cfg.scene.terrain.generator.num_cols = 4     # 减少列数
```

---

## 关键要点

1. **`border_width` 必须足够大**：机器人初始位置在格子中心，外部边界给机器人留出移动空间
2. **`horizontal_scale` 决定分辨率**：值越小分辨率越高，但 PNG 越大、编译越慢。粗糙地形用 0.1 (10cm)
3. **`curriculum` 模式**：行=难度，列=地形类型。机器人按行逐步挑战更难的地形
4. **评估用最小网格**：单环境评估时用 `num_rows=4, num_cols=4` 减少生成开销

---

> 上一章：[01. MuJoCo 基础](./01_mujoco_basics.md)
> 下一章：[03. 机器人建模](./03_robot_modeling.md)
