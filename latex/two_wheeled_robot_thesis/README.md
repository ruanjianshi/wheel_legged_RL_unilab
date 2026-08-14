# 两轮足机器人多步态运动控制与自适应调度毕业论文（LaTeX）

> 论文题目：《基于强化学习的两轮足机器人多步态运动控制与自适应调度方法研究》（28字）
> 基于武汉科技大学 WUSTThesis LaTeX 模板（`WUSTthesis.cls` V1.0，依据武科大研发〔2023〕3号《关于学位论文撰写格式统一要求的规定》）。

## 📁 目录结构

```
two_wheeled_robot_thesis/
├── main.tex                 # 主文件（已适配硕士版式）
├── WUSTthesis.cls           # 模板类文件
├── WUSTThesis.bst           # 参考文献样式
├── WUSTtils.sty             # 工具宏包
├── font/                    # 中文字体（SimSun/SimHei/楷体/仿宋/中宋）
├── figures/                 # 图片目录（论文组织结构图、架构图等）
├── ref/refs.bib             # 参考文献库
└── body/                    # 正文各章节
    ├── cover.tex            # 封面、中英文摘要与关键词
    ├── denotation.tex       # 符号说明
    ├── chap01.tex           # 绪论
    ├── chap02.tex           # 建模与强化学习理论基础
    ├── chap03.tex           # 多任务总体框架（分训-整合）与混合地形统一环境
    ├── chap04.tex           # 行走类步态专家（滚动/粗糙地形）
    ├── chap05.tex           # 精细平衡类步态专家（抬腿/上台阶/单腿平衡）
    ├── chap06.tex           # 高动态敏捷类步态专家（跳跃/后空翻）
    ├── chap07.tex           # 学习式统一调度器设计
    ├── chap08.tex           # 统一系统集成与整体验证
    ├── conclusion.tex       # 结论与展望
    ├── ack.tex              # 致谢
    ├── publications.tex     # 攻读硕士期间学术成果
    └── project.tex          # 攻读硕士期间参与科研项目
```

## 🔨 编译方式

使用 XeLaTeX 编译（需 TeX Live / MacTeX，Windows 可用 TeX Live 或 Overleaf）。所有编译输出（.aux/.log/.toc/.bbl/.pdf 等）统一输出到 `out/` 目录，保持源码目录整洁。

```bash
# 一键编译（推荐）
./build.sh

# 或使用 latexmk（配置见 .latexmkrc）
latexmk -xelatex main.tex

# 清理编译中间文件（保留 out/main.pdf）
latexmk -c
```

- 输出 PDF 位于 `out/main.pdf`
- 草稿期在 `main.tex` 中改用 `\documentclass[draftformat,mathCMR]{WUSTthesis}`
- 送审前切换回 `finalformat` 并隐去封面姓名（见 `body/publications.tex` 中的盲审注释）
- 若在 Overleaf 使用：上传整个文件夹（含 `font/`）即可，编译输出由 Overleaf 自行管理

## ✍️ 需要你填写的部分

| 位置 | 内容 |
|------|------|
| `body/cover.tex` | 学号、姓名、导师姓名与职称、答辩日期、英文姓名 |
| `body/cover.tex` | 中英文摘要、关键词（按研究实际结果细化） |
| `body/ack.tex` | 致谢中的人名与单位 |
| `figures/` | 论文组织结构图、五层架构图、各步态示意图与实验曲线 |
| 各章 `【待填入】` 处 | 实验数据、对比结果、训练曲线（已预留表格结构） |
| `ref/refs.bib` | 继续补充近5年文献与中文文献（硕士需≥40篇，外文≥20篇） |
| `body/publications.tex` `body/project.tex` | 本人发表成果与参与项目 |

## 📐 章节与你的研究框架对应关系（"分训-整合"范式）

| 论文章节 | 对应研究内容 |
|----------|--------------|
| 第3章 总体框架 | "分训-整合"两阶段范式 + 混合地形统一环境 + 三大问题应对 |
| 第4章 行走类专家 | 任务一 `roll_walk_flat_ppo`（含侧向移动）、任务二 `roll_walk_rough_ppo` |
| 第5章 精细平衡类专家 | 任务三 `toe_walk_flat_ppo`、任务五 `leg_lift_stair_n3po`、任务七 `single_leg_balance_*` |
| 第6章 高动态敏捷类专家 | 任务四 `jump_*`（三算法对比）、任务六 `backflip_flat_ppo` |
| 第7章 学习式统一调度器 | 分层RL建模、地形感知、门控切换、滞回与能量指标融合（核心创新） |
| 第8章 统一系统集成验证 | 混合地形整体遍历 + 灾难性遗忘/奖励冲突/局部最优消融实验 |

## 📋 武汉科技大学硕士论文格式要点（速查）

- 论文题目 **≤30字**；中文摘要 **约500字**，关键词3-5个
- 正文 **≥2万字**；参考文献 **≥40篇（外文≥20篇，近5年≥1/3）**
- 页面：A4，上下左右边距各3.0cm，双面打印
- 正文：宋体/Times New Roman 小四号，1.25倍行距，首行缩进2字符
- 一级标题黑体小二号加粗，二级黑体小三号，三级黑体四号，四级宋体四号
- 图题在图下方、表题在表上方，宋体/Times New Roman 五号加粗居中
- 公式按章编号右对齐（如（5-1）），图/表按章编号（如图5.1、表5.1）
- 装订顺序：封面→英文扉页→独创性声明→中文摘要→英文摘要→目录→符号说明→绪论→正文→结论→致谢→参考文献→附录
