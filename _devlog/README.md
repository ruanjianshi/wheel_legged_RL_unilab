# 开发日志 (DevLog)

闭环开发记录系统。每次改动、评估、修复都须记录，形成"问题→分析→方案→修改→评估→结论"的完整链路。

## 目录结构

```
_devlog/
├── README.md                    # 本文件（规范说明）
├── TEMPLATE.md                  # 日志模板
├── INDEX.md                     # 全局索引（机器人 → 有时间线）
│
├── <robot>/                     # 按机器人分类
│   ├── INDEX.md                 # 该机器人所有任务的索引入口
│   ├── <task>/                  # 按任务分类（walk_flat / walk_rough / stairs / jump …）
│   │   └── <algo>/              # 按算法分类（ppo / np3o / sac …）
│   │       ├── INDEX.md         # 该组合的开发记录索引
│   │       └── <YYYY-MM-DD>/    # 按日期分目录
│   │           └── <NN>_<slug>.md  # 单条日志
│   └── ...
│
└── assess/                      # 评估框架本身的开发日志
```

## 日志命名规范

```
_devlog/<robot>/<task>/<algo>/<YYYY-MM-DD>/<NN>_<英文 slug>.md
```

示例（以 xqrobotV2 为例）：
```
_devlog/xqrobotv2/flat_walk/ppo/2026-07-01/01_fix_hip_and_assess.md
_devlog/xqrobotv2/rough_walk/ppo/2026-07-04/04_fix_forward_tracking.md
_devlog/xqrobotv2/stairs/np3o/2026-07-09/01_np3o_stairs_init.md
_devlog/xqrobotv2/jump/ppo/2026-07-11/01_fix_sensor_jump_key.md
_devlog/xqrobotwl/...
```

`NN` 为当天顺序，从 01 开始。

## 日志内容要求

每篇日志**必须**包含（缺失视为不合格）：

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

## AI 开发纪律

**所有通过 AI (opencode) 进行的开发活动都必须自记录：**

1. 每次修改代码 → 写日志
2. 每次调整超参 → 写日志
3. 每次评估出新结论 → 写日志
4. 每次修 bug → 写日志
5. 每次架构变更 → 写日志

完成指令后 AI 应**主动**判断是否需要记录，并提示用户确认。涉及代码/配置变更的操作 **必须** 写入日志。

## 日志索引

- 全局索引：[INDEX.md](INDEX.md) — 按机器人 + 时间线
- XqRobotV2：[xqrobotv2/INDEX.md](xqrobotv2/INDEX.md) — 按任务
