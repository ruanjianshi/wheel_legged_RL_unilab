# [06] 新建 thesis/ 论文开发指导中心

**日期**: 2026-08-10
**来源**: 用户需要指导研究生大论文开发的文件夹,放框架图与开发文档
**关联**: [[05_verify_8_trained_policies]], [[01_strip_repo_to_two_robots]]

---

## 问题描述

用户需要构建一个文件夹用于**指导研究生大论文的开发**,届时放入框架图和开发文档。经确认: 目录名 `thesis/`(仓库根目录),预填充 = 骨架 + 框架图 + 文档模板。

## 解决方案

```
thesis/
├── README.md                      # 总览: 目录说明/快速导航/论文主线/使用约定
├── framework/                     # 框架图
│   ├── README.md                  # 索引 + 预览/导出/命名约定
│   ├── architecture.mmd           # 系统总体架构图 (7 层: 机器人→环境→算法→配置→训练→评估→文档)
│   └── jump_comparison.mmd        # 四算法跳跃 2×2 对比矩阵 (SRL × VMC)
└── docs/                          # 开发文档
    ├── README.md                  # 文档索引 (预置 5 篇计划文档)
    └── templates/
        └── template_dev_doc.md    # 开发文档模板
```

框架图源文件用 Mermaid `.mmd`(GitHub / mermaid.live 可预览),论文用图可导出 PNG/SVG。
与已有 `docs/PROJECT_ARCHITECTURE.md`(XqRobotV2 实现细节)互补,README 中已互相链接。

## 修改文件

- 新建: `thesis/` 下 6 个文件 (见上)

## 验证方法

- `find thesis -type f` 确认结构完整
- mermaid 语法人工核对 (graph/subgraph/node 均合法);本地无 mmdc,未本地渲染

## 后续计划

- [ ] 用户自行提交 (遵循"我自己提交"约定)
- [ ] 框架图补充: 训练流程时序图 / 奖励设计示意图
- [ ] 开发文档按索引逐步填充
