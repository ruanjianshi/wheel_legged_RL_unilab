# [33] 根目录清理: 删 AGENTS.md 只留 CLAUDE.md + 散落文件

**日期**: 2026-08-10
**来源**: 用户要求 "这个文件夹内散落的文件,进行清理没用的agent.md只保留claude.md"
**关联**: [[32_shell_reorganize_by_task]]

---

## 问题描述

仓库根目录 `CLAUDE.md` 是指向 `AGENTS.md` 的符号链接 (内容相同, 重复)。
用户希望只保留 `CLAUDE.md` 一个 agent 规范文件。另有一些散落文件:
- `MUJOCO_LOG.TXT` (mujoco 运行时日志, gitignore)
- `train_rsl_rl.log`, `train_np3o.log` (0 字节空日志, gitignore)
- `picture/AGENTS.md` (无关的 "RL Grid World" 项目文件, 误放本仓库)

## 解决方案

1. `CLAUDE.md` 从符号链接转为独立文件 (内容 = 原 AGENTS.md, 22,892 字节)
2. 删除顶层 `AGENTS.md` + `picture/AGENTS.md`
3. 删除散落 0 字节日志 + MUJOCO_LOG.TXT (本就被 gitignore)
4. 更新所有顶层 `AGENTS.md` 引用 → `CLAUDE.md`:
   - `.github/workflows/docs.yml` (移除失效触发路径)
   - `README.md` (2 处), `docs/tutorial/README.md`
   - `docs/sphinx/source/{en,zh_CN}/4-developer_guide/{6-agent_quick_reference,
     9-sim2sim_contract_status}.md`
5. 保留: `docs/sphinx/AGENTS.md` (独立的 sphinx 文档 agent 规范, 非顶层那个)

## 修改文件

- 删除: `AGENTS.md`, `picture/AGENTS.md`
- 类型变更: `CLAUDE.md` (symlink → 普通文件)
- 编辑: `.github/workflows/docs.yml`, `README.md`, `docs/tutorial/README.md`,
  `docs/sphinx/source/{en,zh_CN}/4-developer_guide/6-agent_quick_reference.md`,
  `docs/sphinx/source/zh_CN/4-developer_guide/9-sim2sim_contract_status.md`
- 删除 (gitignore, 无 git 变更): `MUJOCO_LOG.TXT`, `train_rsl_rl.log`, `train_np3o.log`

## 验证

- `git show HEAD:AGENTS.md | cmp - CLAUDE.md` → 内容完全一致
- `doc_checks.collect_doc_errors()` → 0
- 散落日志从未被 git 跟踪 (git ls-files 确认)

## 后续计划

- [x] CLAUDE.md 转独立文件
- [x] 删 AGENTS.md + picture/AGENTS.md
- [x] 清理散落日志
- [x] 更新所有引用
- [ ] 提交 (不推送)
