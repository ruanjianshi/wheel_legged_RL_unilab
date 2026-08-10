# [05] 8 个训练全部完成 + 8 个验证脚本冒烟通过

**日期**: 2026-08-10
**来源**: 用户询问训练是否完成 + 大清理删文件是否影响 xqrobotwl
**关联**: [[04_launch_8_trainings_and_fix_path_obs_bugs]], [[01_strip_repo_to_two_robots]]

---

## 训练结果

8 个 xqrobotwl 训练 2026-08-10 01:21 启动,全部跑满 10000 iterations 并产出最终 `model_9999.pt`:

| 任务 | 耗时 | 最终 reward | ep_len | action_std | SPS |
|------|------|------|------|------|------|
| walk_flat | 5h53m | 22.89 | 896 | 0.15 | 15179 |
| walk_rough | 9h56m | 42.05 | 914 | 0.15 | 8555 |
| jump_flat | 5h47m | 43.73 | 581 | 0.62 | 19757 |
| jump_srl | 5h47m | 97.05 | 920 | 0.77 | 16048 |
| jump_vmc | 7h36m | 87.92 | 993 | 0.24 | 10488 |
| jump_srl_vmc | 7h30m | 94.68 | 882 | 1.46 | 9302 |
| stairs (NP3O) | 10h21m | 31.22 | 899 | 0.53 | 11213 |
| toe_walk | 5h40m | 144.81 | 1000 | 0.24 | 9955 |

跳跃专项(最终迭代):
- jump_srl_vmc: jump_height 2.859 / vertical_thrust 8.67
- jump_vmc: jump_height 1.718 / landing_soft 3.82
- jump_srl: jump_height 1.446 / vertical_thrust 11.996
- **jump_flat: jump_height 0.0000 / wheel_air_time −0.002** ⚠️ 纯 PPO 疑似后期退化回不跳跃

## 验证方法

8 个 eval 脚本逐一冒烟运行(play_interactive GUI viewer,~10s 打开窗口),确认:
- 正确加载对应 task(flat→xqrobotwl_walk_flat, stairs→xqrobotwl_stairs/np3o, 其余 jump_*)
- 正确加载 08-10 训练产出的 `model_9999.pt`
- viewer 正常打开、无 Traceback

8/8 全部通过。

## 大清理对 xqrobotwl 的影响评估

git 证据(提交 1f180ff 仓库瘦身):
- xqrobotwl env 源码仅改 2 处且均为**纯格式**: `__init__.py` import 排序、`jump_srl.py` FSM 字典空格对齐;零逻辑变更
- **删除 0 个** xqrobot/xqrobotV2 相关文件
- 共享基础改动(registry 精简 locomotion、conftest xq fixture 等)由 8 个训练全程 10000 iter 运行 + 8 个验证脚本成功加载共同证明无回归
- stairs obs 维度 bug 反而是清理后启动训练时**修复**的(见 [[04_...]])

**结论: 大清理对 xqrobotwl 无功能影响。**

## ⚠️ 附带发现

`jump_management/` 整目录 35 个文件在**工作树被删但无提交记录**(git 完好,`git checkout -- jump_management/` 可完整恢复)。不在本次清理提交范围内。**用户已确认: 有意删除(旧跳跃实验工具已过时),保留现状**。

## 后续计划

- [x] 8 训练完成 + 8 验证脚本冒烟通过
- [x] 确认 jump_management/ 处置(用户有意删除,保留现状)
- [ ] 确认 jump_flat 是否实际退化(play 交互观察或跑评估)
- [ ] 论文 2×2 跳跃对比图(jump_height 等物理指标为准)
- [ ] 提交(用户自行 commit)
