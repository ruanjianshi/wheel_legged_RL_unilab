"""八任务评估体系 (按 CLAUDE.md §1.2-1.5 / §7.0 / §7.x / 附录A).

组件:
  tasks.py     8 任务注册表 + §7.x 达标阈值
  engine.py    通用确定性 rollout 引擎 (建env / 加载policy / 跑episodes / 逐行采集)
  metrics.py   附录 A 核心指标 + 追踪/稳定/运动质量指标
  scenarios.py 场景模型 + 行走套件
  verify.py    §7.x 达标判定 (✅/❌)
  report.py    输出 (stdout + JSON + Markdown 报告)
  pose.py      姿态数据 CSV 导出 (§1.5.1)
  infer.py     姿态反推统计 (§1.3/§1.5.2)
  runner.py    统一 CLI 入口
  eval/        8 任务评估模块 (各实现 §7.x)

用法见 runner.py 或 README.md。
"""

from __future__ import annotations

from _devlog.assess.tasks import list_tasks

__all__ = ["list_tasks"]
