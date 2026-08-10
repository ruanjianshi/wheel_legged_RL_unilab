"""八任务评估模块 (各实现 §7.x 达标标准).

每个模块导出:
  evaluate(env, policy, args) -> dict[str, float]   # 指标
可选:
  SUITES / DEFAULT_SUITE                            # 场景套件 (行走类)

任务间完全独立 (env/conf/shell/devlog/video), 可并行评估。
"""
