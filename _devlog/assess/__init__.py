"""XqRobotV2 model evaluation framework.

Standardized evaluation of trained .pt policy models against predefined
scenarios, producing JSON metrics for comparison across runs/checkpoints.

Directory:
    eval/
    ├── runner.py         # CLI entry point
    ├── metrics.py        # Metric computation
    ├── scenarios.py      # Test scenario definitions
    ├── configs/          # Scenario presets (YAML)
    ├── results/          # JSON output per evaluation
    └── reports/          # Comparison summaries
"""

__all__ = [
    "run_evaluation",
]
