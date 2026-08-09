# `unilab.envs` — Tasks

Concrete RL tasks split by family:

- **locomotion** — XqRobotWL, XqRobotV2 wheel-legged bipeds

Every env inherits `NpEnv` and is registered into the task `Registry` so it
can be selected via `uv run train --algo <algo> --task <name> --sim <backend>`.

```{toctree}
:maxdepth: 2

locomotion
common
```

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :template: autosummary/module.rst
   :recursive:

   unilab.envs
```
