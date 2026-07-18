# AGENTS.md - Reinforcement Learning Grid World

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Manual play (keyboard control)
python test.py

# Train agent (default 500 episodes)
python Brain/train.py
python Brain/train.py 1000        # Custom episode count

# Auto-play with trained model
python Brain/train.py play
python Brain/train.py play q_table_best.npy

# Evaluate model
python Brain/eval.py
python Brain/eval.py -m q_table.npy -e 200 --render

# GUI config editor
python GUI/config_gui.py
```

## Project Structure

- **Config/** - YAML configs (train_config.yaml, EnvConfig.yaml)
- **Models/** - Saved Q-tables (q_table.npy, q_table_best.npy)
- **Brain/** - Q-learning implementation (train.py, eval.py)
- **Environment/** - Game environment (GameEnv.py, State.py, Action.py, EnvUtils.py)
- **GUI/** - Parameter config GUI

## Key Facts

- Env config loads from `Config/EnvConfig.yaml` primarily, falls back to `Environment/EnvConfig.yaml`
- Training config: `Config/train_config.yaml`
- State representation: 6-channel tensor (agent, food, 4 obstacle directions)
- Training renders every 10 episodes, set `render=False` in code to disable
- Models directory auto-created on first save
- Evaluation exits early if no model found (must train first)
