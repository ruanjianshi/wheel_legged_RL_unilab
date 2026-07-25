# 14 — 平台自动检测

## 日期
2026-07-22

## 来源
需要训练/验证脚本在 macOS 和 Linux 上都能运行。

## 修改文件

`shell/xqrobotwl/*.sh` (12 个脚本):

```bash
if [[ "$(uname)" == "Darwin" ]]; then
    PYTHON="uv run mjpython"   # macOS: MuJoCo Python wrapper
else
    PYTHON="uv run"            # Linux
fi
```

所有 `uv run` 替换为 `$PYTHON`。

## 关联日志
- `2026-07-22/10` — 高度目标修正（同步修改）
