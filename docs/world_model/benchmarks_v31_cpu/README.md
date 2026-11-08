# V31 本地 CPU 验证（2026-11-08）

环境：macOS / Python 3.12.15 / PyTorch 2.15.0 / pytest 9.2.0，CUDA 不可用。分支 `experiment/v31-actor-diagnostics`，基于 `735e702`。

| 检查 | 结果 |
|---|---|
| 新增专项测试 | 7 passed，69.22秒 |
| 完整 `pytest -q tests` | 173 passed，1 warning，548.51秒 |
| 新增 Python 模块/专项测试 Ruff | 通过 |
| 新增 Python 格式检查 | 8文件通过 |
| Shell `bash -n` / `--help` | 通过 |
| 独立 minimal CLI `--help`（不启动 Isaac） | 通过 |
| `git diff --check` | 通过 |

现有 `train.py` / `trainer.py` 的 Ruff 告警按代码类别与基线比较一致，无新增；没有为本任务重排 Isaac 启动顺序或清理历史代码。

命令：

```bash
PYTHONPATH=source/reflex:tests /tmp/reflex-wm-venv/bin/python -m pytest -q tests
```

原始标准输出见 [pytest_full.log](pytest_full.log) 和 [pytest_diagnostics.log](pytest_diagnostics.log)。未把小 CPU fixture 的 trace 当成 GPU 性能结果。第三方原生适配器当前只验证了接口 fixture，不是实际 TrackerLab checkout 的集成通过；FP32 CUDA Inductor 的数值与性能均待 GPU 实测。
