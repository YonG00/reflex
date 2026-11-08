# V32 本地 CPU 记录（2026-11-08）

基于 `cba5b2e`，分支 `experiment/v32-actor-gathers`。Python3.12.15、Torch2.15.0、pytest9.2.0，macOS CPU；本轮未运行GPU。

| 检查 | 结果 |
|---|---|
| 第一轮专项 | 34 passed / 70.19秒 |
| 补充编译、旧存档、异常回滚 | 3 passed / 44.26秒，含旧存档重测 |
| 最终完整回归 | **209 passed / 1 warning / 649.44秒**，含最终新增36项 |
| 新增Python Ruff和格式 | 3文件通过 |
| train.py / trainer.py静态语法 | 通过 |
| Shell语法 / --help | 通过 |
| git diff --check | 通过 |

原有train.py / trainer.py的Ruff告警分类及数量与基线一致，未新增。唯一pytest警告来自已有TorchScript弃用提示。

```bash
PYTHONPATH=source/reflex:tests /tmp/reflex-wm-venv/bin/python -m pytest -q tests
```

- [完整日志](pytest_full.log)
- [首轮专项](pytest_initial34.log)
- [编译/恢复补充](pytest_compile_checkpoint_rollback.log)

新H32测试对全部想象输出使用零容差，Actor原始梯度、实际Adam参数与moment使用dtype默认容差；没有为了通过而提高容差。`aot_eager`仅验证CPU编译入口及布局变化，不证明GPU Inductor正确，也不提供GPU提速估计。
