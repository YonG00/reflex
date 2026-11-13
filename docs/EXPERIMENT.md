# SuperMimic 思路的世界模型尝试

公开分支：`experiments/supermimic-wm`；整理前代码基点：`7dd548b`。本次整理不改变运行算法。

## 实验范围

这是把 SuperMimic 部分思路移植到 Reflex 的探索，包括状态世界模型、想象Actor更新、解析软终止、Critic箱子信息补齐及执行效率/恢复契约探索。不是完整SuperMimic训练复现，也不是原版PPO对照分支。

- [整体实现](world_model/IMPLEMENTATION_ZH.md)
- [V29解析软终止与Critic迁移](world_model/V29_ANALYTIC_SOFT_TERMINATION_AND_CRITIC_ZH.md)
- [最终40分钟续训及独立视频诊断](world_model/RESUME_NOV08_UTD04_40MIN_ZH.md)
- 第三方参考代码许可证：`docs/world_model/SuperMimic-Reproduced.LICENSE`。代码公式的引用版本与来源保留在V29文档。

各版本配置和checkpoint schema不同，不能把PPO分支183维观测与本线179维等历史布局直接混用，也不能静默迁移。

## 环境与数据

按 [官方安装说明](../README_UPSTREAM.md) 安装依赖并执行资产准备。使用自己的授权动作数据重建 motion 与 AMP dataset。将代码安装进 Isaac Lab 所用的 Python 环境，参阅原安装步骤。

```bash
export ISAACLAB_PY=/path/to/isaaclab/python
export REFLEX_DATA_DIR=/path/to/reflex-data
```

这里的路径是占位符，需替换；不依赖历史服务器的 runtime.env。先确认 GPU、数据和输出位置再启动，以下命令会实际训练。

## 入口与验证边界

```bash
bash scripts/train_stage1_world_model.sh --help
```

该入口先检查motion/AMP数据，再转发world_model.train参数。正式运行前选择与目标实验一致的 `configs/world_model/*.json` 和文档；不提供一个混合所有优化的“通用最佳配置”。配置、checkpoint、迁移标记必须匹配，具体命令见对应版本文档，历史服务器路径需自行替换。

历史最终40分钟续训达到Actor1722、WM梯度22220；独立15秒单环境诊断3次完整投掷、严格接住0次。该诊断分布与PPO CLEAN2批量评测不同，不能直接比较成功率或声称优于PPO。CPU正确性、GPU性能和真实接箱效果应分别阅读。
