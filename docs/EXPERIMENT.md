# 接触后持续抱持奖励实验

公开分支：`experiments/reward-hold`；整理前代码基点：`251e79f`。本次整理不改变运行算法。

## 改了什么

在增维183/305维PPO上沿用C版：放宽缓冲姿态限制、启用compliant_intake/retain，并调整肩部动作缩放；最终加入实际接触与可控状态驱动的持续抱持。保持成功控制时不按固定时间恢复上半身参考姿态和手臂基准。
旧impact/escape负奖励关闭；retain权重0.5、intake权重0.75。AMP、腿部跟踪和安全终止保留；Actor/Critic结构、观测和PPO主流程不变，所有投掷事件照常执行。
详细门槛、接触宽限、状态复位和参考奖励说明见 [实现文档](experiments/C_PERSISTENT_HOLD_ZH.md)。这是多项累计改动，不能当作“只加retain”的单因素实验。

## 环境与数据

按 [官方安装说明](../README_UPSTREAM.md) 安装依赖并执行资产准备。使用自己的授权动作数据重建 motion 与 AMP dataset。将代码安装进 Isaac Lab 所用的 Python 环境，参阅原安装步骤。

```bash
export ISAACLAB_PY=/path/to/isaaclab/python
export REFLEX_DATA_DIR=/path/to/reflex-data
```

这里的路径是占位符，需替换；不依赖历史服务器的 runtime.env。先确认 GPU、数据和输出位置再启动，以下命令会实际训练。

## 训练与评测

```bash
bash scripts/train_C_persistent_scratch600.sh
for seed in 8 9; do
  bash scripts/eval_C_persistent.sh --ckpt /path/to/model_599.pt \
    --condition regular --seed "$seed" --num_envs 320 --out "outputs/persistent_s${seed}.json"
done
```

在干净shell中执行，避免遗留其他实验的BOX_*开关。训练脚本开启persistent+C，禁用A/B和旧负奖励；不加载权重，从零600更新。评测包装器使用相同开关；不能忘记persistent标记后直接用普通C评测。

最终接稳97/640，未优于原版或增维基线；137项CPU测试、GPU短回归、正式训练和评测均有历史证据，具体见 [汇总](EXPERIMENTS_ZH.md)。本次文档整理不代表重新跑了一轮GPU验证。

CPU测试（无需Isaac Lab，需PyTorch和pytest）：

```bash
PYTHONPATH="$PWD/source/reflex" python -m pytest -q tests
```

仅CPU测试不证明仿真效果；GPU结果见历史记录。
