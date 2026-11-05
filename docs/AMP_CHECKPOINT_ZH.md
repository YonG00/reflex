# AMP 存档与恢复

上游的 PPO 存档不包括 AMP 判别器及其优化器。本仓库现将 AMP 状态放在同一 `.pt` 文件的 `infos.reflex_amp_state` 中；不再依赖容易遗漏的独立权重文件。

## 保存内容

- 判别器权重和 Adam 优化器。
- AMP 更新次数、style ramp 配置/进度、最近 loss、跳过更新门控及 style EMA。
- 专家数据 SHA-256 和归一化统计，用于拒绝错误数据。
- 完整有效 policy replay、容量、计数和环形指针。
- Python、NumPy、PyTorch CPU 及当前 CUDA 随机数状态。

replay 容量为 200000×2×83 个 float32，填满后约 126.6 MiB。加上模型和优化器，新存档约 140 MB，仍按每 50 iterations 保存。

## 恢复边界

`MotionOnPolicyRunner` 在训练时调用 AMP 保存/恢复方法。旧文件没有 AMP 字段时，会明确打印 `LEGACY` 警告并重新初始化 AMP；不会假装完整恢复。仅加载模型（`load_optimizer=False`）时不会恢复 AMP 或随机数。

仿真状态没有保存，因此新环境上一帧的配对有效标记会清零，避免将旧环境和新环境的观测拼成错误样本。完整 AMP 状态恢复不等同于整个物理仿真逐帧复现。当前验证范围为单 GPU。

加载自己的完整 checkpoint 时使用 `torch.load(path, weights_only=False, map_location="cpu")`，AMP 字段包含 NumPy 随机数状态；项目现有训练和评测加载代码已使用 `weights_only=False`。

## 验证

```bash
python -m unittest discover -s tests -p test_amp_checkpoint.py -v
```

CPU 测试比较保存/恢复后的下一次判别器更新：loss、模型权重、Adam 状态、随机数抽样、调度计数与 replay 均匹配，并验证不同专家数据会被拒绝。

GPU 实测：从旧 `model_650.pt` 训练索引 651、652，生成含 AMP 状态的 `model_652.pt`；新进程读取它，恢复 AMP 计数 2 和 192084 条 replay，再完成索引 653–656 的训练。测试日志位于 `outputs/experiments/amp_checkpoint_validation_20261105/`。
