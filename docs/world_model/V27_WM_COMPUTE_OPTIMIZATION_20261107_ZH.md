# V27：优先优化 WM 主计算链（本地待 GPU 验收）

## 状态

用户要求先改收益大的计算路径，改好后再开卡。
GPU 测试已停止，服务器连接随后关闭；本轮后续只进行本地 CPU 修改与验证。
没有启动正式训练，也没有覆盖原 checkpoint、原服务器工作区或此前实验结果。

本轮不是继续以日志、Replay 缓存为主线，而是直接减少 WM 的 FK 工作，并为完整监督步融合与 MLP BF16 准备独立候选。
实际 GPU 提速、BF16 对效果的影响、Inductor/CUDA Graph 数值一致性仍未通过验收，不提前给出倍数。

## 本次短暂开卡得到的事实

RTX 4090 24GB，PyTorch 2.7.0+cu128；独立测试工作区：
`/root/gpufree-data/reflex-v26-test`，基底提交 `7ed2ef8`，另有诊断入口补丁。
原工作区 `/root/gpufree-data/reflex` 的未提交改动保留。
只读基线另设 `/root/gpufree-data/reflex-v26-baseline`（`63b3ec8`）。

共同起点为原奖励 V24/H32 的既有 checkpoint：
`/root/gpufree-data/reflex/outputs/world_model/20261106_v24_end_to_end_throughput/4096_rate0.4/checkpoint_final.pt`。
它的 Replay 容量 128、已有 65 帧 × 4096 环境，Actor 更新 350 次、WM 更新 3500 次。

| 验证/测量 | 已观察结果 | 边界 |
|---|---|---|
| 24 步真实事件一致性 | 指令及 Post-control 状态最大误差 0；事件状态最大误差约 1.49e-8 | 64 环境，只读无优化器更新 |
| H32 等待→发射 | 64 条转换；发射速度对预测机器人位置梯度范数 9.17744 | 非真实接箱效果验收 |
| 外部扰动赋速 | 202 次正向扰动写入 | 与原事件机制对照 |
| GPU 基线 WM10+AMP | 2.038～2.125 秒 | 固定 Replay，不含真实采集 |
| GPU 基线 Actor+Critic | 2.664～2.746 秒 | 固定 Replay，不含真实采集 |
| GPU 基线整个更新 | 中位数 4.800 秒 | 仅 3 次稳态样本，不是完整训练迭代 |
| 首次整更新 | 75.16 秒 | 包含冷编译；不混进稳态 |
| V26 优化首个样本 | WM+AMP 2.049 秒，Actor+Critic 2.648 秒 | 只有一个候选样本，数值对照未通过，不能宣称提速 |
| 4096×1000 Replay payload | 30.6549 GiB | 不含 PhysX、模型、优化器、BPTT、索引缓存，24GB 放不下 |

同一基线重复运行的 Actor 梯度范数略有波动（0.6385268～0.6385361）。
V26 对照有一个 149504 元素张量的 3 个元素超出既定容差，最大绝对差 2.2717e-5；
尚未证明是 GPU 重复性噪声还是实现差异，因此没有宣布数值通过。
随后尝试独立确定性验证，但冷编译成本变高；按用户“先改收益最大”的指示中止，未完成该对照及 CUDA Graph 测量。
**不能把未通过对照的 V26 或本地 V27 当成已完成 GPU 验收。**

上述远端原始目录为 `outputs/world_model/20261107_v26_gpu_validation/`。
事件检查首轮因 libtorch_cuda_linalg.so 搜索路径失败，source `scripts/local/env.sh` 后在 `events_r2` 通过。
`paired` 是首次数值未通过的报告；`paired_r2` 是中止的确定性试验，不能作为完成结果。
原始事件、parity、Replay schema 与首次数值未通过的 paired JSON 已在重新开机后归档到 `benchmarks_v26_gpu_20261107/`；中止试验没有完成报告。

## 改造一：终止监督只算真正需要的 FK

原来的 `fit_model` 每个 H8 监督步均调用两次完整 `_fk()`，输出全部 link 的位置、四元数。
但 `predicted_termination_features` 实际只用：

- 前状态：躯干 anchor 的位置、四元数，用于参考对齐。
- 后状态：tracked bodies 的位置和 anchor 四元数。
- H8 第一步若提供真实采集时的 `reference_relative_p`，根本不需要前状态 FK。

`physics.Kinematics.subset()` 复用原 URDF 递推和注册缓冲，仅跳过没有被请求且不是祖先的节点。
不是近似 FK，没有更换坐标系、四元数公式、关节顺序或位置梯度。
本地 G1 URDF 总共 39 个递推节点，anchor 需要 3 个，tracked 集合需要 29 个。
有初始记录参考坐标时，一个 H8 窗口的节点递推次数可由 `8×(39+39)=624` 减至 `7×3+8×29=253`。
这是 **FK 工作量计数**，不是整个 WM 耗时降低 59.5% 的承诺；MLP、损失、反传和采样仍占时间。

涉及：`physics.py`、`task.py:predicted_termination_features`、`trainer.py:fit_model`。
`minimal_termination_fk` 是独立开关，历史配置默认关闭。
Actor 的奖励、完整物理/姿态计算和原生事件未更改。

## 改造二：完整单步监督图，而不只编译 WM 前向

新增 `Trainer.v24_model_training_step()` 与 `compile_wm_training_step`：

```text
本步状态 + 控制目标 + 真实事件条件
    → Physical WM（canonical / normalizer / MLP / integrate）
    → 必要的前后 FK → 预测状态终止特征 → Termination BCE
    → Physical state loss
    → 确定性 Post-control 写入 → post 状态及监控误差
```

把这条计算链交给同一个 fullgraph 编译入口。
返回 Post-control 状态、Physical loss、BCE、post 监控误差，保留全部连续梯度。
外围仍执行原 H8 自回归、合法窗口 mask、窗口均值、损失权重、逐次 backward/梯度裁剪/Adam。
WM-only 500 轮调度、每轮 10 次梯度更新、Actor 2048/H32、UTD0.4 均保留。
这里的图是 **单步完整监督图，不是整个 H8/优化器/仿真都已捕获成 CUDA Graph**。

只允许条件化、确定性事件写入、联合预测 Termination BCE 的 V24 路径使用；不支持的旧架构明确报错。
正常外部事件仍在窗口内，不增加硬终止或跳帧，不将未来状态用于动力学输入。
标签只进入损失；Actor 自主想象不调用这个监督入口。
旧 learned post-control 对照会明确关闭这个只支持确定性写入的融合开关，避免悄悄改变它的训练目标。
对照克隆器同时保留来源的 WM 预热进度，避免重置预热状态。

## 改造三：仅 WM MLP 的 BF16 试验

`models.PrecisionMLP` 仅包装 WM，保留 Sequential 参数命名及网络层结构。
Actor/Critic/Termination 仍使用原生 nn.Sequential，并新增 Actor TorchScript 导出回归，避免混合精度包装影响原导出路径。
只在 Physical WM 的 `net` 中启用 BF16 autocast，输出立即转回 FP32。
权重、权重梯度、Adam 一二阶状态仍 FP32；canonical、积分、FK、Reward、事件写入和状态损失均在 BF16 scope 外。
Actor179、Critic292、Termination Head 与 AMP 没有切到 BF16，没有修改奖励、碰撞或物理精度设置。

`wm_mlp_precision="fp32"` 为默认，`"bf16"` 仅供独立试验。
混合精度必然可能产生数值差异，不能按执行等价优化直接采用，也不能仅凭 loss 有限就宣布效果不受影响。
必须检查 H1/H8/H16/H32、launch/contact 阶段误差、梯度方向及范数、实际 Adam 更新与短真实任务回归。
CPU 已验证 dtype 边界、有限反传、FP32 Adam 状态及保存恢复，不代表 GPU BF16 已验收。

## 独立配置

均基于 V26 原奖励 H32 候选，保留 4096 测试目标、Actor2048、WM3×2048/H8、UTD0.4。
环境数量仍由启动命令指定，配置里的 Replay 容量暂保持 128。

| 配置 | 最小 FK | 单步监督融合 | WM MLP |
|---|---:|---:|---|
| `joint_efficiency_v27_minimal_fk.json` | 开 | 关 | FP32 |
| `joint_efficiency_v27_fused_wm_trial.json` | 开 | 开 | FP32 |
| `joint_efficiency_v27_bf16_trial.json` | 开 | 关 | BF16 |
| `joint_efficiency_v27_fused_bf16_trial.json` | 开 | 开 | BF16 |

先逐项消融，再看组合；不拿 H16 或新奖励与原 H32 比速度。
384 步容量独立候选约 11.7715 GiB payload，**尚未分配或验收**；需要完整装满 Replay 后测训练峰值，不能用空缓冲通过代替。

## CPU 验证

- subset FK 的位置、四元数、输入梯度与原递推精确相等。
- 最小 FK + 联合预测 BCE 的实际 train/save/load/train 对照，WM/Termination 参数与两个 Adam 状态精确相等。
- 固定 `63b3ec8` 的更新对照工具加入 `--minimal-termination-fk`，开启联合 BCE 后，WM/Actor/Critic/Termination 梯度、参数及四个 Adam 状态最大差均为 0。
- 完整监督单步经 CPU `aot_eager/fullgraph` 捕获，初始步与后续步的状态、损失、梯度及裁剪后的实际 Adam 更新均作对照。
- 正常发射位于 H8 合法窗口内；融合训练 → save → load → H32 从等待经历发射 → Actor 更新链通过。
- BF16 范围检查、有限梯度、FP32 参数/Adam 以及 BF16 存档按显式 FP32 执行选项恢复通过。

精确最小 FK 对照原始结果：`benchmarks_v27_cpu_20261107/minimal_fk_optimizer_equivalence.json`。
AOT Eager 验证的是捕获及 autograd 语义，不是 CUDA Inductor kernel、CUDA Graph 或实际 GPU 性能。

```bash
PYTHONPATH=source/reflex python scripts/local/check_efficiency_updates.py \
  --baseline-ref 63b3ec8 --minimal-termination-fk
PYTHONPATH=source/reflex python -m pytest tests/test_wm_compute_acceleration.py -q
```

最终完整 CPU 回归 **95 项通过，241.17 秒**；新计算路径的 7 项专项测试通过（25.07 秒）。
Actor 导出测试仅产生本地 PyTorch 对 TorchScript 的弃用提示，输出与原 Actor 精确相等。
新测试/诊断工具 Ruff 检查通过，改动核心模块的关键错误检查通过。

## 下一轮开卡测试入口及判据

先检查旧进程、GPU、代码实际导入路径与源文件哈希；使用新的工作区和输出目录。
恢复相同 checkpoint/Replay/初始权重、随机条件；已训练 Actor 不重复 500 轮预热。
下面在准备好环境的仓库中运行（必须先设置正确 REFLEX_DATA_DIR 并 source scripts/local/env.sh）：

```bash
bash scripts/train_stage1_world_model.sh \
  --mode efficiency_benchmark --num_envs 4096 \
  --config configs/world_model/joint_efficiency_v27_minimal_fk.json \
  --checkpoint /实际路径/原奖励H32共同起点.pt \
  --output outputs/world_model/新时间戳_v27_minimal_fk --no_debug_vis
```

然后分别对三个 trial 配置创建新目录重复。
入口读取固定旧版 `63b3ec8` 作为原执行路径；每次测量恢复相同模型/优化器/AMP/RNG、事件 RNG 和预热计数，不重新采集 Replay。
包含 WM+AMP10、Actor/Critic、冷编译、稳态、显存与真实参数/Adam/梯度对照。
每个变体还输出初始权重的条件化及自主 H1/H8/H16/H32 预测误差。
数值超阈值会单独标记 `passed=false` / `update_equivalence_passed=false`，仍保留性能诊断；完成测试不等于数值通过。
不通过的路径不用于正式训练。GPU 重复性噪声需要补足同基线重复对照，不通过放宽容差掩盖。

默认不再自动进入耗时的确定性重新编译；`--efficiency_deterministic` 是独立数值试验选项。
`--efficiency_include_cudagraph` 才追加 reduce-overhead 候选；融合监督入口也对应使用 graph 模式，避免标签叫 graph 实际却测默认 WM。
必须用 Profiler 证明确有 cudaGraphLaunch/减少 kernel，不能从选项名判断成功。

`efficiency_capacity_check` 独立装满指定容量再做 4 次完整更新，不保存试验权重。
完整一轮真实采集+联合训练仍需独立限时 train benchmark，不能把固定 Replay 的更新耗时直接叫端到端吞吐。
若完整图 cold compile 超过可接受预算，记录失败/时间并回退；不以长时间等待替代实际收益。

开卡后先测最小 FK、完整 FP32 监督步融合，再看 BF16；收益、数值/梯度/参数更新和短真实回归共同决定是否采用。
本轮没有开始半小时训练。
