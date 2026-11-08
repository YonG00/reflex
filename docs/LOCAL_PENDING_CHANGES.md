# 待补改动

## V31 Actor H32 诊断（2026-11-08，GPU 待测）

- 目的：直接 profile 当前完整 Actor，并以同权重/起点的最小 Actor＋Physical WM H32 对照判断基础计算与任务附加开销；不先入为主认定事件是主因。
- 入口：`scripts/diagnose_actor_v31.sh`；设计、输出解释和局限见 [V31说明](world_model/V31_ACTOR_DIAGNOSTICS_ZH.md)。独立 `experiment/v31-actor-diagnostics` 分支。
- 不变量：真实 sampler、事件、奖励、AMP、软存活、H32/BPTT、Actor/Critic 更新规则保持；记录范围默认关闭，不覆盖 checkpoint 或已有输出。
- 出错症状：记录后轨迹/梯度/Adam 不一致、起点 SHA 改变、捕获中重编译、反向归属出现错误百分比；报告显式保留失败与无法匹配部分。
- 快速回滚：继续 V30 原训练入口，不使用 V31 三个诊断 mode；独立目录保留诊断记录。
- 验证入口：`tests/test_actor_diagnostics_v31.py`（7项 CPU 通过）；完整回归173项通过、1条已有弃用警告、548.51秒，原始日志见 `docs/world_model/benchmarks_v31_cpu/`。GPU 测试与第三方真实 checkout 集成待开卡，不宣称新的提速收益。

> 当前V30实现与GPU实测见 [V30结果](world_model/V30_GPU_RESULTS_20261107_ZH.md)。下方2026-11-05内容是当时的历史状态，不代表当前仍未做GPU验证。

## V30执行效率（2026-11-07）

- 已验证：FK复用、计时器池精简、全量事件子集消除恒等索引、AMP特征批处理；Actor同组耗时降低约25%。
- 推荐配置：`configs/world_model/joint_execution_v30_verified_actor_trial.json`。不改H32、batch、奖励、AMP、事件或软存活规则。
- 不启用：事件张量融合正确但慢约21%；大图融合部分数值失败；稀疏bootstrap没有明确收益，均默认关闭。
- WM精确零列和H8 Graph约0.25秒/10次更新仅为显式基准，正式编码迁移与连续联跑仍待完成。
- 尚未达成：整轮1秒、长期训练效果验证；本轮未启动正式训练。
- 回滚：继续V29配置即可；旧checkpoint未覆盖，新增执行开关默认关闭。
- 验证入口：`tests/test_actor_execution_v30.py`、`tests/test_amp_feature_batching.py`、`tests/test_event_tensor_fusion.py`、`tests/test_stable_canonical_trial.py`及GPU报告原始JSON。

## Stage 1 联合世界模型（2026-11-05，等待 GPU 验证）

依据用户 V2 方案实现独立 WM/想象 Actor/Critic/真实 AMP 闭环与匹配 Oracle PPO。设计选择、启动命令与限制见 [实施说明](world_model/IMPLEMENTATION_ZH.md)。CPU 合成闭环通过不代表真实 Isaac 训练跑通；未启动 GPU 任务。

- 不变量：原 noisy PPO 和原接物评测标准保留；完整 AMP checkpoint 行为维持。
- 预期判据：真实 FK/观测/AMP/动作 parity 通过、连续窗口不跨注入/重置、真实排序和多步误差通过后才更新 Actor。
- 出错症状：观察重建漂移、状态/默认关节顺序错位、奖励模型被利用、跨重启伪连续、恢复后判别器/优化器丢失。
- 回滚：继续原 `scripts/train_stage1.sh`；新实验用独立日志/checkpoint，不覆盖现有 PPO 存档。
- 验证入口：`tests/test_world_model.py`、`tests/test_amp_checkpoint.py`；开卡后先执行 `--mode preflight`。

## AMP checkpoint 保存与恢复（2026-11-05，已实现并启用）

用户要求后续训练考虑完整保存 AMP 状态。当前上游 `OnPolicyRunner.save()` 仅保存 `model_state_dict`、`optimizer_state_dict`、`iter`、`infos`；本仓库 `MotionOnPolicyRunner.save()` 只扩展了导出逻辑，没有把 `AmpPPO._disc` 及 `_disc_opt` 接入保存。问题不是 `save_interval` 配置。

已在旧服务器检查 13 份 Stage 1 存档（0、50、100、…、550、599），均没有 AMP 状态；日志与输出目录未发现独立判别器文件。无法从这些存档补回旧判别器。

旧保存机制的续训在索引 685 附近暂停，保留最近的 `model_650.pt`。新实现已由实际训练生成 `model_652.pt`，其中包含完整 AMP 状态，并经新进程重新加载验证。修改源代码不会热更新旧进程，本次明确经过了进程重启。

新状态存入同一 checkpoint 的 `infos.reflex_amp_state`，顶层四个字段保持不变，现有项目推理加载接口仍可使用。单 GPU 下保存判别器与 Adam、归一化校验、专家数据 SHA-256、调度标量、全部有效 policy replay，以及 Python/NumPy/PyTorch CPU/当前 CUDA 随机数状态。环境重建后清空跨步配对有效标记，不声称仿真状态或逐帧轨迹完全无缝。

CPU 回归测试通过：保存/恢复后的下一次判别器更新，loss、权重、Adam 状态与随机数抽样均一致，错误专家数据会被拒绝。真实 GPU 检查确认 `model_652.pt` 含 AMP 计数 2、192084 条 replay；文件约 137 MB。4096 环境的新进程已恢复该状态并完成 4 次更新。

正式续训已从该 4096 分支的 `model_656.pt` 启动，恢复 AMP 计数 6 和完整 200000 条 replay，从全局索引 657 继续至 1999，累计目标 2000。实现提交为 `f87649b`，训练尚未完成。

### 需要覆盖

- 判别器模型权重、判别器 Adam 优化器。
- AMP 更新计数、style ramp 进度、最近判别器 loss、自适应跳过门控、style EMA。
- 专家数据与归一化参数的版本/校验，避免恢复到不同数据。
- 明确 policy replay buffer 的恢复策略。当前缓冲区容量 200000×2×83 个 float32，约 126.6 MiB；保存完整缓存会增大 checkpoint，轻量模式必须说明缓存会重建。
- 仿真环境重建后清空跨步配对有效标记，不能把旧环境上一帧与新环境第一帧拼成训练样本。
- 兼容旧 checkpoint：无 AMP 字段时明确警告并记录重新初始化，不把 warm start 当作完整 resume。
- 同一 checkpoint 保存全部相关状态，维持现有推理接口兼容，避免单独权重文件遗漏。

### 验证与改动本意

- 不变量：未启用 AMP 的训练/推理照常工作；恢复后 AMP 参数、优化器、调度与已声明缓存策略一致。
- 出错症状：resume 后 ramp 归零、判别器 loss 突变、Adam 动量丢失、缓存被错误配对或 GPU 内存异常增加。
- 验证入口：CPU 上执行 train→save→load，对比判别器输出、优化器与计数，并验证恢复后的下一次更新；覆盖旧存档降级路径和实际存档字段检查。
- 回滚：保留旧 checkpoint，撤销 AMP 保存/恢复扩展；必须恢复为明确警告的非完整 AMP resume。

完整 replay 在填满后约 126.6 MiB，因此新 checkpoint 比原先约 7 MB 的 PPO 存档明显增大。已验证范围为单 GPU；本次未验证多 GPU 各 rank 私有缓存和随机数的逐 rank 恢复。
