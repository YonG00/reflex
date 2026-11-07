# V26 效率优化：本地实现与 CPU 验证

## 范围与当前结论

2026-11-07，核对 GitHub main 为 `63b3ec8a89963f8a90366a867f12d814dcf8142a`。
沿用用户提供的两个 SuperMimic 复现调研，不重做全网调研，不替换 PhysX。
当前无卡：已实现 CPU 可验证的执行优化和独立实验配置，GPU 性能和真实任务效果尚未验收。
不得把下文 CPU 索引缓存命中时间换算成训练加速倍数。

保留 V24 原奖励、Actor 179 / Critic 292、WM 3×2048、WM H8、Actor batch 2048 / H32、
UTD 0.4、每轮 WM 更新 10 次、AMP、确定性原生事件和纯软存活。
V25 的站立优先奖励与 H16 没有混入本轮候选，也没有覆盖其配置、结果或存档。

## 已定位的重复工作与修改

| 位置 | 原问题 | 本轮修改 |
|---|---|---|
| `replay.py` | 每次采样重复 nonzero、合法性和留出环境筛选；环形缓冲多次 cat 同一有效性元数据 | 缓存合法窗口、训练/留出池、想象起点及有序 valid/edge；每次新增数据或相关张量原地修改时失效 |
| `trainer.py:fit_model` | 10 次 WM 更新都提取指标，前 9 份最终被覆盖 | 只计算、提取最终更新的指标；反传、裁剪、优化器步骤和非有限梯度检查仍逐次执行 |
| `joint.py` | 20 个 Critic minibatch 分别把 loss 转为 Python 标量 | 保存 detached GPU 标量，最后用 FP64 聚合后取出；参数、优化器仍 FP32 |
| `Trainer.sample` | 为采样计时，在每次 gather 前后显式同步 CUDA | 新候选关闭这些专用计时栅栏；仍按同一 CUDA stream 顺序执行；原同步计时可选保留 |
| `train.py` | 每个采集步 TensorBoard flush、打印全部 JSON | TB 后台 30 秒 flush，控制台每 10 步输出；JSONL 与 TensorBoard 标量仍逐步保留，结束 close 会 flush |
| `trainer.py` | joint 的 ready 绕过常规 warmup | 独立 WM-only 完成轮数及 Actor/Critic 门控，持久化到 checkpoint |

缓存不保留随机采样结果，不改变有放回均匀采样或随机数消耗。
签名包含 count、pointer、相关张量对象与 `_version`，可检测恢复时对最后一帧 boundary 的修改。
正常发射、悬停和重投没有新增监督边界，窗口仍使用既有 boundary/terminated/timeout。
缓存仅驻留当前 Replay 版本的元数据，checkpoint 不序列化缓存。
返回的索引是内部只读缓存，调用者不得原地修改。

`time/replay_sample_host_seconds` 是去掉计时同步后的主机提交时间，**不是 GPU gather 耗时**。
只有打开 `synchronize_replay_timing` 才输出原 `time/replay_sample_seconds_inclusive`。
WM 完整阶段、Actor 完整阶段和 Critic 完整阶段仍有同步计时边界，避免异步提交冒充提速。
梯度裁剪中的非有限检查、实际动态索引产生的同步尚保留，不通过删安全检查换速度。

## WM-only 调度与恢复

独立候选 `configs/world_model/joint_efficiency_v26.json` 显式设置 `wm_only_rounds=500`。
每次完整 `fit_model()` 增加一个完成轮数；采集步、验证次数均不计入。
前 500 轮继续采集真实数据、训练 WM/联合 Termination BCE/真实 AMP，不更新 Actor/Critic。
第 501 轮完成 WM 更新后才进入联合更新。10 次/轮意味着预热阶段 5000 次 WM 梯度更新。
`ready()` 和直接 `update_actor()` 均执行门控，避免不同入口绕过。

checkpoint 新增 `wm_warmup.rounds_completed` / `inherited_trained_actor`。
恢复中途预热保留进度；旧 checkpoint 的进度由 WM 更新计数推导。
已有 Actor 更新的模型恢复时标记继承已训练策略，避免重新强制预热。
显式重建 Replay 不清除 WM-only 已完成进度，仍执行既有真实采集 initial_collect_steps。
旧实验配置默认 `wm_only_rounds=0`，保持既有实验可复现；从零新实验应使用新候选配置。

## Replay 容量：先核算，再扩容

本轮候选暂保留容量 128，便于与此前 4096/H32/rate0.4 短测对照。
**1000 步/环境尚未启用**，不能仅从状态 84 维估计 Replay 显存。
真实结构还有物理参数、事件控制状态、事件池、前后状态及监督等字段。
`Replay.memory_report()` 按 tensor dtype、完整 shape 和 global_fields 核算真实 payload，另报当前索引缓存字节。
训练第一次真实采集后输出 `replay_memory.json`，含当前容量与目标 1000 步的实际 schema 投影。
它不包含 PhysX、模型、优化器、H32 BPTT、编译缓存与 allocator 余量，不能直接当整卡峰值。

可先在无卡机器上用已有、包含 Replay 的 checkpoint 核算目标 4096×1000：

```bash
PYTHONPATH=source/reflex python scripts/local/benchmark_replay.py \
  --checkpoint /实际路径/checkpoint.pt --schema-only \
  --target-envs 4096 --target-capacity 1000
```

global_fields（如 event_pool）按真实全局字段计算，不乘环境数量。
无 Replay 的瘦 checkpoint 会明确报错；应从新采集获取 schema，不能猜测其容量。
开卡后先记录小容量完整训练峰值，再估算新增 payload/cache 与整卡安全余量；不在 24G 卡上盲目分配 4096×1000。
若不可行，分别报告可行容量和差距，不静默调整训练配置。

## 编译候选及尚未做的优化

已保留原 packed_heads 融合，不重复实现已验收的事件最小视图/物理属性缓存。
新增 `wm_compile_mode`，默认 default；独立 `joint_efficiency_v26_cudagraph_trial.json`
只把 WM `_dynamics` 的 Inductor mode 改为 reduce-overhead，供自动 CUDA Graph 对照。
这是**待 GPU 验证的试验开关**，不是已证明捕获完整 WM/优化器更新的手写 CUDA Graph。
H8 反传中输出复用、WM 训练/Actor 冻结梯度切换、编译重建和 cold start 均需检查。
未通过前不默认开启，不修改已有稳定 normalizer 的 copy_ 缓冲地址策略。

更大范围 Actor 融合此前已有冷编译超预算记录，不能以新增一个 compile 开关声称完成。
BF16 MLP 本轮未启用；需要 GPU 证明输出、梯度、Termination BCE、实际 Adam 更新及长期效果可接受后再单列消融。
目前 FP32 物理、积分、FK、Reward、优化器和原生事件规则均未修改。
仍可进一步排查预检查时丢弃的 Replay batch、动态索引、事件调度与额外 FK；本轮没有用改变随机数序列来悄悄删除这些操作。

## CPU 实测

小型合成动力学（WM 3×16 / H2，Actor batch4 / H3）、相同初始权重与随机种子，对比固定旧版本 `63b3ec8`：
WM + Termination 梯度、Actor/Critic 梯度、网络参数以及四个 Adam 的状态最大差均为 **0**。
此对照实际执行 backward/step，并包含候选的 Replay 异步计时路径；不代表 GPU 编译数值或 PhysX 一致性已经通过。

1000×4096 合成 valid/edge 元数据、CPU 单线程、中位数：

| 操作 | 耗时 |
|---|---:|
| 原版 H8 索引重建 | 44.38 ms |
| 新版 H8 未命中路径 | 35.18 ms |
| 新版 H8 缓存命中 | 约 0.001 ms |
| 实际 variable-window 起点池，无缓存 | 44.11 ms |
| 同一起点池，缓存命中 | 约 0.001 ms |
| 清空索引缓存后首次 H8 重建 | 37.47 ms |

H8 索引结果逐元素精确相等，有效窗口 3,621,516。
每次真实新增数据后仍要重建一次，表中命中时间不能代替整轮采样成本；也未包含大字段 gather。
原始结果在 `benchmarks_v26_cpu_20261107/`。

复现命令（使用本地 CPU PyTorch/pytest 环境）：

```bash
PYTHONPATH=source/reflex python scripts/local/benchmark_replay.py --device cpu --steps 1000 --envs 4096
PYTHONPATH=source/reflex python scripts/local/check_efficiency_updates.py --baseline-ref 63b3ec8
PYTHONPATH=source/reflex python -m pytest tests -q
```

测试结果：完整套件 87 项通过（213.44 秒）；随后追加的容量投影测试也通过，新增 12 项小套件全部通过（3.89 秒）。
原有 76 项与新增 12 项均已有通过记录。Ruff 对新工具、Replay、Joint 和新增测试检查通过；训练入口和 Trainer 的关键错误检查通过。

新增测试覆盖未满/满/回绕 Replay、训练/留出集、固定种子采样与 RNG 一致、边界原地修改、load/add 缓存失效、
内存核算、预热第 499/500/501 轮、真正 train→save→load→train→Actor 更新、旧 checkpoint 继承，以及关闭采样计时栅栏。

## 开卡后的验收顺序

1. 固定卡型、代码 SHA、PhysX/Isaac/PyTorch 版本，检查现有进程与显存；新目录存放所有结果。
2. 使用同一原奖励 H32、4096 环境、Actor2048、WM3×2048/H8、UTD0.4，准备同一个有 Replay 的兼容 checkpoint。
   不用 V25 H16/站立奖励与 V24 H32 混比；若需重建 Replay，先一次采集并存成共同起点，不能各变体重采不同数据后声称严格等价。
3. 基线显式关闭索引缓存、保留同步采样计时；候选仅打开缓存/异步采样计时/日志优化。
   分别测 WM 10 次更新、Actor 前向/反向/完整更新、Critic、完整轮次与真实 transitions/s，冷编译及保存开销单列。
   保留最后一步 WM 日志优化的旧代码对照可使用独立 worktree 的 `63b3ec8`，不覆盖现有 checkout。
4. 复跑已有真实事件一致性、等待起点 H32 发射转换和梯度、完整 Actor/Critic Adam 对照；抽检 Physical/Reward/Termination。
5. 单独测试 reduce-overhead；同一 Replay、权重与随机条件检查 H1/H8/H16/H32、梯度及实际参数更新。
   使用 Profiler 判断是否真的捕获 CUDA Graph、是否发生 graph break/重复编译；失败则保留 default。
6. 记录 PyTorch allocated/reserved 峰值与整卡峰值，核算 4096×1000；容量变化单列，验证起点池构建和采样开销。
7. 从零验证 WM-only 调度（500 轮内 Actor/Critic 参数不变，5000 WM steps，保存恢复不重置），再做限定短联合回归与视频/摔倒/成功率检查。
8. 只有 GPU 数值/事件/性能验证通过才推荐正式训练；本轮没有启动任何 GPU 测试或长训练。

GPU 性能表目前全部待测：纯采样、WM10、Actor F/B/完整、Critic、完整迭代、显存峰值、冷编译成本。
不得预填复现仓库的 0.32/0.405 秒，或把本机 CPU 缓存时间当成新的端到端吞吐。
