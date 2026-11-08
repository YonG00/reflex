# V31：定位 Actor H32 的开销

## 当前状态

本分支基于 main `735e702`，只增加诊断入口与可开关的记录范围，不改变训练算法，不启动长训练。GPU 诊断已完成，详见 [2026-11-08结果](V31_GPU_RESULTS_20261108_ZH.md)：完整链路通过记录一致性检查，第三方原生最小链路通过，Reflex新编译边界仍有数值超差。V30 的旧 trace 早于最终 AMP 特征批处理和全量事件子集优化，不能用来解释当前约 1.8 秒 Actor 的各项占比。

已有数据只用于提出问题：V30 同组 Actor 2.5077→1.8709 秒；WM 约 0.25 秒是独立显式 Graph 基准。第三方约 0.29 秒 Actor/Critic 的任务、输入与奖励不同。不能把这些不同实验相减后，把差值全部归因于事件或 FK，也不能据此宣称完整一轮已低于 1 秒。

## 两个互补实验

### 完整链路

`diagnostics_v31/full.py` 调用当前真实 `Trainer.update_actor()`，保留 Replay 采样、H32 递推、奖励、AMP、事件、确定性写入、存活权重、Actor BPTT 和 Critic 更新。

记录观测、Actor、动作目标、Physical WM、FK、packed Reward/Termination、事件输入 gather、原生调度、事件合并与状态携带、确定性写入、批量 AMP 特征/判别器、Return、Critic、裁剪及 Adam。记录边界放在 eager 调用层，不包裹 `_view()`，避免破坏事件最小视图的函数身份判断；不逐模块调用 synchronize。

先恢复相同 checkpoint 参数、已有 Adam 动量和随机状态，预热 2 次；无 Profiler 测完整更新 20 次，报告 median/P90 和峰值显存。然后预热带记录范围的调用，单独捕获一次完整 CPU/CUDA trace。捕获期间若产生新编译，`attribution_usable` 为 false，该 trace 不能当稳态瓶颈证据。

实际 Replay sampler 始终执行，不替换为缓存 batch 的伪 sampler；固定 RNG 让起点相同。另测索引缓存冷/热采样。恢复参数、随机状态和缓存的耗时位于计时区间外。这是冻结 Replay 的更新基准，不是持续采集的端到端吞吐。

未加记录和加记录版本比较完整想象输出、Actor 梯度，以及实际 Actor/Critic 更新后的参数和梯度。结束恢复原 Actor/Critic、Adam、归一化、事件 RNG 与 Replay 索引缓存，不保存训练 checkpoint。覆盖统计列出等待→发射、重投返回等待、截断次数；没有触发的分支不能据此声称通过事件专项验证。

### 最小可微计算链

`diagnostics_v31/minimal.py` 从完整进程导出只包含张量与基础类型的 `minimal_capsule.pt`。完整链路起点和导出起点 SHA 必须相同。离开 Isaac 进程后，加载原 Actor、Physical WM、归一化及动作控制映射，做 H32 顺序递推及真实 BPTT。

三组仅改变编译边界：

| driver | 执行方式 |
|---|---|
| eager | Actor/WM eager |
| segmented | 仅 Physical WM 单步编译 |
| single_step | Actor→目标→Physical WM→诊断观测投影一起单步编译 |

H32 外层保留 Python 递推，不编译 H32 大图。所有组使用同一初始权重、输入、随机状态、诊断损失和初始化的 Adam。WM 参数冻结，但不使用 `no_grad`，检查最终物理状态损失对 Actor 叶参数的梯度非零，并逐元素比较该梯度；比较轨迹、动作、损失、梯度和实际 Adam 参数更新，不放宽 `torch.testing.assert_close` 默认容差。不能直接对编译函数返回的第一步动作求 VJP 来衡量完整物理梯度：动作和物理状态在编译边界处是并列输出，该方法会遗漏同一步 WM 的依赖；初版该诊断字段已被替换，不涉及训练梯度改动。

这是独立计算对照：用固定可微投影代替任务观测更新，固定 context/事件特征，没有任务 FK、奖励、AMP、调度和真实仿真。诊断损失为状态偏移平方均值加小幅动作项，不代表 Reflex 训练目标。不能把它的速度当完整 Actor 的速度，不能用于判断站立/接箱效果。

`tracker_native.py` 可读取用户已有 TrackerLab 代码的原生 `Policy` 和 `WorldModel.step`，用同一计时框架、FP32/TF32 off、batch 2048/H32 做独立对照。输入与权重为确定性合成；为避免零输出 WM 初始化让物理梯度为零，明确给全零 Linear 输出头添加 std=0.001 扰动，并在报告中标识。它不是训练复现，也不是同一模型的因果消融：网络尺寸、历史观测和状态结构仍不同。服务器提供的目录没有 `.git`，因此修正为记录 `revision=null`、缺失原因与实际网络源码 SHA，而非伪造 commit。GPU 实际结果另见本轮结果报告。

## GPU 使用

使用包含本分支的新 checkout，已有数据和真实训练 checkpoint，不覆盖旧输出：

```bash
export ISAACLAB_PY=/实际环境/bin/python
bash scripts/diagnose_actor_v31.sh \
  /绝对路径/checkpoint.pt \
  /绝对路径/新的诊断目录 \
  /root/trackerlab-supermimic
```

第三个参数可省略。额外的第三方 Python 依赖应由现有环境提供，可用 `PYTHONPATH` 补充；脚本不会下载、修改第三方代码或安装依赖。默认每个进程最多 2400 秒，可通过 `V31_STAGE_TIMEOUT` 修改；重复次数默认 20，可通过 `V31_REPEATS` 修改。首次编译列入 warmup，不混入稳态统计。

脚本固定现有 V30 verified 配置、4096 真实环境、Actor 2048/H32，以 `--reinitialize_replay` 在独立进程采集新 Replay。不会恢复旧 Replay，不影响原 checkpoint。完整与最小 Reflex 实验共享此次采集的冻结起点；第三方原生实验使用自身明确标记的合成起点。

输出：

- `full/actor_full_v31.json`：稳态统计、正确性校验、采样 SHA、覆盖次数、编译计数、算子表及反向归属。
- `full/actor_full_trace.json`：Chrome/Perfetto trace。
- `full/actor_full_trace_summary.json`：仅按 trace 中 kernel/memcpy/memset 类别计算的活动并集、短 kernel、同步及传输统计；排除 `gpu_user_annotation`。首轮旧报告中的 timeline 含 GPU 用户范围，应以此独立重算结果为准。
- `full/minimal_capsule.pt`：可移植 Reflex 最小计算输入与权重。
- `reflex_minimal/minimal_actor_v31.json`：三种编译边界的前向、BPTT、裁剪/Adam、完整更新耗时及数值检查。
- 可选 `tracker_minimal/minimal_actor_v31.json`：第三方原生最小链路，不能当第三方完整训练基准。
- `*.log`、`revision.txt`、`working_tree.txt`、`gpu.csv`：进程、版本与频率/显存遥测。

脚本检查 `completed`、`numeric_passed`、完整报告的 `attribution_usable` 和最小链路物理梯度，不只检查退出码。超时或失败保留日志/部分报告。实验可能消耗大量 host 内存保存 trace；GPU 实测前不承诺峰值显存或冷编译耗时。

## 如何据数据决定下一步

CPU self 是各线程工作量，GPU kernel 时长和是直接关联工作量，都不是独占 wall-clock 占比；不可把嵌套范围耗时相加。反向按 autograd sequence 与 forward thread 归属；共享或无法匹配的工作单列 `backward/unmatched_or_shared`，不猜测分摊。

CUDA timeline 报告活动区间并集与未观测活动间隙，包含拷贝，不等于 SM 利用率。需结合 trace 中同步、kernel 启动、索引与编译计数解释，CUDA Graph 内部事件缺失也可能造成观测空白。

先对照以下证据，再修改热点：

1. 最小同模型三组的前向/BPTT差距，判断编译边界本身的收益。
2. 完整链路的 CPU 调度、索引/同步及 GPU 工作，判断是启动开销、网络计算还是反向图规模。
3. FK/事件输入 gather/状态 carry、packed heads、AMP 已批处理部分的真实成本，避免重复优化。
4. 第三方原生最小链路的模型规模和耗时，辅助判断基础网络计算差异；保持任务差异单列。

当前源码可观察到事件调度前仍有较广泛的字段/pose gather，以及每步 pose/state carry。它们只是待测候选，不是已证实主因。本次不删除功能、不改变事件/梯度，也不将任何未经 GPU 验证的优化放入正式路径。

## 本地验证

专项 7 项 CPU 测试已通过（69.22 秒），覆盖实际 sampler、带非空 Adam 的恢复、记录范围异常恢复、packed 编译入口、序号反向归属、重叠 timeline、H32 最小链路及第三方接口 fixture。完整 CPU 回归 **173 passed、1 warning、548.51 秒**；警告来自已有 TorchScript 弃用提示。日志和环境见 [CPU记录](benchmarks_v31_cpu/README.md)。CPU 的 `aot_eager` 测试不证明 CUDA Inductor 的数值、显存或性能通过。

```bash
PYTHONPATH=source/reflex:tests python -m pytest -q tests/test_actor_diagnostics_v31.py
bash -n scripts/diagnose_actor_v31.sh
bash scripts/diagnose_actor_v31.sh --help
```

回滚：不使用三个新诊断 mode 即可；训练的记录范围默认关闭。V30 配置、历史结果与 checkpoint 均保留。
