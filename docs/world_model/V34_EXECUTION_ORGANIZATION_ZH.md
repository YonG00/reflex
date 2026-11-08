# V34：融合边界定位与连续事件写入候选

状态：GPU短测完成。原生事件及完整更新数值通过，事件写回候选收益约1.85%，暂不提升；最小链融合仍数值失败，未训练。详见 [GPU结果](V34_GPU_RESULTS_20261108_ZH.md)。基线为V32，两个研究方向同时推进。所有网络规模、FP32、Actor2048/H32、WM2048/H8、原生回调、随机顺序、奖励、软存活和梯度链保持。

## 已有证据与边界

V33的无Profiler连续20次更新：基线Actor＋Critic中位1.5630秒；记录器1.5662秒。前向0.8314秒、反向0.6607秒、Critic0.0450秒；原生事件32步执行区间合计中位0.3796秒，包含等待，并非可全部消除的GPU工作时间。两次gen2 GC分别0.6374和0.6433秒，解释尾延迟，不解释通常的全部耗时。记录前后参数及梯度最大差0，稳态新增图/graph break为0。原始证据见 `benchmarks_v33_gpu_20261108/native_phases.json`。

约81310个Kernel及1608次stream同步是Profiler记录，不可把调用数直接换算为可节省的wall-clock。同步调用CPU self合计约12.82ms，不是全部同步对运行时影响的因果估计。

## 方向一：为什么单步融合还没有通过

Reflex已经提供`actor_physical_core`（观测/归一化/Actor/动作目标/Physical WM/可选FK）；并非没有单步融合入口。奖励heads与FK分开，是已有Torch2.7四元数反向代码膨胀的工程边界；事件回调、计时器和随机采样仍在图外。V30全core与网络core候选有轨迹超差且没有收益，不直接再次启用。

同一V31 capsule的最小H32+BPTT控制：eager约0.4626秒，WM分段编译0.1053秒、单步融合0.0771秒，但后两者严格数值不通过。分段/单步的最大状态差分别1.83e-4/1.10e-4、Actor首层参数差2.17e-5/1.66e-5。梯度差小并不能证明实际Adam更新一致；这些速度不算已验收成果。最小链不含原任务完整FK/奖励/事件，不能推断完整Actor同等收益。

本轮新增`diagnostics_v31.minimal.boundary_probe`：在基线每步的相同state/action-history/control-history输入上比较候选单步输出，报告每步state/action/aux误差和首次不通过位置。计时区间外运行；原H32递推、Actor物理梯度和实际Adam检查继续保留。它用于区分局部编译误差与滚动放大；当前没有GPU证据证明FMA或某个算子就是根因。

下一次先运行同capsule三边界诊断，必要时才追加单个算子或禁用FMA的独立试验。没有给训练改精度、编码、容差或编译环境。CUDA Graph不能修复数值差异，不在通过前强行加入整个H32。

## 方向二：减少事件输入装配及写回节点

`ContiguousWritePlan`为独立候选，普通配置默认关闭：

- 原生完整Batch根姿态/根速度写入，用连续列切片替代二维高级索引，省去arange/slot Tensor和部分索引反向。
- 部分环境、Tensor slots、不连续slots、重排索引保持原record实现，不猜CUDA索引内容，不做额外nonzero或同步。
- 子plan提取只gather一次并clone，不再构造零初始化临时plan。仍有独立可变存储及相同梯度来源。
- `before`从未参与WritePlan的输出计算；候选不构造无用途的旧姿态/速度gather。基线保留原路径。
- Scheduler仅增加plan factory扩展点；全部原生回调、触发和随机采样保留。没有省略无写入回调，也没有改变PhysX规则。

这只是执行结构候选，未声称大幅突破。切片写回仍有CopySlices节点，是否优于原索引必须实测前向、BPTT和完整Adam。没有将整个事件回调编译为大图。

## CPU验证

最终候选专项21通过（10.98秒），V31/当时V34诊断专项23通过（107.24秒）；旧checkpoint训练—保存—恢复—H32事件推进全链1通过（142.81秒）。相关回归初跑109通过、2失败，分别发现tuple状态探针兼容和源码哈希迁移问题，均修复并独立复测通过；未把初跑结果改写成全套通过。日志见`benchmarks_v34_cpu/README.md`。

新事件源码只通过显式`--migrate_event_execution`接受已固定的历史runtime/writes哈希，其他callback/数学变化仍拒绝。迁移不改权重、Adam或Replay数值；GPU诊断脚本显式传该参数。该候选尚未经过GPU验收，不能用于正式训练。

## GPU入口与接受标准

```bash
bash scripts/study_actor_v34.sh CHECKPOINT NEW_OUTPUT_DIR
```

先64真实环境运行现有`event_condition_probe`，严格比较真实PhysX与候选调度/写回，包含强制事件前缀、扰动正例、依赖预测机器人位置的发射梯度、H32等待→发射。再4096环境、2048/H32进行交替20轮真实Actor/Critic更新，基线与候选相同Replay/权重/Adam/RNG，比较完整轨迹、raw梯度、实际参数更新及峰值显存，冷编译单列。诊断退出恢复状态，不写新checkpoint。

分别确认发射、返回等待及截断覆盖，未触发的场景不得标记通过。需要进一步用相同起点的完整Profiler比较Kernel/同步/反向来源；CPU aot_eager并不证明CUDA编译通过。原生probe与配对性能均通过仍不自动替换训练路径。

单步融合误差使用原V31 capsule CLI，结果新增`boundary_probe`。该测试与完整任务性能分开记录，不将最小链耗时当完整更新。无收益或不等价的候选不提升为正式配置。

## 4096 batch与显存

当前精简Context、Replay512、4090的通过记录为WM2048＋Actor2048/H32。V30文档明确Actor4096候选没有实测，早期全局4096采用2048 microbatch并不能证明单批4096能放下。WM4096＋Actor4096没有当前组合的通过记录。

Replay768在Actor2048时已有OOM；Replay容量、PhysX、编译缓存和BPTT同时影响整卡峰值，不能根据4096真实环境数推断batch4096安全。后续若需要，固定Replay和其他设置，在独立进程依次测WM4096、Actor4096及组合，记录OOM，不自动降低容量或改microbatch掩盖失败。

回滚：不选择V34候选配置，`contiguous_event_writes=false`，继续V32。新增执行flag属于checkpoint加载的显式可切换执行选项；模型和优化器权重形状不改变。

## 同capsule融合误差测试命令

完成原生/配对检查后，使用先前同一份最小链capsule（不重新随机替换起点）：

```bash
PYTHONPATH=source/reflex /opt/conda/envs/isaaclab/bin/python -m reflex.world_model.diagnostics_v31.minimal \
  --capsule /tmp/reflex_v31_gpu_20261108/full/minimal_capsule.pt \
  --device cuda:0 --batch 2048 --horizon 32 --warmups 2 --repeats 10 \
  --drivers eager segmented single_step --output /tmp/新的V34融合定位目录
```

该CLI的完成不代表数值通过，需查看`numeric_passed`、各driver的`numeric`和`boundary_probe`。逐步检查在性能计时外，会额外占用GPU时间；首次编译与稳态分开解释。没有默认排队No-FMA或整个H32编译实验。
