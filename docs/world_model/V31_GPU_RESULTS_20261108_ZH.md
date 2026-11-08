# V31 Actor H32 瓶颈诊断（2026-11-08）

## 结论

本轮完成完整当前链路 Profiler 和两个最小计算对照，没有启动正式训练，也没有将新融合方式切入训练。**完整 Actor 的主要问题是任务递推中的大量碎算子、动态索引、状态携带及其反向调度；不是 Critic 或 Replay 采样，也没有证据支持“基础 Actor/WM 矩阵计算本身必然要 1.8 秒”。**

这不是某一个事件函数独占全部差距的证明。完整链路部分反向无法唯一归属，仍显式列出；Profiler 自身增加开销，不能将其中的模块 CPU/GPU work 数字直接当作无 Profiler 的独占 wall-clock 占比。

## 环境与实验合同

- RTX 4090 24GB，PyTorch 2.7.0+cu128，FP32、TF32 off、OMP=1。
- 独立 worktree `/root/reflex-v31-diagnostics`。完整链路源码 `40f3f80`；诊断统计/最小链路修正 `ceb15e1`。旧工作目录及未提交改动未覆盖。
- 源 checkpoint：`/tmp/reflex_v29_scratch_joint30_20261107/checkpoint_final.pt`，V29 预热500轮后联合30分钟的最终存档。
- 真实环境4096，Actor2048/H32，WM3×2048，现有 packed heads、FK复用、AMP特征/判别器批处理、Critic批处理、原生事件、解析软存活保持。
- 重新采集40向量步后冻结Replay。每次恢复相同权重、Adam及随机状态；20次无Profiler稳态更新，另捕获一次完整trace。恢复与复制快照在计时区间外。不是持续采集吞吐或完整 WM＋Actor 轮时。
- 完整链路与 Reflex 最小链路起点 SHA 相同：`eb7a406b5e58f997c8cc86c3b9a245de04c3258c0a70cf78e32cbbe6260737a8`。

## 完整链路的稳态结果

| 指标 | 中位数 | P90 |
|---|---:|---:|
| 完整 Actor＋Critic 调用 | 1.7539s | 2.5160s |
| Actor（含采样、想象、BPTT、目标构造、裁剪/Adam） | 1.6958s | 2.4604s |
| Critic 更新 | 0.04644s | 0.04819s |
| Replay 起点采样，冷索引缓存 | 2.474ms | 2.974ms |
| Replay 起点采样，热索引缓存 | 2.167ms | 2.332ms |

完整调用包含 Actor/Critic 阶段外的统计，不能要求两个子项中位数之和与完整中位数精确相等。20次完整调用范围1.7210～2.5559秒，其中两次约2.5秒；目前没有证据将该尾延迟归因于GC或具体系统调度，不能先当成确定结论。

首次完整更新含编译69.15秒，第二次1.751秒。Torch峰值 allocated15.57GiB、reserved16.24GiB；Isaac/PhysX等额外分配不在Torch计数内。原始遥测见 `benchmarks_v31_gpu_20261108/gpu.csv`，覆盖首轮driver，未覆盖后续R2最小链路。

记录范围前后：完整想象输出、Actor梯度、实际Actor/Critic参数与梯度 **max_abs=0**；捕获期间新增图、frames、graph breaks均为0。实际覆盖44次等待→发射、2次返回等待、501次截断。这些不是接箱成功次数，也不能证明所有碰撞或重投子类型已覆盖。

## Profiler 的直接证据

从原始Chrome trace只统计 `kernel`、`gpu_memcpy`、`gpu_memset`：

| 项目 | 实测 |
|---|---:|
| CUDA kernel | 97,596次 |
| ≤5µs kernel | 94,572次，96.9% |
| kernel时长和 | 277.59ms |
| kernel＋拷贝/清零活动区间并集 | 290.34ms |
| CPU更新范围（Profiler内） | 3,164.48ms |
| 该范围内未观测设备活动间隙 | 2,874.14ms |
| `cudaLaunchKernel`＋`cuLaunchKernel` | 71,469＋26,127次 |
| `cudaStreamSynchronize` | 1,608次 |
| `aten::nonzero` / `aten::item` | 758 / 689次 |
| Device→Host 拷贝 | 1,386次，共5,050字节 |
| Device→Device 拷贝 | 7,892次，共1.86GB（十进制） |
| `cudaGraphLaunch` | 0次 |

小量标量频繁回CPU、海量短kernel和碎反向节点，支持调度/索引链路是重点。不能把290ms简单从无Profiler的1.754s中相减后，称差值是某模块的可消除时间；Profiler将CPU范围放大到3.164s。活动区间并集也不是SM利用率，不能报告成“GPU只有9%利用率”。

**统计修正**：初版 `actor_full_v31.json` 的 `attribution.device_timeline` 混入 `gpu_user_annotation`，错误地把用户范围当设备活动。该字段已明确废弃，以同一原始trace重算的 `actor_full_trace_summary.json` 为准；完整无Profiler计时、梯度检查及直接关联work表不受影响。源码已添加排除GPU用户标记与合成trace回归测试。

下面是按最近范围和反向序号归属的关联设备工作，不是模块独占wall时间；计数包含被CPU事件关联的设备活动：

| 范围 | 关联设备活动数 | GPU work时长和 |
|---|---:|---:|
| native events前向 | 22,002 | 38.85ms |
| native events反向 | 17,268 | 26.45ms |
| 事件输入gather前向 | 4,866 | 21.64ms |
| pose/state carry前向 | 2,784 | 6.63ms |
| pose/state carry反向 | 10,277 | 13.61ms |
| packed Reward/Termination前向 | 4,504 | 7.16ms |
| packed Reward/Termination反向 | 7,316 | 11.42ms |
| Physical WM前向 | 1,774 | 31.20ms |
| FK前向 | 4,021 | 5.15ms |
| FK反向 | 4,449 | 6.50ms |
| 无法唯一归属的反向 | 4,968 | 34.11ms |

FK实际调用34次，说明现有FK复用已经生效，不应重新把“每步两次全身FK”当现状。CUDA Graph启动0不等于没有 `torch.compile`；本配置有编译单元，只是本次完整trace未执行Graph。旧V30曾有Graph实验数值失败、冷编译过高或无收益，不能仅把开关打开就宣布解决。

## 最小 Actor＋Physical WM 对照

全部2048/H32、FP32/TF32 off、相同计时框架。每个模型内部三组固定权重、起点、损失、初始化Adam；真实BPTT，WM冻结但不断梯度。数值检查维持默认容差，包含最终物理损失对Actor叶参数的梯度。

| 模型 / 编译方式 | 前向中位 | BPTT中位 | 完整中位（含裁剪/Adam） | 严格数值检查 |
|---|---:|---:|---:|---|
| Reflex eager | 157.19ms | 302.19ms | 462.59ms | 对照 |
| Reflex仅WM单步编译 | 50.86ms | 53.60ms | 105.29ms | 未通过 |
| Reflex Actor＋WM单步编译 | 37.36ms | 38.76ms | 77.08ms | 未通过 |
| TrackerLab原生 eager | 76.44ms | 104.48ms | 182.95ms | 对照 |
| TrackerLab原生仅WM编译 | 71.25ms | 93.41ms | 166.32ms | 通过 |
| TrackerLab原生Actor＋WM单步编译 | 69.20ms | 91.57ms | 162.44ms | 通过 |

子项中位数不保证精确相加。Reflex最小链路峰值allocated约2.32～2.38GiB；TrackerLab约6.20～7.64GiB。R2 warmup命中已有Inductor磁盘缓存：Reflex首样本0.94/3.52/1.66秒、TrackerLab0.40/10.75/17.66秒（按表内顺序）。不是全新缓存的冷编译成本。

**Reflex新编译边界未通过，未启用到训练**：

- H32状态最大误差：仅WM编译1.83e-4、Actor＋WM编译1.10e-4。
- Adam后Actor首层1个权重超差，最大差2.17e-5 / 1.66e-5。
- 动作最大差7.15e-7 / 9.54e-7；物理终点对Actor梯度最大差约1.0e-6，物理梯度范数均约2.6501。
- 不因为超差元素少就放宽容差。此处是新计算边界与eager的诊断差异，不等于本轮修改了正式训练行为。

最小链路不含真实任务FK/奖励/AMP/事件，使用固定诊断观测投影和合成目标。不能把0.077秒当完整Reflex Actor，也不能据此声称20倍等价加速。它支持“基础网络/积分存在较快执行路径”的判断；正确性阻碍仍需处理。

TrackerLab输入/权重为明确标注的合成对照，不是其完整训练复现。原生Actor约911.65万参数，Reflex约26.01万；WM分别约882.29万 / 906.66万参数。原生最小链路通过，也说明不能单用“Reflex网络更大”解释差距。两者状态和历史结构不同，原始时长不可作为同模型因果消融。

服务器第三方目录没有Git元数据，初版适配器在读取SHA时失败；修正后如实记录revision为空与实际源码SHA，原生网络未改。初版对“编译函数返回动作”的VJP存在并列输出边界问题，也已改为对Actor叶参数求物理梯度；训练反向链没有改动。

## 下一步优先级

1. **精简事件外层输入gather**：按当前minimal view与回调真实依赖只索引必需字段，避免把完整Replay字段及全部pose逐步gather。先验证全子集/部分子集、外部计时池、随机数消费和发射位置梯度，不能跳过无写入但更新计时器的事件。
2. **压缩pose/state carry反向图**：考虑内部packed pose一次mask，外部保持现有接口；避免逐body `where` 生成海量小反向节点。不能改变被截断轨迹冻结方式。
3. **在上述固定形状边界尝试分段融合/Graph**：独立核验状态、事件、奖励、梯度和Adam。结合packed heads热点继续分离纯计算与历史依赖；不编译整个H32大图。
4. **处理编译数值差异及尾延迟**：定位最小链路H32漂移与Adam敏感元素，不改容差掩盖；GC/系统调度需单独取证。

当前Replay采样约2ms、Critic约46ms，AMP已批处理，均不是第一优先级。没有本轮等价加速收益可合入，也没有整轮低于1秒的结果；不能从最小链路时长预估正式训练成功率。

## 验证与存档

- `40f3f80`：本地完整CPU173 passed、1条原有弃用警告。
- GPU诊断修正：专项CPU7 passed（67.57秒）；新增GPU标记/非Git目录覆盖后相关2 passed（2.42秒），Ruff及shell语法通过。
- 完整GPU链路：数值与参数更新一致；第三方原生最小链路：三组一致；Reflex新编译边界：未通过，原始失败明细保留。
- 完整trace约297MB，留服务器 `/tmp/reflex_v31_gpu_20261108/full/actor_full_trace.json`；小型报告归档 [原始数据](benchmarks_v31_gpu_20261108/README.md)。
- 诊断结束已确认无训练/诊断进程、GPU占用约92MiB且利用率0%。未关机，未启动半小时训练，未覆盖任何checkpoint。
