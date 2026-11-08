# V32 GPU验收：Actor执行耗时降低约10%

## 结论

两个执行优化在同一Replay、checkpoint、随机状态下通过GPU验收。完整Actor＋Critic中位耗时 **1.8246→1.6424秒，降低9.99%**，更新吞吐提升约11.1%。这不是整个WM＋Actor训练轮次计时，整轮1秒目标尚未达到。没有启动正式训练或覆盖checkpoint。

Profiler确认变化来自目标链路：kernel数量97,596→81,310（−16.7%），事件外层gather和pose carry反向明显减少。原生事件本体、1608次stream同步及1.86GB设备内拷贝基本未变，下一步应审计事件视图/控制字段拷贝和同步来源。

## 固定条件

- RTX4090 24564MiB，Torch2.7.0+cu128，FP32、TF32 off、OMP=1。
- 独立worktree `/root/reflex-v32-test`，实现 `d9e3a3b`，原目录未覆盖。
- checkpoint：`/tmp/reflex_v29_scratch_joint30_20261107/checkpoint_final.pt`。
- 4096真实环境、Actor2048/H32、WM3×2048/H8；固定V30 verified配置，奖励、AMP、软存活、事件及动作机制不变。
- 真实采集40向量步后冻结Replay，四组各预热2次；每轮轮换顺序交错测20轮，共80次完整更新。
- 每次恢复相同Actor/Critic、已有Adam、归一化和RNG；恢复/hash/CPU快照不计时，真实Replay sampler保留。

## 四组配对结果

| 开关 | 完整更新median | P90 | 相对基线耗时下降 | Actor median | Critic median |
|---|---:|---:|---:|---:|---:|
| reference | 1.82464s | 1.86344s | — | 1.76623s | 0.04730s |
| 只精简事件外层输入 | 1.75920s | 2.85124s | 3.59% | 1.70104s | 0.04688s |
| 只packed pose carry | 1.71811s | 2.78978s | 5.84% | 1.66063s | 0.04683s |
| 两项合用 | **1.64241s** | **1.70411s** | **9.99%** | **1.58570s** | **0.04614s** |

中位数子项不保证精确相加。组合20轮均快于同轮基线，逐轮完整耗时节省约0.126～0.217秒。单开两组存在约2.8～2.9秒尾延迟，未专门做GC/OS取证，不将其原因当成已知事实；组合组范围1.6076～1.7068秒，基线1.7727～1.8654秒。

必须用此同进程配对数据计算收益。后续独立组合Profiler进程另测5次，中位1.5695秒，样本和进程环境不同于本次20轮调度，**不采用该更小值替代1.6424秒或扩大加速比**。

Torch allocated峰值：reference16.213GiB、事件输入16.214GiB、pose carry16.257GiB、组合16.259GiB；组合约多47.6MiB。四组都预热后reserved16.930GiB。包含同一进程的多布局编译缓存，不等于单配置孤立显存；不包含PhysX等非Torch分配。没有显存溢出。

本轮顺序预热：基线首次72.97秒，首次新pose布局58.27秒，组合首次1.69秒。后者已复用前组编译缓存，不能当“组合全新冷启动只需1.69秒”。稳态新增图、frames、graph breaks全部0。

## 数值和事件验收

三组优化对基线：

- 完整想象输出（Physical、Post-control、事件WritePlan/控制状态、参考阶段、观测、奖励、AMP、软终止、有效/截断）：`max_abs=0`。
- 原始Actor梯度：`max_abs=0`，有限且非零。
- 实际Actor/Critic更新后参数及梯度：`max_abs=0`。
- 同一个真实Replay采样起点SHA，结束后模型未变、Actor/Critic/Adam及RNG恢复；不保存训练checkpoint。

覆盖44次等待→发射、2次返回等待、501次截断。不是接箱成功数，也不代表所有真实碰撞和重投子类型均覆盖。原生回调源码没改，CPU另外覆盖机器人速度/关节/常量姿态写入、其他来源时钟、预测机器人位置相关发射梯度。

## 完整Profiler核对

仅额外捕获胜出组合。与V31旧基线trace相比：**起点、Actor、WM和RNG的SHA全部相同**；V32基线False分支未增加GPU算子，因此可核对目标设备活动变化。时间收益仍以上述同进程四组配对为准。

| 指标 | V31基线 | V32组合 |
|---|---:|---:|
| 真实kernel数 | 97,596 | **81,310** |
| ≤5µs kernel | 94,572 | 78,266 |
| kernel时长和 | 277.59ms | 244.03ms |
| kernel/拷贝/清零活动区间并集 | 290.34ms | 256.71ms |
| `cudaLaunchKernel` | 71,469 | 55,183 |
| `cuLaunchKernel` | 26,127 | 26,127 |
| `cudaStreamSynchronize` | 1,608 | 1,608 |
| `cudaGraphLaunch` | 0 | 0 |

下面为Profiler直接关联设备活动（包括被关联的拷贝），**不是独占wall-clock比例**：

| 范围 | 活动数：前→后 | GPU work：前→后 |
|---|---:|---:|
| 事件外层gather | 4,866→**320** | 21.64→2.85ms |
| pose/state carry前向 | 2,784→**416** | 6.63→2.22ms |
| pose/state carry反向 | 10,277→**775** | 13.61→1.71ms |
| 原生事件前向 | 22,002→22,006 | 38.85→38.97ms |
| 原生事件反向 | 17,268→17,268 | 26.45→26.64ms |

记录范围前后轨迹、Actor梯度、实际Actor/Critic更新最大差0，捕获期间无重编译。组合trace约248MB，留服务器 `/tmp/reflex_v32_gpu_20261108/combined_profile/actor_full_trace.json`。活动并集不等于SM利用率，Profiler会增加CPU开销，不能把GPU work直接从无Profiler wall时间相减来分配热点占比。

设备内拷贝仍7,892次、1.86GB；设备→主机仍1,386次、5,050字节。本轮精简的是gather kernel和pose反向节点，不能声称同步或控制状态拷贝问题已经解决。本轮没有CUDA Graph加速收益。

## 使用与后续

新增显式试验配置 `configs/world_model/joint_execution_v32_verified_actor_trial.json`，仅较V30 verified配置多启用两个执行开关。默认TrainConfig及原有训练配置保持原状；没有自动训练或合并main。此配置通过的是执行数值与性能验收，不是新的站立/接箱成功率验收。

后续优先：

1. 按实际回调列出控制字段的读取/写入，检查能否减少每次视图构造的全量clone、重复编解码；不能跳过无写入但会更新状态的回调。
2. 定位 `nonzero`、标量回CPU和同步的具体调用点，再做保留RNG消费顺序、来源时钟与发射梯度的固定形状/纯张量优化。
3. 只有剩余热点适合时才扩大局部融合或Graph；不直接编译H32大图，不启用V31未通过的融合版本。

本轮基础代码完整CPU209项通过，GPU四组全部通过；新增配置经TrainConfig解析和差异校验，仅两个执行字段不同。报告/小型原始JSON见 [数据目录](benchmarks_v32_gpu_20261108/README.md)。GPU任务已结束，未启动正式训练。
