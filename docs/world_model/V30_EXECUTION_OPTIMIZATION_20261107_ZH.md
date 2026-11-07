# V30：WM H8 与 Actor H32 执行优化（GPU 待验证）

## 目标和当前状态

目标是固定 Actor2048/H32、WM3×2048/H8/10次更新、FP32、原奖励/AMP/原生事件，将 WM+Actor/Critic 完整优化轮尽量压缩至1秒以内。真实采样、初次编译、checkpoint开销分别报告，不混入或省略。当前只完成本地实现和CPU验证，尚无V30实际GPU加速结果，也未启动训练。

基线 ca5ea0b；独立分支 experiment/v30-actor-execution，V29配置与既有结果不覆盖。所有新增执行选项默认关闭；候选配置后缀 trial，不授权正式训练。

## 第三方报告的有效结论与限制

已阅读服务器 /root/trackerlab-supermimic/BENCH_REPORT.md 及 bench_wm_actor.py。相同4090上，报告WM10次约133.4ms、Actor/Critic约267.2ms，合计400.6ms；Actor4096/H32。脚本真实执行 backward、裁剪和Adam，不能解释成只测forward。它未启动Isaac，使用合成动作和随机Replay，采样/端到端数字是估计，不能作为真实训练吞吐。

其MLP使用BF16、TF32开启；机器人状态模型输入133维、无箱子原生事件或Reflex的对抗式AMP奖励。报告中的AMP指自动混合精度。Actor网络并不比Reflex更小。下一次针对性对照应同时关闭BF16和TF32、Actor设2048，保留原始脚本/报告。仅传 --amp none 不足以关闭脚本硬编码的TF32。Reflex此前BF16梯度验收失败，本次仍以FP32为基线。

不能预先断言事件调度是主要瓶颈。主动的编译边界与运行时graph break是不同概念；需要Profiler证据。

## 依赖审计和实施

| 模块 | 必须递推的部分 | 当前候选 |
|---|---|---|
| WM H8 | 前一步Post-control进入下一步Physical；mask复位 | compile_wm_rollout：整个H8条件化递推、精确事件写入和物理损失编译；Replay采样/Adam/日志在图外 |
| Actor H32 | 动作、Physical、任务历史、事件计时/控制状态 | fuse_actor_core：单步观测/归一化/Actor/target/WM/Physical FK融合；H32仍顺序执行 |
| 原生事件 | 等待/发射/重投/推力、全局及其他环境时钟、RNG顺序 | compact_event_pool_updates：只消除冗余pool clone，不缩减时钟拓扑，不跳过无写入事件 |
| FK | Physical状态FK必须更新；机器人pose写入后必须更新 | reuse_post_event_fk：无机器人根姿态/关节位置写入时复用Physical pose FK；速度变化仍由现状态计算 |
| Critic | Actor需要H32末端及有效截断点的可微bootstrap；λ-target需要全部值 | sparse_actor_bootstrap：全轨迹Critic特征无梯度，Actor仅计算每轨迹截断/末端bootstrap；λ-target仍全量no_grad估值 |
| AMP | 历史特征逐步建立；判别器纯前向可批处理 | 保留已验证的batch_imagination_amp |
| Reward/Termination | hold/cradle/regrace等历史及奖励函数副作用影响后续步骤 | 保留原顺序packed heads；未把evaluate_step整个延迟到轨迹末尾 |

本轮没有直接编译H32大图。部分纯奖励可能仍可分离，但必须逐个证明它不写历史/缓存、不依赖奖励执行顺序；尚未将未经验证的奖励批处理加入正式路径。解析软终止公式、存活率、奖励权重完全保持V29。

### 代码入口

- trainer.py：执行开关、H8 rollout、Actor单步融合及pose复用。
- events_v24/execution_v30.py：可选的native scheduler执行适配；fast_view.py只增加asset factory挂钩。
- actor_bootstrap.py / joint.py：固定形状bootstrap gather，保持Actor梯度与Critic λ-target路径。
- imagination_batching.py：reference → FK复用 → pool更新 → sparse bootstrap → fused core逐项对照；可选reduce-overhead图试验。
- wm_rollout_benchmark.py：同一组固定Replay窗口、同一权重和Adam状态，比较单步编译与H8编译，保存实际梯度和更新差异。
- train.py：新增 actor_execution_v30_check、wm_rollout_v30_check。

原生回调算法与事件状态编码不变。pose复用使用“是否调用过机器人pose writer”的主机元数据，绝不读取CUDA mask.any来同步。即使相等常量pose写入或全false mask，也保守重算，避免错误改变梯度。

### Actor4096

新增 actor4096_trial 作为独立超参数消融，WM batch仍2048，真实环境数由 --num_envs 决定。Actor2048仍是等价加速基线。4096单批H32反向显存可能与512 Replay冲突，必须独立进程测试，不能根据采样4096就认定Actor4096可用。暂不自动降低Replay或改成microbatch来掩盖OOM。

## CPU 验证

新增测试覆盖：0/1/2个全局timer池、shared/private、其他环境时钟触发、事件RNG及输入不变；箱子写入、机器人速度/姿态/关节写入、常量/相等/all-false写入；H32发射、pose写入与截断；稀疏bootstrap的Return、λ-target、Actor梯度和Adam；aot_eager fullgraph单步融合前后轨迹/梯度；H8 ordinary/padding窗口的loss、梯度和Adam；旧checkpoint迁移和benchmark状态恢复。

CPU使用小模型与aot_eager验证计算语义，不代表Inductor/CUDA Graph可捕获、显存容量或GPU性能。全量151项通过（380.55秒，1条既有TorchScript弃用警告）；之后新增的checkpoint、WM基准恢复、Actor完整更新基准恢复各1项单独通过，共覆盖当前154项。日志及摘要见 benchmarks_v30_cpu_20261107/。新增模块与测试Ruff通过，修改模块语法编译通过；train.py/trainer.py全文件仍有既有lint告警，未为本次优化进行无关重排。

## 开卡后验收顺序

1. 独立worktree，确认SHA/实际导入路径、GPU/进程/磁盘；不覆盖V29checkpoint或正在运行任务。
2. 固定checkpoint/Replay/seed28032，Actor2048/H32，逐项执行 actor_execution_v30_check。比较Physical/Post-control、Reward/软Termination/AMP、事件状态、Actor梯度方向/范数和Actor/Critic实际Adam更新；默认dtype容差，不为提速放宽。
3. native事件探针必须实际覆盖等待到发射、重投、推力、悬停、截断；随机起点未触发事件不算覆盖通过。
4. wm_rollout_v30_check固定10组H8窗口，比较单步/H8编译、物理损失及参数更新。分别记录冷编译、稳态WM10、Actor前向/BPTT/完整更新、Critic、完整轮、采样吞吐及显存。
5. --actor_graph_trial只对单步fused core尝试reduce-overhead；Profiler检查cudaGraphLaunch实际次数，保留trace。出现pending backward/fallback/OOM即记录失败，不宣称Graph加速。
6. 每个配置独立进程测峰值显存；同进程多候选的缓存驻留不能用作容量结论。4096 Actor单独测，不与2048混算收益。
7. 只有正确性与事件覆盖通过，且稳态实测有收益的选项才进入正式训练。1秒是目标，不是已取得的结果。

### GPU 调用模板（尚未执行）

在确认镜像、环境和数据可用后，使用已有训练入口加载同一个V29解析305/compact checkpoint；正式基线配置仍用V29，Actor执行对照由check内部切换，不启动训练：

```bash
bash scripts/train_stage1_world_model.sh \
  --mode actor_execution_v30_check --num_envs 4096 \
  --config configs/world_model/joint_efficiency_v29_compact_analytic305_batching_512_trial.json \
  --checkpoint "$CHECKPOINT" --reinitialize_replay \
  --output "$NEW_ACTOR_BENCH_DIR"

bash scripts/train_stage1_world_model.sh \
  --mode wm_rollout_v30_check --num_envs 4096 \
  --config configs/world_model/joint_efficiency_v29_compact_analytic305_batching_512_trial.json \
  --checkpoint "$CHECKPOINT" --reinitialize_replay \
  --output "$NEW_WM_BENCH_DIR"
```

CUDA Graph候选在第一条命令追加 --actor_graph_trial，使用新的输出目录。CHECKPOINT及输出目录必须事先设为实际路径。需要512满Replay显存结论时另跑容量模式，40帧短对照不证明512容量。最终组合端到端短回归只有上述验证通过后执行。

4096 Actor配置不能直接加载2048批次配置的完整checkpoint并静默忽略差异；现有严格恢复检查会拒绝。其独立消融需明确加载权重/恢复优化器策略后再测，不能当作相同配置的等价对照。

## 首次 GPU 运行发现（仍在验证）

2026-11-07用户开卡后，独立服务器worktree /root/reflex-v30-test，572f747，4090，无其他GPU训练。数据盘仅剩约1GB，测试输出/编译缓存放系统盘 /tmp/reflex_v30_gpu_20261107，不删除原结果。

首次WM测试：单步编译10次更新稳态约0.32秒（AMP另约0.01秒），H8全段编译完成后，10次更新的首层权重相对单步编译最大差约2.80e-5，368/503808元素超默认容差，未通过。不能据此将H8融合用于训练。首次基准在异常处只保存了baseline耗时，不能推断H8候选的性能。

复查发现WM基准还应恢复fit_model内部增长的wm_rounds_completed和AMP状态；这不直接解释确定性WM参数差异，但会污染重复测量和副作用恢复。已补齐AMP/Adam/历史帧及轮次恢复，新增未融合baseline，并将所有候选耗时和失败详情保留，numeric_passed=false不等于通过。补充CPU恢复测试再次通过。Actor测试与WM候选独立进行，不因一项失败跳过其他项。
