# Return-based reset 与分阶段 UTD 实验（3584旧机容量验证准备中）

用户已明确选择 TrackerLab 原用途：改变真实环境参考轨迹重置起点，不改变 WM Replay 采样。
此前未启用的最近64步/全量混采实现已撤回。实验与最初“UTD＋新鲜度”不同，不能沿用原实验归因。

## 当前实现

- `return_reset.py`：50帧一个bin；softmax(-25 * bin_return / max_return)；50%均匀、40%当前softmax、10%历史权重。历史权重每次采样按0.999/0.001更新，回报按0.8/0.2更新。
- 实际奖励计算之后、原生reset之前累计真实奖励与折扣回报。真实终止无bootstrap；timeout/参考轨迹重新采样按平均奖励bootstrap。只替换 `_adaptive_sampling`，原 `_resample_command` 的机器人写入和外部事件流程继续执行。
- 首次采样不使用不存在的回报；参考轨迹结束的重新采样会关闭当前回报段，不与下一参考段混计。
- checkpoint保存bin回报和历史权重；不恢复旧模拟器的未完成episode累计量。启用新模式时禁止没有对应状态的checkpoint静默恢复。
- `joint_updates_per_env_step` 可独立配置为0.1，WM500轮预热仍按原0.4；切换时记录新计数锚点并保存恢复。
- Replay/Actor起点采样均保持原样。旧配置默认不启用新reset模式。

## 回报归一化

TrackerLab 的 `max_step_reward/(1-gamma)` 不能直接套用其跟踪任务常数。当前stage1原奖励配置正权重之和为65.7，控制dt=0.02，本实验固定采用1.314作为每步正奖励包络尺度，`max_return=131.4`（gamma=0.99）。该尺度是用于采样权重的保守归一化，不能解释为实际能同时取得的最大奖励，也不裁剪/修改真实奖励。`ball_lateral_offset_penalty`等虽为正权重但奖励可能非正，包络仍保守计入。启动时按实际RewardManager的正权重重新求和，与配置不匹配立即报错，不静默沿用旧任务尺度。GPU阶段仍需检查实际bin_return、采样概率是否过度集中或近乎均匀。

## 本地验证与GPU入口

CPU相关测试覆盖尾bin、重复bin回报汇总、终止/非终止bootstrap、原reward输出不变、参考段清零、非法历史权重事务性拒绝、Trainer实际save/load、UTD切换及恢复。默认配置仍使用旧采样，Replay代码没有改动。

GPU诊断（当前未运行）：

```bash
bash scripts/check_return_reset.sh /root/gpufree-data/reflex_return_reset_check_NEW
```

诊断使用独立输出、4096环境/H16/双batch4096；采集320步，显式测试原生reset与参考耗尽，通过已有event_condition_probe检查原生事件写入和等待到发射梯度，再做10次WM更新与一次Actor/Critic更新。诊断明确设置已完成预热计数以检查更新链路，结果丢弃，不算正式训练。完整通过必须有`return_reset_check.json`和`return_reset_events.json`，不能只看进程退出码。

GPU诊断通过后，从零总30分钟入口（当前未运行）：

```bash
bash scripts/train_return_reset_30min.sh PASSED_GPU_CHECK_DIR NEW_TRAIN_OUTPUT_DIR
```

使用独立`joint_return_reset_h16_batch4096.json`，无checkpoint参数；WM500轮预热UTD0.4，其后UTD0.1，每轮WM10次、Actor1次。Replay保持均匀采样，Actor起点采样保持原样。`--max_seconds1800`从训练循环开始；不使用`--joint_max_seconds`，所以不会额外再训练30分钟。最终阶段记录包含总时间、首次Actor开始时间、Actor实际可用时长、预热完成时间、真实转移和更新数。计时不包含Isaac启动和最终存档；允许正在执行的一次更新完成后越过截止点。

## 剩余验收

- GPU真实事件/梯度/数值与显存检查尚未执行，CPU检查不能代替。
- 启用新reset模式需对应的采样器checkpoint状态；旧checkpoint不可静默当成新模式恢复。物理环境状态不恢复，因此只恢复bin回报/历史权重，不接续旧环境未完成回合统计。
- 正式30分钟训练、独立PhysX评测、同时间/同真实样本数历史对照尚未执行。
- 本实验现在改变的是UTD与真实reset起点分布，不能将其结果归因为Replay新鲜度改进。

参考源码：https://github.com/Renforce-Dynamics/trackerlab-supermimic/blob/main/source/trackerLab/trackerLab/managers/supermimic.py

## 2026-11-09 GPU实际结果与启动记录

机器RTX4090，Torch2.7.0+cu128、IsaacSim5.1/IsaacLab2.3.1，4096环境、WM/Actor batch4096、H16。原始JSON在`benchmarks_return_reset_gpu_20261109/`。

- reset/参考耗尽检查、原生事件对齐、发射梯度、WM10次更新、Actor/Critic实际参数更新均通过，无OOM。
- 采集320个向量步（1,310,720条），诊断中真实终止31,951次；显式原生reset8环境，参考耗尽重采样8环境。
- 原生事件24步对照通过，控制测试覆盖12,150次扰动写入；H16等待→发射4,095条；发射速度对机器人位置梯度范数66.45958。
- Actor梯度范数0.01966776，实际参数发生变化。Torch峰值allocated约16.05GiB、reserved约16.65GiB；不等于进程总显存。
- 首次WM10次更新15.785秒，首次Actor更新81.222秒，含冷编译；**不可作为稳态性能**。随后正式训练日志WM10次约0.668秒。
- CPU相关最后一组25项通过；另有较广41项测试通过，存在重叠，不相加计数。

旧训练已按用户要求停止（停于WM314轮、尚无Actor更新）：`/root/gpufree-data/reflex_return_reset_h16_b4096_total30_20261109_r1`，PID1595（启动时）。从零、无旧权重。总预算1800秒从训练循环起算。TensorBoard为该目录下`tensorboard/`。

旧基线已截取训练循环前1800秒：10,285,056条真实转移、988轮WM（9880梯度更新）、488次Actor更新；原完整旧实验实际2519.92秒，不作为等时对照。详见`return_reset_baseline_30min_20261109.json`。

训练后自动评测等待任务已启动，确认训练完整存档且退出后，对最终checkpoint生成单环境seed42、15秒无坐标轴视频，目录为训练目录加`_play15s`；这只是短视频检查，不等于完整成功率评估。任务状态在训练目录同级`.post_eval.json`。正式训练结果尚未产生，不能宣称站立/接箱改善。

## 用户要求开启WM优化后，从零重启：双batch3840

用户明确更正为“两者从4096减小256”，即WM/Actor/microbatch均3840，不是256。真实环境仍4096、Replay512、WM H8、Actor H16、网络及奖励/事件规则保持原值。新配置`joint_return_reset_h16_batch3840_wm_graph.json`使用stable_heading_v1、H8 rollout与training step编译，WM独立reduce-overhead。

4096 Graph补测OOM调用栈确认为Actor backward：当时进程23.26GiB、Torch allocated18.27GiB、Graph private pool1.78GiB，额外136MiB无法满足。可扩展分配策略后4096连续3轮参数更新通过，但其中一次CUPTI初始化失败，CUDA活动缺失，该次Graph计数0不能解释为未执行Graph。3840新验收无OOM，Profiler实际确认20次Graph调用，稳态无新增compile/frame/graph-break。

3840同权重/Replay五轮配对中位数（不含真实采集）：

| 项目 | 未融合H8 | H8 Graph |
|---|---:|---:|
| WM10+AMP | 0.692144秒 | 0.464932秒 |
| Actor+Critic | 0.895070秒 | 0.877184秒 |
| 合计 | 1.585366秒 | 1.339419秒 |

实际参数/梯度比较与连续3轮WM/Actor/Critic/AMP参数/梯度比较均通过原容差。**严格逐元素轨迹比较未全通过**：critic_obs最大3.8743e-5、AMP style最大3.9160e-4；物理状态最大4.2915e-6、reward最大1.3039e-6、Actor梯度最大1.8626e-9。事件/发射/有效/截断标记一致。没有放宽原比较容差，原JSON的numeric_passed保持false。按用户此前允许小数值误差做实际训练效果验证及本次开启重训要求，启用独立受控试训，不晋升默认配置；`check_graph_trial.py --allow-feature-trial`显式记录严格失败与连续更新通过的区别，拒绝其他轨迹失败和优化器更新失败。原始证据`benchmarks_return_reset_graph_20261109/b3840_paired_continuous.json`。

本次3840冷编译约15分钟，原900秒诊断预算在即将完成时由有界看护最多延长600秒，实际顺利结束；没有将冷编译时间计入配对稳态结果。旧4096隔离缓存测试被中断后复用现有缓存，保留其未完成状态，不算通过。

新训练目录：`/root/gpufree-data/reflex_return_reset_graph_h16_b3840_total30_20261109_r1`，启动worker PID43739。无checkpoint参数，从零训练，总1800秒仍包括500轮WM预热。启动后实际日志WM10次约0.452–0.456秒，`execution/wm_cuda_graph_replays=20`，确认生产路径实际使用Graph，非仅打开开关。GraphReplayMonitor只计数公开CUDAGraph.replay调用，不改输入、算子、返回值和梯度，不依赖CUPTI。

新TensorBoard：上述目录下`tensorboard/`。训练后评测等待任务已重新绑定新训练，不再等待被停止的旧训练。最终效果仍待训练完成后独立评测；本次同时改变UTD、reset分布、batch及WM执行/编码，不能将效果差异全部归因某一个因素。

## 后续故障与当前状态（2026-11-09，31428旧机）

以上“已启动”仅为历史记录，不能作为当前运行状态。3840 Graph正式任务在第二次Actor的Critic全轨迹估值ELU分配128MiB时OOM。已增加Critic分块4096行＋非重入激活重算，保持有效Actor batch3840/H16及全部输入梯度；CPU完整Actor Adam/Critic更新对照通过。真实checkpoint Critic GPU输入形状17×3840×305，输出/输入梯度/Adam通过原容差，独立峰值580625920→337306112bytes；此单次计时含不同预热，不用它宣称速度提升。

低显存恢复r1完成35次Actor更新后因fall_terminated解析一致性检查停止，非OOM；校验修复前的低显存恢复r2随后又在反向申请120MiB时OOM，说明3840仍缺乏显存余量。用户建议再次减256，下一配置为WM/Actor/microbatch3584，真实4096环境不变，不曾启动256 batch任务。3584尚未GPU容量验收，不能声称稳定。

奖励校验已区分两个问题：
1. 紧凑状态FK与PhysX位姿保持原2e-4/1e-5连续误差门限，仍严格断言，不修改物理模型。
2. 原奖励/终止函数在**同样原生几何**输入的独立诊断facade中严格比较；与紧凑FK输入计算的原始奖励/终止差异单独记录，不再将输入的微小差异误报为公式不同。真实环境奖励、阈值、终止、重置规则和想象函数均未改。诊断facade仅存在于真实Adapter校验局部，不能进入Replay/WM/Actor输入；requires_grad的学习输入显式拒绝诊断facade。

GPU4096环境×400步复验通过（1638400条新增真实转移）：全体FK位置最大差3.5524e-5，原公式同几何奖励最大差7.4506e-8；终止标记一致。这次原始compact termination disagreement为0，**未直接复现此前fall_terminated现场**。独立捕获low_hold：真实箱/躯干距离0.600007951m、compact FK距离0.599999487m，跨越原0.6m门槛，原始加权奖励差0.024；该差异保留在证据中，不是被放宽奖励容差消除。证据在benchmarks_parity_geometry_20261109/。

CPU最近相关30项通过，另有9项几何诊断/编译/批处理回归通过，存在重叠不相加。新增显式批次迁移工具仅改checkpoint的三个batch配置，保留全部权重、Adam、normalizer、学习契约和计数，另写迁移元数据并拒绝覆盖原件。

用户改用183.147.142.40:31428旧机；也是RTX4090 24GB，不能把OOM消失归因于换机器。迁移前旧机3处未提交修改已patch+tar备份并stash，数据/旧checkpoint不覆盖。本轮最新可恢复存档为30340机lowmem_resume_r1/checkpoint_00200.pt（Actor16、WM516），SHA256为9abd50727630eee2ff77be536aced854396e0afed33f1d855c31bc44cd52c379。先做3584满Replay512容量验证，验证模型更新不留存，完成后才考虑继续剩余训练预算；不以warmup之后重新计30分钟。
