# V28：精简 WM Context 与 Actor 分段批处理（本地候选）

## 验收边界

本次先完成本地实现和 CPU 回归。没有本版本的 GPU 耗时、显存或训练效果结论；候选未接入正式训练配置。GPU 开启后独立短对照，收益与数值正确性均通过才考虑启用。原 V24/V27 配置与存档保留。

## 依赖审计：为什么不能把 H32 全部摊平

参考的具体实现：

- [supermimic_vibed policy_update](https://github.com/zita-ch/supermimic_vibed/blob/main/supermimic/train.py)：顺序 ImagStep，再 flatten 时间/环境维，批量计算 reward/soft termination。
- [TrackerLab runner](https://github.com/Renforce-Dynamics/trackerlab-supermimic/blob/main/source/trackerLab/trackerLab/algorithms/supermimic/runner.py)：顺序状态推进，随后处理轨迹目标。

Reflex 的差异必须保留：

| 环节 | 依赖 | 本轮处理 |
| --- | --- | --- |
| Actor、动作映射、Physical WM | 下一状态依赖当前动作 | 逐步推进 |
| 原生事件、确定性写入 | 依赖预测位置、计时器、历史；改变下一步 | 逐步执行，不跳空写入 |
| 解析奖励 | strict hold、threepoint 等更新计数器，历史会进入下一次事件 | 保持原顺序，暂不整体延后 |
| Termination | 概率达到 1、截断及硬边界开关影响 valid 和事件执行 | 保持逐步执行与软存活逻辑 |
| FK | Physical 与 Post-control 可能不同；初始化事件还可能写机器人关节 | 保留必要两次计算，不假设所有写入都只影响箱子 |
| AMP 判别器 | 依赖每步 before/Post-control 起点与 Physical 终点；不改变事件状态 | 收集两端 83 维特征，H×B 一次网络调用，保留梯度 |
| Actor return Critic | 只读完整轨迹，无循环控制作用 | H+1×B 一次调用；估值保留输入梯度，detach 后复用给 λ-return |
| Critic Adam | 后一次更新依赖前一次权重 | 保持优化器更新顺序、次数和 minibatch 不变 |

因此本轮先实施 AMP/Critic 批处理。未把奖励、Termination 或全部 H32 编译成大图。进一步延后纯奖励子集需要单独证明历史读写、参考坐标与初始步分支等价，不能把第三方无事件任务的计算顺序直接照搬。

## 精简 Context 与解析信息分离

独立 `stage1_dr_v1` 模式，旧模式默认仍为 `full`。`ContextSpec` 根据实际 schema 与机器人 body_names 推导：

| WM 参数 | 当前预期维度 |
| --- | ---: |
| 箱子尺寸、质量、材质 | 3+1+3 |
| 机器人随机质量 | 30 |
| torso_link COM xyz | 3 |
| 机器人原生接触材质 | 87 |
| 合计 | 127 |

canonical State 86 + Target 29 + Context 127 + Event 4 = **246**；旧输入为 884，旧 Context 为 765。启动时还会读取真实启用 DR，输出 `wm_context_audit.json`，未知随机化或躯干以外 COM 变化会拒绝静默忽略。实际维度以启动审计为准。

`compact_context.py` 保留完整物理信息的无损版本库：每个 Replay 转移存 127 维 context 与版本 ID，其余列按每环境变化建新版本。随机质量引起的惯量变化仍完整保存，不能把惯量误当常量。采样时恢复完整 765 维供解析奖励、COM 积分和事件使用。常量字段异常变化有守卫。当前仍存在全 context 解码与选择成本，不能仅按维度比例宣称加速。

`CompactContextWorldModel` 的首层真正是 246 输入；完整 context 只在解析链使用。旧 WM 不做静默权重迁移；旧 Actor 可作为固定采集策略。`context_ablation.py` 在同一冻结真实 Replay 上重新初始化 full/compact WM，保持 Actor/Critic/AMP 对照条件，输出 H1/H8/H16/H32 和真实接触标签分组误差。接触监控关闭时不会虚构接触误差。

## 本轮代码

- `trainer.py`：`batch_imagination_amp`、`batch_imagination_critic` 默认 false；独立 diagnostic trace 可以核对 Physical/Post-control/targets/mask/event/history，不影响正常路径。
- `joint.py`：批量 Critic 的可微 values 传入 Actor return，同一 values.detach 用于 Critic targets；保持截断 bootstrap 和 λ-return。
- `imagination_batching.py`：固定 Replay 起点、随机种子、模型/Adam 状态，依次对照 reference、AMP、AMP+Critic。检查全部返回张量、事件 trace、梯度和实际 Actor/Critic 参数更新；每个变体结束恢复初态。
- `train.py`：增加 `imagination_batching_check` 入口；不保存训练 checkpoint。
- `tests/test_imagination_batching.py`：H32 AMP/Actor Adam 对照、截断与软存活的 Critic 输入梯度对照。
- `tests/test_v24_events.py`：在等待→发射 H32 原生事件测试中增加批处理逐张量与 Actor 梯度对照。

## 开卡后的命令与验收

使用同一份支持 V24 deterministic 的 checkpoint，在独立输出目录运行（将 CKPT 替换为实际路径）：

```bash
bash scripts/train_stage1_world_model.sh \
  --mode imagination_batching_check --num_envs 4096 \
  --config configs/world_model/joint_efficiency_v28_actor_batching_trial.json \
  --checkpoint "$CKPT" \
  --output outputs/world_model/v28_batching_unique_run --no_debug_vis
```

候选配置保持 Actor 2048/H32、WM H8/3×2048、原奖励。对照器自身切换开关，不依赖配置启用开关。不能用 V25 H16 结果冒充 H32 对照。

输出 `imagination_batching.json`：首轮诊断含编译与 CPU 快照的总成本、预热更新、3 次稳态 forward/BPTT/完整 Actor/Critic 更新时间、参数和梯度差、更新峰值显存。首轮总成本不等于纯编译时间，必要时结合 profiler 单独归因。数值检查使用 PyTorch 原 dtype 默认容差，不沿用旧探针较宽参数容差。

报告中的 waiting_to_launch、launched_to_waiting、truncations 是覆盖统计，缺失场景必须补专门原生事件探针。不能把一次随机起点比较通过等同于飞行接触、重投、推力与截断全覆盖。

精简 WM 独立对照：

```bash
bash scripts/train_stage1_world_model.sh \
  --mode wm_context_ablation --num_envs 4096 \
  --config configs/world_model/joint_efficiency_v28_context_reference.json \
  --checkpoint "$CKPT" --context_rounds 500 \
  --output outputs/world_model/v28_context_unique_run --no_debug_vis
```

先用短轮次确认真实 DR/schema、物理重建逐位一致、边界与保存恢复，再跑完整从零 WM 对照。不能把 CPU 合成数据结果当作真实接触或接箱效果。独立新训练配置为 `joint_efficiency_v28_compact_context_trial.json`，不接受旧 full WM 直接续训。

## CPU 实测记录

- 全套回归：108 passed，1 warning，257.82 秒（批处理首轮实现与精简 Context 联合回归）。
- 新增检查使用默认 dtype 容差，没有放宽对照阈值。
- 合成数据 fresh full/compact WM 训练→保存→恢复、策略不变、无损 Replay 版本重建通过；不构成真实 DR/GPU 性能结论。
- 本地 GPU 数据：无。下一步是开卡短回归，不直接启动正式训练。
- 后续增加事件 trace、对照器恢复检查后，针对性回归 30 项通过（176.90 秒）；最终 Context/批处理组 14 项通过（13.58 秒）。日志见 `benchmarks_v28_cpu_20261107/`。这些是 CPU 合成夹具的测试时间，不是训练吞吐。

静态 DR 审计来源：`catch_env_cfg.py` 中 BOX_DR=1 的箱子质量 startup×U[0.8,1.2] 且重算惯量、材质 reset 随机化；BOX_ROBOT_MASS_DR=1 的所有机器人连杆质量 startup×U[0.9,1.1] 且重算惯量。继承的 `tracking_env_cfg.py` 含机器人材质 startup、torso COM startup 和关节 default position startup。stage1_teacher.env 关闭额外 v38 wide mass/curriculum；关节 default position 已通过实际 Target 与原 Actor/解析侧数据使用，不重复塞入 WM context。真实 resolved EventManager 仍需开卡复核。
