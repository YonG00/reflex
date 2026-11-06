# V24 Actor 性能定位与等价加速（2026-11-06）

## 已完成：最小事件视图

原事件调度每个回调都调用 AnalyticTask._view，构建奖励/参考/全身刚体数据，
其中 `_body_data` 再次逐刚体计算速度FK。hold-init/realign当步无写入也支付完整成本。
当前启用的5类原生事件只使用根部、箱子、关节、控制状态及原生参数，不读取全身速度FK。

新增 `events_v24/fast_view.py`，只提供这些字段。回调本身、调用顺序、计时器、
随机抽样、WritePlan及全部连续梯度保持原样；未跳过任何回调。
`minimal_event_view=true` 已加入确定性配置。此标志作为执行优化允许旧checkpoint完整恢复；
模型/Adam/Replay结构不变，manifest单独记录执行实现SHA，回退设置false即可。

### 原因证据

预热后完整Actor更新8.54、8.86秒，Critic约0.06秒。
Kineto一次H32记录685,003次 `cudaLaunchKernel`，WM前向不是主瓶颈。
Profiler的递归父子scope会重叠，不能把各scope累计时间直接相加；其本身及事件聚合
有显著开销，不用于宣称训练吞吐。

第三方本地源码版本ba61053：`scripts/bench_imagination.py` 的0.33秒对应2048/H32的
想象前向+反传；`ImagStep` 将WM、FK、Reward、Termination与下一观测合并捕获。
其文档环境为RTX5090/Torch2.14.1；当前是RTX4090/Torch2.7.0。
当前Reflex原 `compile_step=false`，只单独编译WM和位置FK；任务奖励、速度FK及观测仍有
大量eager小算子。当前还包含原Reflex奖励、AMP和原生箱子事件，因此不直接要求秒数逐项相等。

### 同起点H32、2048批量，不更新优化器的对照

每侧独立预热，再测3次，固定采样与事件RNG；表中为中位数。

| 指标 | 完整旧视图 | 最小事件视图 |
|---|---:|---:|
| 前向 | 5.317 s | 3.212 s |
| 反向 | 3.624 s | 3.563 s |
| 前向+反向 | 8.941 s | 6.815 s |
| 单轮全身速度FK构建次数 | 163 | 32 |
| 事件调度主机时间（单次instrumented） | 2.820 s | 0.742 s |
| 全身速度FK主机时间（嵌套于其他scope） | 2.083 s | 0.405 s |

总时间下降约24%；剩余反传约3.6秒，尚未达到1秒。
单次真实Actor Adam更新（2048/H32，含起点采样、裁剪、优化器）为6.872秒。
对照的首轮58.97秒含编译，不用于计算速度收益。

### 等价验证

- CPU全套69项通过；最小视图指令、控制状态、梯度与旧视图完全一致；
  用禁止 `_body_data` 的测试确认事件视图不再调用全身速度FK。
- GPU旧/新H32轨迹与Actor梯度通过，小批梯度最大差0。
- 原生事件24步，含10次真实扰动赋速，Post-control和写入指令最大差0，
  事件状态误差1.2e-7；H32有7条等待起点经历发射，发射位置梯度范数3.397。
- 相同初始参数、Adam状态、2048起点、RNG下实际做一次Actor/Critic更新：
  Actor参数最大差1.68e-8，Critic参数最大差6.71e-8，Actor loss/梯度范数相同。
  两次诊断更新后恢复初始权重、优化器和计数，不另存正式训练模型。
- 旧执行配置checkpoint恢复检查通过；未修改Actor179/Critic292、WM3×2048、
  奖励、AMP、动作控制、终止规则或H8监督。

证据：`evidence/20261106_v24_minimal_view.json`。
服务器输出：`outputs/world_model/20261106_v24_view_benchmark/`、
`20261106_v24_minimal_view_probe/`、`20261106_v24_update_equivalence_r2/`。

## 融合编译已验收

推荐配置 `joint_deterministic_events_v24_fused_packed.json`（通用 fused.json 同配置）：
静态形状 dynamic=False，严格FP32，已编译Physical WM/FK，打包刚体姿态后融合
Reward+Termination，Observation单独编译。原生事件仍按原顺序运行并保留连续梯度。
未采用全图TF32候选：H32累计观测误差0.01115超标，未放宽容差。
`fused_heads.json` 为失败TF32实验；`fused_fp32.json` 为通过的未打包对照。

| 同条件指标 | 编译前最小视图 | 融合后 |
|---|---:|---:|
| H32前向+反向（2048，预热后三次） | 6.475–6.576s | 2.517–2.560s |
| 完整Actor Adam更新时间（预热后） | 6.872s | 3.554、3.718s |
| 完整Critic更新时间 | 约0.06s | 0.045–0.067s |

完整更新时间与纯H32测试不同，包含采样、梯度裁剪、优化器等工作；融合首轮77.779s
包含冷编译，不计入稳定吞吐。实测完整Actor下降约47%，仍未达到第三方不到1秒。

GPU2048/H32输出等价检查通过：观测最大差2.496e-4（绝对+相对联合判据），
奖励3.409e-5，终止概率2.533e-6，离散终止/发射转换差0，Actor梯度最大差3.050e-7。
实际完整Adam更新：Actor参数最大差1.378e-7，Critic5.960e-8；Actor loss相同，
梯度范数0.5283737/0.5283756。诊断后恢复原权重和全部优化器。

另修复Critic相对自身旋转的浮点抵消噪声：六维旋转明确输出恒等旋转，真实导数为0。
原292维结构不变；否则近零特征会让Adam放大编译误差。已测试精确恒等与零梯度。
CPU全套70项通过（201.56s）；新增1024新世界恢复入口及完整融合保存恢复回归通过（138.65s）。

服务器证据目录：`20261106_v24_fusion_fp32_check`、`20261106_v24_fusion_packed_check`、
`20261106_v24_fusion_update_identity`、`20261106_v24_fused_actor_timing`。

## 1024真实采集环境与恢复

此前GPU对照使用256真实环境；Actor始终为2048起点/H32。
显式 `--reinitialize_replay` 保留全部模型/AMP/归一化/Adam和更新计数，允许改变
Replay容量及事件调度器source_world_envs，其他学习/任务契约仍严格校验。
旧环境索引数据不复用，迭代调度从0重新采集40步，来源及新旧环境数量写入initialization。
随机事件使用新世界的原生机制；不改变奖励、物理碰撞或事件规则。

## 已启动限时联合试训

运行目录：`outputs/world_model/20261106_v24_fused_1024_joint_30min/`。
代码版本2b1d61e，真实环境1024，Actor2048/H32，WM H8每轮10次更新；
输入checkpoint为`20261106_v24_paired_r5/deterministic/checkpoint_final.pt`。
显式重建Replay后保留Actor3、WM30、Termination30次更新及对应Adam/AMP。
预算1800秒（训练循环含冷编译，完成当前更新后保存），每500采集迭代存档并最终保存。
TensorBoard位于运行目录的`tensorboard/`；逐步指标`metrics.jsonl`，终端日志`run.log`。

1024预检40步通过，H8有效窗口28346，想象起点32760；Actor观测误差8.073e-6、
AMP1.335e-5、Critic1.095e-5。预检不更新模型；不能视为训练后接箱效果。
正式训练已进入真实采集，后续效果以最终checkpoint与真实play评测为准。
