# Stage 1 联合世界模型实验

状态：核心代码与 CPU 合成闭环已实现。**Isaac Sim 接口、真实数据学习质量、接物成功率和训练效率均尚未通过 GPU 验证。没有启动新训练。**

后续进展以 [L40 实测记录](../experiments/2026-11-05_world_model_l40.md) 为准：真实接口预检已通过，首轮模型质量未放行 Actor。当前收束为固定随机数据的 WM-only 对照；默认 WM 训练窗口已对齐参考的 8 步，32 步 Actor 想象不在当前验证范围内。本文下方完整训练命令保留作为后续入口，不代表当前正在运行。

设计依据：用户的《Reflex × SuperMimic：Stage 1 动态接箱 Teacher 改造方案 V2》。该方案是实验目标，不是已经证明有效的算法。参考 [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)，审阅版本 `ba6105349069bf040f1fdad60873d13138d505a2`。λ-return 方程及归属见同目录 `SuperMimic-Reproduced.LICENSE`；没有移植其 MuJoCo 接口。

## 实现与明确的研究选择

- 独立入口 `reflex.world_model.train`，循环为真实采样 → 联合模型多步拟合 / 真实 AMP 判别器训练 → 32 步可微想象 → Actor 路径梯度 / Critic λ-return。Actor 不经过 PPO clipped loss。原 PPO 入口保留；共享 AMP 代码抽到 `reflex.amp.context`，原完整存档回归测试继续保留。
- 保存原始 world-frame 84 维物理状态：机器人 71 + 箱子 13。四元数 wxyz，机器人和箱子的线速度均为 COM 世界速度、角速度为世界速度；位置是 link 原点。积分通过实际 COM 偏移修正 link 速度。网络使用 root heading 不变的 86 维特征、82 维残差积分，不把原始 84 维直接当作所有网络的宽度。
- Oracle Actor 为 179 维。保留原参考 q/dq、锚点、本体、上一动作，关闭观测噪声与检测器，去掉 blind/trust 观测，增加箱子相对姿态、角速度、实际尺寸和质量。保留真实 reward manager 的全部原始奖励及权重；没有修改成功标准。由于信息条件改变，新增同配置 Oracle PPO 对照，不能只与原 noisy PPO 比较。
- 原 170 维 noisy PPO 不改变。Oracle 的两阶段抬臂改用已有的 launch-driven 分支，两个新算法使用同一分支；原感知触发计数器不是新实验的控制合同。
- 每一步用实际导入的 URDF、真实关节顺序做 Torch FK，重建 torso anchor、参考相位、root/box 局部观测及 AMP 83 特征。每个环境的随机化 default q/dq 分开保存，不能假定共享。保留原 AMP 的列拼接 6D 与 Actor 的矩阵前两列 row-major 展平差别。
- WM 的动作是 `process_actions` 后、gimbal 前的关节目标。真实 `process_actions` 与纯 Torch 两阶段/动作尺度/腕锁/可选目标裁剪逐步比对。**每个物理子步的 wrist gimbal + PD + PhysX 合并为 WM 要学习的闭环 plant**；不是声称 gimbal 前目标就是实际最终 PD 目标。gimbal 的反馈依赖当前物理状态，没有额外影响动力学的跨步 latch。
- ξ 读取实际生成几何、PhysX mass/inertia/COM/material，以及有效 stiffness/damping/armature/friction/effort limits。尺寸不从随机化后的质量反推；查询成本也计入端到端吞吐。参考相位、yaw、上一动作、blend、控制参数和奖励历史另存，不混成物理状态。四个奖励/扰动历史计数和 episode/launch 时间由辅助 memory head 学习，属于模型近似，必须随真实多步数据检验。
- 第一版**所有 task reward 都使用逐项学习头**。真实监督标签直接取原 reward manager 每项加权贡献（包含 control dt），并断言逐项和等于真实总奖励。`manifest.json` 逐项登记 func、weight、代理类别，没有无声删除项。这样能完整覆盖依赖接触传感器/历史计时/离散判定的奖励，但 scalar 拟合好不保证梯度正确；逐项 reward error、Actor 的各奖励梯度范数、task/style 梯度都记录。后续可把经过验证的项替换成解析函数。
  奖励头额外输入每步重建的完整 Oracle observation（包含相对参考位姿、上一 raw action）以及当前 raw action。仅用 canonical state 与关节目标无法表达固定参考的锚点偏差，也无法区分腕锁后目标相同、动作变化惩罚却不同的情况；回归测试包含这两个反例。
- AMP 判别器只训练真实政策转移与原专家池；想象只查询冻结的判别器，保留对输入的梯度。归一化、reward 公式、权重与 ramp 复用原实现。Critic 是 179 维 Oracle observation value，作为首版的部分可观测近似；并非完整仿真状态 value。

## 数据与放行条件

`IsaacAdapter` 在真实 reward 计算后、自动 reset 前截取末状态、终止和 reward labels。外部 root/joint 状态写入按实际改变的环境标边界，避免 hover 代码每步写全部环境时误伤已发射轨迹。clip 结束的 command 重采样、episode reset、发射、重投、push、cradle kick 均中断连续窗口。重启恢复时也在旧 replay 末端插入边界。

人工悬停段不拟合为物理动力学。想象起点从已经发射、相位与历史窗口足够长的有效状态均匀采样，覆盖飞行和接取过程；不只采发射帧。当前四阶段统计用距离/速度几何分箱，**不等于通过传感器证明的接触标签**。未实现可选的固定比例分层采样。

模型拟合和验证按整个环境 ID 隔离：`env_id % 5 == 0` 只用于验证，避免相邻帧泄漏。记录 1/4/8/16 步物理误差及逐项 reward 误差。默认阈值只是保守实验初值，不能当作文献验证过的安全界限。每次拟合后验证，Actor 只有 `--enable_actor` 且预热完成、全部误差门通过、真实动作排序探针通过才更新；不通过会记录原因并继续采集/拟合，不宣称已训练 Actor。

排序探针对相同显式物理/控制/事件/传感器状态执行重复动作与 ±扰动动作，比较 4 步真实回报与模型回报的排序。发生外部事件、终止或 reference 重采样的样本排除。至少 16 对有效比较，排序一致率至少 0.6；重复分支回报差不超过 0.02。**PhysX 接触 warm-start 缓存不能通过现有 API 完整恢复**，因此探针包含重复分支检查，并在报告里保留限制。该探针仍需 GPU 验证，不是已证明的反事实真值器。

探针的所有真实步，包括重复、失败或最终被筛掉的分支，都计入 `progress/probe_transitions` 和 `progress/real_transitions` 总数；`progress/collect_transitions` 单列进入 replay 的采样预算。恢复 simulator 计数器不等于这些计算免费。计时包含 collect、model fit、AMP、validation、ranking、imagination、Actor、Critic 和 checkpoint，CUDA 阶段边界同步；replay sampling 耗时标明是包含在上述阶段内的子项，不能重复相加。正式比较以总真实交互和实测端到端耗时为准。

Timeout 与真正 terminal 分开：模型终止头不将 timeout 当 terminal；虚拟 32 步截断保留 Critic bootstrap。真实仿真状态不随 checkpoint 恢复，不能声称逐帧无缝。Checkpoint 保存 Actor、Critic、WM、归一化、FK/参考表、三个优化器、完整 WM replay、共享 AMP 完整状态、计数、配置、合同及 RNG。checkpoint 使用原子替换；不同专家数据、配置、URDF/动作合同会拒绝完整 resume。

## 开卡后执行顺序

当前默认参数刻意使用 64 个真实环境开始接口验证；不是为 24 GB 显存优化后的最终配置。先设置服务器现有 Python：

```bash
export ISAACLAB_PY=/opt/conda/envs/isaaclab/bin/python
# 1. 真实 FK / observation / AMP / action 对齐 + 事件窗口检查，无 Actor 更新
bash scripts/train_stage1_world_model.sh --mode preflight --num_envs 64 \
  --preflight_steps 512 --output outputs/world_model/preflight

# 2. 完整小规模训练链；Actor 仍须通过模型误差与真实排序门
bash scripts/train_stage1_world_model.sh --mode train --num_envs 64 \
  --max_iterations 60 --enable_actor --save_every 10 \
  --output outputs/world_model/smoke

# 3. 加载本实验 checkpoint 录像（非正式成功率统计）
bash scripts/train_stage1_world_model.sh --mode play --num_envs 64 \
  --checkpoint logs_or_outputs/checkpoint_final.pt --video \
  --output outputs/world_model/play

# 4. 相同 Oracle 条件的 PPO，从零开始
bash scripts/train_stage1_oracle_ppo.sh --num_envs 64 --max_iterations 60 \
  --seed 42 --run_name oracle_control
```

`--max_iterations` 是累计更新目标。完整 resume 使用相同 JSON、num_envs、环境标志和 `--checkpoint`，输出目录另起。`preflight.pt` 的 collect_steps 临时修改已恢复，允许用相同配置加载；仅用于连通性检查，不是有效接物模型。完整训练前应先确认模型 loss、4/8/16 步误差和 ranking 有足够的飞行/接触覆盖，且 TensorBoard `gate/actor_enabled` 确实为 1。

正式评测复用原 CLEAN2 evaluator 和 summarize，没有重写成功定义：

```bash
bash scripts/eval.sh world_model --ckpt PATH/checkpoint_final.pt \
  --num_envs 320 --seed 8 --out outputs/eval/wm_regular_s8.json
bash scripts/eval.sh oracle_ppo --ckpt PATH/model_59.pt \
  --num_envs 320 --seed 8 --out outputs/eval/oracle_regular_s8.json
```

这两个模式只切换 Oracle observation/动作触发合同和对应 Actor loader，仍使用相同 condition/protocol/成功汇总。WM Actor 的 observation normalizer 会显式加载；旧 PPO 的原 Identity normalizer 断言保留。运行种子 8、9，按原规则汇总 640 次投掷，再比较相同真实 transitions 与端到端 wall time。不要按 iteration 数直接比较两个算法。

每个 run：`manifest.json`（配置/合同/奖励分类/版本）、`parity.json`、`metrics.jsonl`、`tensorboard/`、`ranking_*.json`、`checkpoint_*.pt`、可选 `videos/`。失败的 parity 直接终止；不要绕过后开始训练。

## 本次无卡验证与尚未证明的内容

CPU：Torch FK/坐标系与 heading 不变性、四元数积分/梯度、腕锁/两阶段动作、ring wrap 和外部边界隔离、timeout bootstrap、完整合成 collect→fit→imagine→Actor/Critic/AMP 更新、模型/判别器冻结、逐项梯度、全状态存档恢复后的下一次更新一致性。原 AMP 下一步更新回归也通过。另执行 Python 编译、静态错误检查、shell 语法检查。

这仅证明实现的 CPU 合同与合成闭环。真实 Isaac 初始化和所有 parity、真实接触学习质量、参数是否合适、是否胜过原 PPO、24 GB 最佳环境数，以及与标准 PPO 相比是否提速，都仍未证明。第一轮 GPU 测试可能暴露 URDF 导入固定关节变换、PhysX getter、传感器缓存或参考动作索引差异，应修复真实不一致，而不是放宽门限掩盖。
