# V29：解析 Soft Termination 与 Critic 箱子信息补齐

## 状态与边界

本地实现与 CPU 验证；GPU 未开启，无真实状态分布、稳态 GPU 耗时或站立/接箱效果结论。不进行长训练。旧 Learned Head、292 维 Critic、旧配置与 checkpoint 保留；新模式显式配置启用。

## 一、原有链路与查实的问题

Actor 观测为 179 维。`PrivilegedObservation` 旧实现拼接 `actor[:67] + body_pos42 + body_ori84 + actor[67:166]`，合计292。它已有箱子相对位置和线速度，但漏掉 Actor 最后的13维：姿态6、角速度3、尺寸3、质量1。

新增 `critic_include_all_actor=true` 后，在原292尾部补 `actor[166:179]`，得到305维。Actor仍179、动作29，Critic隐藏层512/256/128不变；第一层输入必须从292改305，不能称为旧权重直接兼容。保留原292列顺序以支持审计与显式迁移。新增信息来自当前观测，不向Actor增加特权信息，不改原生PPO任务定义。真实采集 parity 先检查原生292，再以真实Actor观测末13列验证新305。

原终止链路：真实环境产生硬终止标签→Replay；WM H8预测状态→终止几何特征→MLP→BCE联合更新；想象中MLP sigmoid得到hazard，进入连续存活权重。原 `pure_soft_survival` 绑定 learned head；原 valid 还有 `p.detach() >= 1` 路由。

## 二、第三方精确公式与时钟

核对版本：[supermimic_vibed 0525a23](https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/supermimic/mdp.py)，`soft_termination()`：

```
e_height = abs(reference_root_z - predicted_root_z)
e_tilt   = abs(reference_R[2,2] - predicted_R[2,2])
p_height = sigmoid((e_height - 0.25) / 0.02)
p_tilt   = sigmoid((e_tilt   - 0.8 ) / 0.04)
p        = 1 - (1-p_height)*(1-p_tilt)
```

[同版本 train.py](https://github.com/zita-ch/supermimic_vibed/blob/0525a23a2dcfa61f8543f98b84c075fe66d9d286/supermimic/train.py) 的 ImagStep 先预测 S1、递增 f1=f+1，然后以 S1/f1计算软终止。本实现相应使用 **Physical状态 + task.phase（下一参考帧）**，参考末尾受真实 reference truncation 约束，索引clamp到最后有效帧。不沿用旧Head的当前帧anchor几何定义。

Reflex的84维状态是link-frame、wxyz四元数；根为pelvis，取参考轨迹pelvis的z/quat，不偷换为torso anchor。单位四元数的 `Rzz = 1-2*(x*x+y*y)`，直接计算矩阵分量，不用欧拉角。仿真/积分已有单位四元数；世界yaw旋转不改变z与Rzz，因此不需要引入Actor看不到的新信息。CPU用独立完整3×3矩阵实现对照。

阈值、温度固定，没有新增可调试验参数。公式本身含绝对值的零点次梯度、sigmoid的有限精度饱和；“保持梯度”表示不人为detach/硬化，不能保证极端状态下梯度永不为0。

## 三、新模式数据流

`analytic_soft_termination=true` 要求：state WM、pelvis参考、`pure_soft_survival=true`、`learned_termination=false`、`joint_predicted_termination=false`、`hard_imagination_boundaries=false`。非法组合直接报错。

- 真实环境：PhysX、硬终止、reset、奖励与事件保持原规则；真实标签仍存Replay。
- WM H8：Physical预测误差→WM反传；正常事件按原Recorded Commands协议推进Post-control；不创建Termination MLP或其Adam，不做BCE。融合WM步同样支持此分支，省去BCE专用FK。
- Actor H32：Actor→WM Physical→原解析奖励/AMP + 解析hazard→原生事件与确定性写入→下一状态。事件与历史仍按步执行。
- Actor目标：连续 `c_(t+1)=c_t*(1-p_t)`；概率不detach、不变bool，也不因概率饱和1更新valid。valid只受真实时间/参考截断等既有非概率边界影响。即便存活权重已为0，仍递推完整H32。
- Critic：原λ-return、优化器步数与网络隐藏结构保留；新305输入开关独立于hazard模式，可分别消融。

原 `fall_terminated` 奖励列替换为 `p * (原weight*dt)`，外层c_t使它按“首次终止概率质量”计费，不能再叠加原硬终止罚。其他解析奖励与权重保持，包括连续摔倒整形项。新hazard仅表达指定根高度/倾斜规则，不拟合真实环境所有身体/箱子硬终止条款；这属于所要求的设计差异，不能把它当作BCE校准概率。

## 四、保存、恢复与显式迁移

默认load严格检查配置与learning_contract：旧/新hazard或292/305混用会报错，不静默忽略Head或改变第一层。

- 旧Head→解析：`--migrate_analytic_soft_termination`。只允许v3、已pure-soft、无hard boundary的Learned源模型；移除Head与其Adam，保留WM/Actor/Critic及其优化器、Replay、AMP。新Head更新计数归0并记录migration。
- 旧292→305：`--migrate_critic_box_observation`。原292列权重保留，新13列权重初始化0，故迁移瞬间同一物理输入的value保持不变；新列normalizer mean0/std1，原292列normalizer保留。**Critic Adam显式重新初始化**，Actor/WM/AMP不重置。新列会在后续训练中学习，隐藏层权重不丢弃。
- 两项可同时迁移。奖励、Horizon、更新比例等其他变化仍由原严格检查拒绝，不能借此标记绕过。
- 新存档后续恢复不再传迁移标记；反向迁移不支持。新解析模式不能通过macro warm-start隐式丢弃旧Head。
- full765→compact127的WM结构差异独立处理，仍要求重新初始化WM，不能混入此次checkpoint迁移。

## 五、GPU短测入口

先保持292布局做hazard单因素对照（同一源checkpoint/Replay/seed/GPU），不混入305变化：

```bash
bash scripts/train_stage1_world_model.sh \
  --mode soft_termination_check --num_envs 4096 \
  --config configs/world_model/joint_efficiency_v29_learned_critic292_trial.json \
  --checkpoint "$CKPT" \
  --output outputs/world_model/v29_soft_check_unique --no_debug_vis
```

`soft_termination_check.json`记录真实状态与H32轨迹矩阵公式差、概率分布/精确0和1比例、H1/H8/H16/H32纯软存活率与考虑truncation后的有效存活率、梯度有限性/范数/方向差、forward/BPTT与完整Actor/Critic更新耗时、峰值显存、首轮含编译成本。对照的两个算法目标不同，梯度不要求相等；公式对照不放宽默认容差。结束恢复权重、Adam、随机状态与计数，不保存训练模型。等待→发射覆盖不足要补原生事件探针，不能仅看概率统计就认可所有场景。

GPU通过后，组合模式的独立短回归配置为：

```
configs/world_model/joint_efficiency_v29_analytic_critic305_trial.json
```

从旧Learned292源加载此配置时显式增加两项迁移标记；保持checkpoint原有Replay要求，如源没有Replay需同时按原规则 `--reinitialize_replay`。不启动半小时训练。

另外两个配置 `...analytic_critic292_trial.json`、`...learned_critic305_trial.json` 用于分离hazard和Critic信息补齐的作用。

## 六、本地验证与待验证

代码位置：`analytic_termination.py`（公式）、`observation.py`（305）、`trainer.py`（无Head H8/H32、合同）、`termination_migration.py`（显式迁移）、`isaac_adapter.py`（双层parity）、`soft_termination_check.py`（对照）、`tests/test_analytic_soft_termination.py`。

已覆盖：矩阵公式与梯度、阈值处梯度、Critic完整字段、H8无BCE、参数/优化器迁移与恢复、原生等待→发射、p=1仍展开H32、编译后公式与融合WM路径、测试结束状态恢复。

待GPU验证：真实状态与参考坐标对齐、真实概率分布/饱和、真实H32存活率与Actor梯度、原生305观测parity、性能与显存、站立/接箱效果。CPU合成测试不证明这些真实效果，也不说明解析hazard一定优于Learned Head。

### CPU 结果（2026-11-07）

- 全套回归 **117 passed，273.24 秒**。该轮两条warning分别为旧TorchScript弃用提示和诊断日志标量转换提示；后者已改为仅在日志侧detach，不改变目标的梯度。
- 最终新增组 **9 passed，23.39 秒，无warning**，并包含305 Critic + AMP/Critic批处理 + 解析hazard组合的真实优化器更新/恢复测试。
- 独立3×3矩阵公式与优化实现的前向/梯度在CPU测试中最大绝对差0；原生事件夹具在hazard=1时仍完整经历等待→发射。
- 日志和合成CPU探针JSON存于 `benchmarks_v29_cpu_20261107/`。JSON中的时间/分布只用于确认探针可执行，不代表真实环境状态、GPU训练速度或最终成功率。
