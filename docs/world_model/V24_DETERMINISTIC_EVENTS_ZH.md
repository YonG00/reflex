# V24 确定性事件写入版本

## 范围和对照

原配置 `configs/world_model/joint_conditioned_events_v24.json` 和
`ConditionedWorldModel` 保留为学习 Post-control MLP 的对照。
新增配置 `configs/world_model/joint_deterministic_events_v24.json`，开启
`deterministic_event_apply`。不改变原任务奖励、碰撞、AMP、Actor 179 维、
Critic 292 维或动作执行结构；不增加站立课程。

## 数据流

- Physical WM：84 维状态经 canonical 编码，与29维控制目标、物理 context、
  4维当前事件阶段特征拼接；3×2048网络输出82维残差，经原 integrate 得84维 Physical。
- NativeEventScheduler：Physical及FK、事件状态、计时器、已有投掷参数和当前时钟
  输入原生事件回调；输出 WritePlan(targets84/mask84)及新事件状态/计时器。
- Deterministic Apply：`torch.where(mask, targets, physical)`。位置 targets 已转换为
  环境局部坐标；根线/角速度沿用世界坐标；四元数使用原 API 请求值，不再次积分或
  归一化未写入字段。后写覆盖先写；机器人根速度推力、关节写入也适用。
- 采集：PhysX→Physical标签/原奖励及终止→原生事件/API写入→Post-control标签。
  Replay 保留原始事件指令、计时状态及 reset/参考边界；普通事件不截断窗口。
- **条件化监督 H8**：每步 Physical预测→以真实 Physical 标签计算状态误差及预测
  特征 Termination BCE→应用该步记录事件指令→下一步预测输入。
  `L = L_physical + term_loss_weight * BCE`，Post误差仅作诊断，不进入优化目标。
- **自主 Actor H32**：Actor原观测→动作→Physical预测→原解析奖励/AMP/终止→
  原生调度器基于预测状态推进并生成必要新随机条件→确定性写入→下一步观测。
  Actor不读取Replay未来事件指令、真实未来状态或未来随机结果。

### 随机事件监督口径（用户已确认）

真实任务的推力、悬停随机转向、重投等会在运行时新采样。
自主预测若另抽随机数，后续轨迹不再与单条Replay标签处于同一条件；不能把这种差异
全部当作物理网络误差。因此训练明确使用记录事件条件下的H8，而非自主随机H8逐点监督。
自主预测评测和Actor仍按自身调度推进。`event_supervision_protocol=recorded_commands`
写入配置和 checkpoint 合同。不要把条件化误差当成自主误差。

## 写入语义修正

旧 recorder 只记录“数值变化超过1e-7”的分量。确定性写入必须记录 API **实际请求的
全部分量**：相同数值的写入也会改变梯度语义，小于阈值的写入也不能丢失。
对原程序显式 `where(..., 控制值, 当前值)` 的透传分支，原生回调通过
`event_write` 标注 intervention provenance；透传分量不进入 mask，H8仍推进预测状态。
恒值零写入即使数值未变仍进入 mask，梯度不会误当identity。
同样标注hold-init关节写入中未控制的关节，避免灌入真实腿部状态。
该 helper 不改变原API参数、写入调用、随机调用、触发或任务规则。
新旧模型对照均使用同一修正后的采集器。旧格式 Replay 的 mask 无法可靠补全，
不能静默当成新格式恢复；事件代码哈希/学习合同会使完整恢复显式失败。

## 权重与保存恢复

完整恢复要求配置、事件代码/参数合同、模型结构和Replay schema一致。
旧学习型V24→确定性V24须显式 `--warm_start_macro OLD_CHECKPOINT`：
复制Physical网络及全部4维阶段输入归一化参数、Actor/Critic及其优化器、AMP和终止头；
丢弃旧 `post.*`，重建WM优化器、Replay和本实验计数。
不宣称这是逐位连续的完整恢复。未显式迁移直接加载旧完整checkpoint会报错。

按 context_dim=98、3×2048：原学习型WM 9,441,444参数；确定性WM 9,007,186参数；
删除434,258参数。参数减少不代表已经测得GPU提速。

## 验证和当前限制

GPU短回归已完成，最新结果、限制及存档见 [GPU验证记录](V24_GPU_REGRESSION_20261106_ZH.md)。
以下CPU阶段的“待开卡”描述保留为实现时验收清单，当前状态以该GPU记录为准。

2026-11-06 本地最终回归：`PYTHONPATH=source/reflex /tmp/reflex-wm-venv/bin/python -m pytest tests -q`
**67 passed，83.46秒**。事件/模型模块及原生回调语法检查通过；`git diff --check`通过。
两轮审查发现并修复了随机分支错标签和透传回灌真值问题。
全规则 Ruff 检查仍有原有风格告警（例如闭包循环变量、类级可变常量），不计为全量lint通过。


本地CPU覆盖：等待→发射、预测机器人位置→写入目标梯度、重投时钟、批量关联定时器、
H8条件化监督（不消耗调度器随机流）、H32等待起点经历发射、Actor/Critic更新、
联合BCE向Physical反传、训练/保存/恢复、旧权重显式迁移及不兼容恢复拒绝、
局部/世界坐标、关节子集、相同值/微小值/重复覆盖写入。
CPU测试使用小网络和合成物理数据，不能证明PhysX等价或真实接箱效果。

GPU probe已新增真实验证：在 interval 回调前捕获Physical，记录真实API指令，
以确定性写入结果对比回调后的真实Post-control；同时仍验证原生与想象事件指令及
控制状态一致。该新增断言尚待GPU执行，不使用历史V24的通过结果代替。

尚待开卡：真实发射/悬停/自动重投/推力/实际扰动分支、episode与参考边界、
GPU H32梯度/保存恢复、同初始化/种子/采样及更新次数的学习型与确定性型短回归。
分别记录 conditioned/free 的H1/H8/H16/H32误差、launch_write样本数、
发射速度RMSE(m/s)、Physical/Post误差、Actor/WM时间和梯度。
free随机分叉后的逐点误差含随机分支差异，不能单独据此判定Physical网络准确性。
若自然采集中没有实际发射样本，必须报缺失，不以launched标志替代发射速度验证。
**未运行GPU对照，无实际误差或GPU耗时数据；不启动半小时训练。**

开卡后先用8环境运行：

```bash
bash scripts/train_stage1_world_model.sh --mode event_condition_probe \
  --num_envs 8 --config configs/world_model/joint_deterministic_events_v24.json \
  --output outputs/world_model/v24_deterministic_event_probe
```

验证通过后再进行有时间/更新次数上限的短回归，保留新旧两份独立输出目录。
