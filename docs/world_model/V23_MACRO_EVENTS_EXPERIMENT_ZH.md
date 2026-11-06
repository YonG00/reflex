# V2.3 黑盒外部控制实验

用户最终选择：不再只学连续自由物理，也让 WM 学习程序悬停恢复和赋予初速度。此选择覆盖 V2.2 的“在箱体外部写入处截断”策略；V2.2 保留作历史对照，不再作为这组实验配置。

## 训练目标与时序

`joint_macro_events_v23.json` 保留3层2048共享主干、10次WM/1次Actor、2048窗口/Actor起点、8步监督、32步想象。输入增加当前已知发射标记；不提供未来发射指令、未来初速度或未来随机数。

同一WM输出三个预测：

1. `physical`：真实物理步结束、reward-hook时刻的状态。
2. `post_control`：同一步中程序完成位置/速度写入后的状态；用学习到的残差修正physical，dt=0，不再额外做一次物理积分。
3. 下一步发射标记logit。训练时用真实after_launched做BCE，自回归时用预测概率反馈WM、阈值解码更新原观察/目标处理的离散标记，不调用发射或悬停程序。

真实数据的`physical_next_state`来自reward-hook；非复位的`next_state`来自完整env.step后的真实快照。悬停、发射、任意native位置/速度写入保留为监督目标，`external_boundary`仅作统计，不再截断WM窗口。终止/超时、reference重采样teleport仍隔离：目标回退最后physical状态，回合之间不拼接。

损失是有效窗口平均的 physical state loss + post-control state loss + launch-mode BCE + 预测physical特征的termination BCE。状态维求和，两个BCE权重1；同一次backward、统一梯度裁剪、两个不重叠且同步学习率的Adam。所有真实合法阶段均匀有放回采样，包括静止/等待、飞行、接触和减速。

Actor用预测physical计算原Reflex奖励、AMP和物理终止；后续观察/状态用预测post_control和预测发射标记。这样程序写入后的速度不会错误地替代写入前奖励时刻。箱体外部事件不再触发Actor截断；回合或参考动作终点仍截断并bootstrap。真实PhysX、原奖励及依赖传感器保留，新增接触监控继续关闭。

## 实验性质与初始化

这不是SuperMimic原版的严格复现，而是用户选择的黑盒宏观动力学对照。未输入隐藏的发射时间/已采样投掷参数，因此不宣称完整Markov状态或可消除预测歧义；确定性网络可能学到混合条件的平均变化，最终以真实闭环结果判断。

从V2.2短回归存档热启动：保留Actor/Critic/归一化/终止头/AMP及可复用物理WM权重，扩展输入与输出。新WM优化器和Replay重建；新模型、数据目标和learning_contract单独存档，不能冒充原存档的完整resume。

每次Actor仍记录真实梯度范数；昂贵的全量逐奖励梯度分解已独立验证，本组限时试训`diagnostic_every=0`，避免一次约396秒的只读诊断占用训练预算。WM、BCE、校准、速度、真实episode/throw/严格成功统计继续记录。

## 放行检查

- CPU：热启动物理预测保留、macro标签保留速度写入、8步窗口跨箱体写入、预测mode反馈、等待不调用外部scheduler、Actor梯度非零、联合损失与终止梯度、普通程序写入与非done reference reset区分。
- GPU受控native API：写前/写后速度与位置逐步记录；同一launched=True下速度写为4m/s仍纳入macro目标；no-op写入不伪造物理变化；实际发射后状态保留；地面接触减速数据保留。
- GPU短联合回归：须真实train→save→load，并报告稳定Actor更新耗时和宏观预测误差；通过前不启动半小时正式试训。

截至本记录建立时，CPU已通过，GPU受控探针和新模型短回归正在执行，尚无本组接箱成功率结论。
