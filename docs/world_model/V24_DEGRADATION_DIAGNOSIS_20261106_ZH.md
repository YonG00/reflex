# V24 联合训练退化定位与终止边界候选修复

2026-11-06。用户要求停训定位；原1024环境运行已停止，保留checkpoint_00500.pt。
停止于采集迭代806，最后已记录Actor309、WM/Termination3090次累计更新。
SIGINT打断后没有checkpoint_final.pt，不能把最后日志对应权重称为已保存。

## 已有证据

真实奖励由最初10次Actor更新均值-0.004降至约-0.19；累计回合长度从约29步降至约5步。
终止主要是ee_body_pos：任务实际参数为左右踝关节相对参考高度偏差阈值0.25m，
不是接箱手腕偏差。严格接稳成功数一直为0。
后期数据越来越短：WM有效监督步数约6.84→3.3，H32有效想象比例约99%→23%。
这支持策略退化与采样质量之间存在反馈，但没有证明数据文件损坏。

留出误差为混合状态维度RMSE，不应直接解释为米：
首次H1 Physical约0.207，H8条件化约0.374，H32条件化约0.982；
后续H1约0.418，H8约0.489，H32约0.939。
不同时间Replay分布变化，不是固定测试集严格对照；更不能据此认定网络层数不足。

## 确认的想象边界不一致

原imagine中事件调度active排除task.terminated，但有效轨迹仅按truncated与
终止头sigmoid是否达到1停止。解析终止已发生时仍可能继续推进物理状态、计算原奖励。
原fall_terminated是非超时终止当步的一次性惩罚；想象可能在终止后的失败状态重复计罚。

第500步存档只读GPU审计（16起点，H32，无优化器更新）：
- 总Actor梯度范数0.06550，纯survival路径0.06778。
- fall_terminated的with_survival梯度在总梯度方向的投影0.7573。
- cmd_limit_excess投影0.3719，action_rate_l2投影0.1322。
- AMP投影-0.2499；bootstrap梯度范数0。
投影可负、可大于1，不是奖励占比或独立因果贡献。它强烈提示终止/惩罚路径主导，
但单批审计不能证明全部退化仅由此造成。融合编译的输出与完整更新等价测试已通过，
当前没有证据将退化归因于编译。

## 独立候选：保留软存活，解析终止为吸收边界

新配置joint_deterministic_events_v24_boundary_fix.json，hard_imagination_boundaries=true。
未发生解析终止时，仍用原可微sigmoid概率乘积做软存活；发生解析终止则done=1，
保留终止当步奖励，后续无有效轨迹、无未来回报或bootstrap。原生事件规则不变。
这相对第三方纯软存活增加了硬边界，是明确的算法语义变化，不能称为完全等价加速。
原V24配置默认false保留作对照，旧权重需--migrate_imagination_boundaries显式迁移，
保留全部模型与优化器，存档initialization记录迁移；不允许静默恢复不同契约。

CPU全套70项通过（223.52s），包含终止后奖励改为10000仍不影响回报、终止当步奖励保留、
迁移严格校验。GPU16/H32同项测试通过：terminal_reward_max_error=0、future_reward_influence=0，
Actor梯度范数0.68368，优化器更新0。

## 待判定

从与旧运行相同的初始checkpoint、seed42、1024环境、2048/H32出发，限90采集迭代、
最多300秒短回归，目录20261106_v24_boundary_fix_checks/short_regression。
这是候选检验，尚不代表真实接箱效果恢复。没有重启半小时正式训练。

完整分项审计耗时413.66秒。首次与训练并行时因显存峰值叠加导致审计OOM，训练未退出；
已改为串行。新增--aggregate_gradient_audit仅测任务/AMP/存活/价值，不遍历所有奖励项，
减少后续诊断成本，不改变任何训练目标或梯度。

## 用户最终选择：纯软存活，不采用解析硬终止候选

用户明确要求优先SuperMimic纯软存活。上述硬边界版本只作为对照，不能作为正式方案。
硬边界短回归已结束：相同初始存档/seed42/1024环境，前10次真实单步奖励
+0.03699，第二10次+0.04807；旧版分别-0.00424、-0.03347。
这些结果仅说明终止成本/边界有影响，不授权将硬边界变成正式算法。

新的joint_deterministic_events_v24_pure_soft.json：
- Termination Head sigmoid给出d，c按原SuperMimic式c_next=c*(1-d)推进。
- 解析task.terminated不截断Actor想象，也不关闭原生事件调度。
- 仍保留真实episode超时、参考轨迹结束对应的truncation与bootstrap处理。
- 原真实环境的一次性fall_terminated成本C不改；想象用其期望C*d替代硬指示器。
  外层c加权后，c_t*d_t是首次终止的概率质量，不是每步重复收费。
  该替换明确改变想象奖励表示，原真实奖励函数及其权重保持原样；没有新增奖励项。
- WM Physical监督、联合Termination BCE、Actor/Critic结构、AMP、H8/H32/2048不变。
- --migrate_soft_survival显式恢复旧权重与所有优化器，学习契约记录版本。

数学单元测试验证gamma=1时sum(c_t*C*d_t)=C*(1-c_H)，且负终止成本下，
提高hazard会增加损失，不能依靠更早终止逃避反复惩罚。其他奖励列保持不变。
新增GPU检查强制将所有解析终止标记设True，要求reward/done/obs/valid/launch轨迹
与未强制时一致，以证明不存在隐藏的解析硬停止或事件门控。
纯软短回归目录20261106_v24_pure_soft_fix_checks/short_regression，
最多140采集迭代、420秒（含冷编译）。未恢复半小时正式训练。

## 纯软修复验收结果

CPU全套71项通过（215.29s）。GPU16/H32检查：强制所有解析终止标记为True后，
obs/done/valid/launch误差全部0，reward最大差7.45e-9，Actor梯度范数2.0376，
优化器更新0。说明解析终止不暗中门控轨迹或事件；终止成本核实为-1.0。

同初始checkpoint/seed42、1024采集、2048/H32，均新增40次Actor更新：

| 更新区间 | 旧版真实单步奖励 | 纯软修复 |
|---|---:|---:|
| 1–10 | -0.00424 | +0.03901 |
| 11–20 | -0.03347 | +0.05741 |
| 21–30 | -0.08752 | +0.05574 |
| 31–40 | -0.22102 | +0.05142 |

第140采集迭代：累计平均回合奖励-0.4813→+1.4174；累计平均回合长度9.279→34.171步；
严格接箱成功均为0。相同阶段初始WM30/Actor3次，结束WM430/Actor43次。
修复最近10次H32有效比例约81.1%，旧版64.0%；Actor更新约2.86s（旧版2.30s的有效轨迹更少）。
修复循环耗时326.83s，含冷编译。这里是短回归，没有证明长期收敛或接箱已学会。

checkpoint_final.pt已写入，大小1,418,869,325字节，路径：
outputs/world_model/20261106_v24_pure_soft_fix_checks/short_regression/checkpoint_final.pt。
没有继续半小时正式训练。纯软方案取代硬边界候选作为下一轮验证方向。

证据：evidence/20261106_v24_pure_soft_fix.json。
运行时服务器Git尚为2b1d61e加已部署补丁；trainer.py/train.py的SHA256与本地及远程
Git提交e9d5df1完全一致，原始SHA256已附证据文件，不能把旧manifest.git_sha误当生效源码。

## 继续半小时纯软联合训练

用户随后授权继续半小时。最终存档GPU恢复检查通过：解析终止标记强制替换后，
obs/done/valid/launch误差0，reward误差1.86e-9，Actor梯度范数3.7143，无优化器更新。
从上述pure-soft checkpoint完整恢复模型、Adam、AMP、Replay及计数（Actor43、WM/Term430），
没有重新初始化Replay或模型，没有启用解析硬终止。
运行目录：outputs/world_model/20261106_v24_pure_soft_joint_30min/。
配置1024真实环境、2048/H32、WM每轮10次，预算1800秒，每500采集迭代及结束保存；
TensorBoard在tensorboard/，指标metrics.jsonl，日志run.log。
此记录仅证明恢复验收通过并已启动，不代表半小时训练或效果评测已经完成。
