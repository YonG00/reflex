# V36：先验证完整Actor，再决定是否研究训练效果

状态：本地实现与CPU验证完成，GPU待开卡；没有新的GPU提速/训练效果结论，正式配置不变。保留约0.077秒的最小Actor＋WM融合候选，不因严格容差失败直接撤掉，也不根据误差小直接启用。

## 已知证据与缺口

V35已经确认：禁用Triton FMA后，原始features与eager相同、encoder误差从约3e-5降到约1.9e-6。单靠该开关仍不能让最小H32和Adam全部通过。

`wm_native_split`（原生SiLU/除法、编码网络编译、原积分在图外）在同一capsule最小链通过：H32状态差0、Actor首层梯度最大差约5.59e-9、首层实际参数差约3.28e-7，耗时0.3543秒，对照eager0.4910秒。它不是约0.077秒的激进融合，也不是完整任务加速结果。

前次真实任务回归：日志仅能证明三个H8分支执行完。完整Actor冷编译超过600秒预算，未产出最终JSON，H8数值门控也未及时保存，**不能记为H8/完整Actor通过**。超时后shell退出但Python子进程未退出，已结束自有GPU进程；本轮补进程组清理及原子落盘。没有用当前CPU测试覆盖这项GPU缺口。

## 推进顺序

1. 完整H32：状态物理量、奖励、AMP、软终止/存活、原生事件、梯度及实际Adam更新。
2. 优先比较生产基线与快速融合的完整Actor＋Critic耗时/峰值显存/冷启动成本；非WM编译关闭的隔离模式不作生产测速。
3. 若收益值得投入且没有行为/数值异常，再准备固定Replay连续更新及真实采样短训。当前入口不启动这两项，也不保存新策略checkpoint。
4. 后续分别按相同更新/真实样本预算、相同wall-clock预算，比较站立、回报、成功率、收敛与吞吐。没有提前指定宽松容差或以吞吐代替效果。

## 严格检查与风险描述并存

原dtype默认`torch.testing.assert_close`继续原样执行，失败仍记录在`numeric_passed`/failures。新`numerical_risk_v35.py`提供**无新通过阈值**的描述性指标：

- 分开报告米、弧度、m/s、rad/s的状态最大差/RMS，四元数角误差对正负号等价。
- 原始误差仍完整保留；额外给出参考软存活前缀×valid加权的状态RMS，识别回报已不再使用的晚期/截断状态误差。权重只用于诊断，不改loss，也不替代未加权最大误差。
- 原始Actor梯度与实际应用梯度的方向、范数、相对L2；Adam比较的是相同初始权重出发的**更新delta**，不能拿post-weight余弦掩盖差异。
- 奖励/AMP/软终止差异、H1/H8/H16/H32存活差异，以及发射模式、事件写入mask、有效/截断标记的精确不一致计数。

这些指标不自动把失败变通过。事件规则变化、NaN/Inf、梯度链异常要先处理；只有连续浮点差异并不自动淘汰研究候选。完成研究计时不等于具备正式训练资格。

## 完整Actor生产对照入口

```bash
ISAACLAB_PY=/opt/conda/envs/isaaclab/bin/python \
REFLEX_DATA_DIR=/root/gpufree-data/reflex/data \
bash scripts/study_actor_core_v36.sh \
  /tmp/reflex_v29_scratch_joint30_20261107/checkpoint_final.pt \
  /tmp/新的V36完整Actor对照目录
```

使用版本化V32生产配置：4096真实环境，Actor2048/H32、WM3×2048、FP32/TF32关闭、原Reward/AMP/事件/软存活保留。源checkpoint恢复AMP/优化器后采集40步新Replay，所有组共享同一起点/权重/Adam/RNG，保持原生随机调用及控制规则。

- reference：现有完整执行路径。
- fused_no_fk：`actor_physical_core`融合Actor、实际观测/动作处理和Physical WM，FK继续原独立路径；不复制最小链的观测代理到任务里，不编译整个H32。
- 只切换已有编译执行开关，保持网络、H32、batch、奖励权重、事件实现不变。FMA采用生产默认1，与V35的FMA=0缓存分开。
- 每组2次预热；完整轨迹/梯度检查后20轮交替配对真实Actor/Critic更新。严格数值失败仍可完成**研究计时**并打印失败，不自动进入训练。
- 输出`actor_fusion_v36.json`含状态/梯度/事件/实际更新比较、风险指标、完整更新分布和显存。单次traced前向/BPTT不是稳态分项估计。
- `performance_usable`要求稳态无新增编译；冷启动含编译/初始化/进度记录，不能称为纯编译时间。观测到明确收益前不做后续训练实验。

默认总预算300秒。若冷编译超预算，保存活动阶段/每60秒Python堆栈，再停止该私有进程组；不再无限等待。若CPU/冷编译阻止完成，报告未完成，不能报最快样本。

## Physical WM真实数据数值隔离入口

```bash
V35_FULL_STAGE=wm bash scripts/study_physical_full_v35.sh CHECKPOINT NEW_OUTPUT
V35_FULL_STAGE=actor bash scripts/study_physical_full_v35.sh CHECKPOINT ANOTHER_NEW_OUTPUT
```

默认`V35_TASK_EXECUTION=eager`仅关闭非WM的FK/奖励/观测等编译，计算本身、所有事件、随机规则与模型保持。它用于数值隔离；生成的实际config.json归档，不能拿其耗时对比生产路径。

H8和Actor可以独立进程执行，不因后续Actor冷编译失去WM结果。字段`candidate_wm_numeric_passed`、`candidate_actor_numeric_passed`与全链`candidate_numeric_passed`分开，不用部分通过冒充全链通过。`all`可显式选择；WM脚本默认阶段wm、总预算240秒。需恢复生产编译可显式选`V35_TASK_EXECUTION=compiled`，该入口不替代上述V36性能对照。

## 进度、超时与恢复

`diagnostic_watch_v35.py`在每个WM分支和冷阶段原子保存JSON，warm中记录Actor前向/反向和观测/WM/FK/head的当前调用；不包裹依赖函数身份的_view。暖图和稳态计时不装记录hook。堆栈用于区分代码生成、等待或实际执行，不以GPU利用率单独判断。

`run_diagnostic_group.py`创建独立进程组，超时先请求中断、宽限后清理剩余组，父shell提前退出也检查子进程；恢复继承的nohup信号设置。CPU用故意忽略信号、父提前退出、实际超时、外部终止的心跳进程验证停止，不只看父进程退出码。

诊断结束恢复Actor/critic/WM、优化器、RNG和临时编译指针。中断会保留已完成检查和失败状态；强制终止不能声明模型已恢复或全部检查通过。

CPU证据见`benchmarks_v36_cpu/README.md`。没有GPU验证、没有正式训练配置合入，也没有以这轮CPU结果证明0.077秒可以用于真实任务。
