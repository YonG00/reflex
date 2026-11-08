# V36 融合收益与执行路径审计

2026-11-08；离线代码审计，无训练算法修改。

## 比较口径

同一轮V34最小链：eager 0.45953s，WM单独编译0.10579s，Actor+WM融合0.07719s。相对已经编译WM，进一步融合仅节省0.02860s。约0.491s的eager来自另一轮，不应混算。这些编译候选严格数值未通过。

完整V32基线已经编译WM、FK、观测和packed奖励/终止。不能把最小链eager到融合的全部收益，当作完整生产基线还能兑现的收益。最小链使用观测代理、固定事件条件、合成目标，不等于完整任务。

## 入口审计

`actor_fusion_study_v36.check`给候选显式设置 `_actor_core=compiled`。`trainer.imagine`在core存在时跳过原观测/Actor/target分支，随后走reward_heads_from_poses而非重复Dynamics的_v24_step。静态代码未发现候选同一步重复执行Actor或Physical WM。

core包含实际观测、归一化、Actor、动作控制与Physical WM。V36候选明确排除FK；奖励、事件、AMP和Critic仍在图外执行。外围耗时实际占比需要GPU测量，不能凭代码体积判断。

## GPU证据

r7关闭定时堆栈后无139；完成5组配对，总预算600秒退出124。部分原始JSON已归档：reference中位1.586098738s，candidate中位1.538156942s，约3.02%缩短。候选严格轨迹失败，单次实际参数/梯度更新检查通过，不等于训练等价。

r8现场检查：退出0、20/20组、completed=true、performance_usable=true、numeric_passed=false；稳态新增graph/frame/graph-break均0，Actor恢复，WM未变。候选完整更新中位1.595167235s。完整JSON仍在服务器 `/root/gpufree-data/reflex_v36_actor_cached_r8/run/actor_fusion_v36.json`。SSH在本地传输时断开，尚未归档r8完整原始JSON，不报告未保存的reference中位或最终加速比。测试进程已退出，检查时GPU85MiB。

## 下一步

离线准备同真实输入的分段core/融合core前向及VJP对照，核对实际编译边界。下一次开卡分别测core/FK/heads/events/AMP/BPTT/critic，Profiler分项与不插入逐模块同步的稳态计时分开。严格误差、事件覆盖和实际Adam更新检查保留，不改变H32和训练语义。

目前代码审计与实验准备不需要GPU；GPU数值、kernel和反向计时验证才需开卡。

## 进一步离线研究与诊断实现

### 必须区分的计算依赖

| 部分 | 当前执行 | 反向依赖与审计结论 |
|---|---|---|
| Actor/动作控制/Physical WM | H32逐步；候选单步融合 | core存在时原分支不再调用，无静态重复WM证据 |
| FK、真实观测与Critic特征 | 每步观测使用姿态；Physical FK独立编译 | 批量Critic网络不代表Critic特征已全部批量；sparse_actor_bootstrap已有实验，不能重复声称新优化 |
| 奖励与软终止 | packed heads逐步执行 | 历史更新与事件依赖存在，不能整体随意挪到H×B批量 |
| 事件输入/WritePlan/写回 | 顺序推进，含动态活动索引 | 写入与预测状态的连续梯度必须保留；调用数不能代替实耗时 |
| 姿态与状态携带 | packed pose carry及where | 仍在BPTT路径中，不能detach或省略无写入事件的内部状态更新 |
| AMP与Critic网络 | 已启用批处理 | 不重复实现已有批处理，不把其误报为缺失 |
| Actor Return | joint.returns内H32顺序存活递推 | 仍有逐步张量运算，不代表它一定是主要热点；须GPU归因 |
| 指标及Critic有效样本筛选 | 更新级别float/int和mask索引 | 有同步点，但不是每个H32步都发生；不能按调用次数夸大占比 |

### 新增研究开关

原有完整Profiler的scopes会拒绝fused Actor，且会替换actor.forward与targets。新增显式 `allow_actor_core=True`：只在 `_actor_core` 外打点，不替换融合内部Actor与target函数；旧入口默认行为保留，整段full-step图仍拒绝混用。连续梯度、原始事件与网络结构不变。

`V36_PROFILE=1` 让现有V36对照完成后，再分别运行生产分段/融合完整Profiler。沿用真实Replay、相同权重/RNG与Snapshot恢复，输出完整trace及前向/反向归因；对比sample SHA，检查打点前后轨迹/梯度、实际Adam更新与捕获期间编译计数。无法归因的反向工作保持unmatched/shared，不分摊猜测比例。Profiler时间与未打点完整更新分开，不相减构造虚假模块wall-clock。

待GPU调用（当前不执行）：

```bash
V36_PROFILE=1 V36_STACK_SECONDS=0 V36_TIMEOUT=600 V36_REPEATS=5 \
  bash scripts/study_actor_core_v36.sh CHECKPOINT NEW_OUTPUT
```

输出 `run/profile_reference/`、`run/profile_fused_no_fk/`；父结果profiles为打点自等价检查，不替代原分段/融合间的严格数值检查。没有正式训练配置变更，也未提交或push这些研究改动。

### 本地验收结果

- 新增融合Profiler内部函数身份、轨迹/梯度自等价与实际Adam更新、捕获期间无新增编译检查；新增可选配对Profiler与同起点检查：2 passed（95.96s，CPU/aot_eager）。
- 新增候选Profiler失败后的flag/core恢复：1 passed（2.17s）。
- 原完整Profiler实际采样/梯度/Adam/RNG恢复与scope异常恢复回归：2 passed（15.65s）。
- Ruff、shell语法、git diff --check通过。

以上为5项互不重叠的针对性CPU测试，不代表CUDA/Inductor性能或GPU数值通过。下一阶段需要GPU分别获取完整分段与融合Profiler；无需先启动正式训练。

## 后续GPU定位已完成

r8完整JSON已经取回本地，20组中位1.627665→1.595167s；r9配对与两组完整Profiler完成并归档，结果见V36_FULL_PROFILER_RESULTS_20261108_ZH.md。本文此前“未归档”记录为当时状态，现已补齐。未开始正式训练，下一步可离线研究事件装配/写回及奖励反向碎片化。
