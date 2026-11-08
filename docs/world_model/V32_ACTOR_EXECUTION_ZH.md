# V32：精简 Actor 事件输入与 pose carry（本地准备）

## 状态与范围

基于 `cba5b2e` 的独立分支 `experiment/v32-actor-gathers`。按V31实际Profiler优先处理外层事件gather及pose/state carry，不重复修改已经生效的AMP、Critic批处理或FK复用。

本轮用户要求先本地改完、再通知开卡。**未连接服务器，未启动GPU、正式训练或新checkpoint迁移。** 两项为默认关闭的独立执行开关；尚无实测GPU提速结论，也没有启用V31数值未通过的Actor/WM融合。

## 1. 事件输入依赖审计与精简

当前链路：`Trainer.imagine` → `NativeEventScheduler.step` / V30 `compact_pool_step` → `_invoke` → 原生最小视图 `fast_view.build` → 原任务回调。

| 读取位置 | 真正需要的 batch 字段 | 用途 |
|---|---|---|
| `fast_view.build` | `event_origins` | 物理状态局部/世界坐标转换 |
| 同上 | `task_history` | 持有、cradle等历史计数 |
| 同上 | `dimensions` | 原生箱体半尺寸 |
| Scheduler时钟池分支 | `event_agent` | 每个想象起点对应真实来源世界的时钟索引 |

机器人/箱子根状态、关节状态来自本步预测的Physical 84维状态；事件状态、完整来源环境时钟池、episode/common步数本来就是独立参数，继续原样索引和传递。任务名称/关节名称及实际启用回调参数仍来自scheduler原有静态信息。

最小视图明确 `del poses`，不读取Replay参考运动、765维解析context、旧观测/动作或全身FK pose。这些数据仍保留供WM、奖励、观测等原消费者使用，只取消送进事件层后立即丢弃的gather。

实现：`actor_execution_v32.gather_event_inputs()`。

- 开关 `compact_actor_event_inputs=false`：完整原路径。
- 开启且 `_view.__func__ is fast_view.build`：仅gather上述3个字段，存在来源环境时钟池时另带`event_agent`；`poses=()`。
- 自定义或被替换的 `_view`：保留全字段、全pose原始索引，不能擅自推断依赖。
- 保留active/valid/truncated选择、原生回调、随机数次数与顺序、完整时钟池、写入合并和WritePlan。没有因为本步无写入而跳过事件。

原 `runtime.py`、事件算法、`fast_view.py` 和外部写入语义未改。

## 2. pose carry 批处理

原每步对每个body的position/quaternion分别执行 `where(valid, new, old)`。开启 `packed_pose_carry` 后，每一同形状pose组执行：

```python
new_pack = stack([new[name] for name in old_names], dim=1)
old_pack = stack([old[name] for name in old_names], dim=1)
pack = where(valid[:, None, None], new_pack, old_pack)
result = dict(zip(old_names, pack.unbind(dim=1)))
```

返回接口仍是原tuple/dict，key顺序跟随old，与原逻辑一致；不改变四元数、不归一化、不detach。`unbind`一次产生全组body视图，避免逐body切片构造大量独立mask反向节点。真实机器人姿态写入后的FK、未写入时的FK复用均保持。

已审计下游 `ObservationBridge`、`UrdfFK.velocities`、Task视图及AMP：读取pose，Task中需要修改的anchor先clone，body数组由stack新建，不对这些多输出视图原地写入。

**风险和待测**：新的pose视图stride不同，packed头可能需要额外编译；stack/unbind也可能增加拷贝或反向成本。不能仅根据where数量减少就认定GPU更快。该开关默认false，必须与gather开关分开测试。

## 3. 代码位置与checkpoint

- `actor_execution_v32.py`：两个小型纯Torch执行辅助函数。
- `trainer.py`：两处分支与两个默认false配置字段，原路径保留。
- `actor_execution_v32_check.py`：同Replay、同权重/RNG的四组配对检查。
- `train.py`：新增仅诊断的 `actor_execution_v32_check` mode，必须显式指定checkpoint。
- `scripts/check_actor_v32.sh`：GPU阶段一键入口，拒绝覆盖旧输出，有超时和JSON验收。

网络结构/参数schema、Actor179、Critic305、WM3×2048/H8、Actor2048/H32、奖励、AMP、软存活、真实终止/reset均未改。两个执行开关加入现有checkpoint执行配置差异白名单，不变更权重或Adam；已覆盖旧config缺少这两个字段的恢复。开关可关闭回到旧路径，不提供或要求新的网络权重迁移。

## 4. 本地验证

新增测试覆盖：

- 只读取审计字段，完全不触及pose；全量/部分/空选择；无时钟池时无需event_agent；自定义视图回退、标量和元数据保留。
- position/quaternion逐元素完全一致，混合valid/all-invalid，非连续布局、字典顺序、空组、输入不变，以及float64输入梯度完全一致。
- H32四类回调：等待→发射、机器人外部速度写入、相同数值常量姿态写入、关节外部写入；分别覆盖部分轨迹截断与三个开关组合。
- 完整想象输出按零容差比较；有限且非零的Actor原始梯度、实际Adam参数与moment按默认容差比较，没有放宽阈值。
- 其他来源环境的时钟触发、随机数状态及依赖预测机器人位置的箱子赋速梯度。
- 带非空Adam状态的四组检查及恢复；旧checkpoint字段缺失兼容；异常回滚。
- CPU `aot_eager` packed heads入口，包括packed pose的新stride。

首轮34项通过（70.19秒），补充编译/旧存档/失败恢复3项通过（44.26秒，含旧存档重测）。最终新增36项均包含在完整回归 **209 passed、1 warning、649.44秒** 中；警告为原有TorchScript弃用提示。日志见 [CPU记录](benchmarks_v32_cpu/README.md)。CPU测试不能验证CUDA Inductor的数值或实际收益。

## 5. 开卡后的验收

```bash
export ISAACLAB_PY=/opt/conda/envs/isaaclab/bin/python
export REFLEX_DATA_DIR=/root/gpufree-data/reflex/data
bash scripts/check_actor_v32.sh \
  /tmp/reflex_v29_scratch_joint30_20261107/checkpoint_final.pt \
  /tmp/新的V32输出目录
```

需在带本分支的独立worktree运行，复用既有机器人资源。脚本固定V30 verified配置、4096真实环境、2048/H32、FP32，重新采集40个向量步后冻结Replay。不会训练WM或进行正式联合训练，不保存checkpoint。

四组：reference、event_gathers、pose_carry、combined。每组预热2次；稳态20轮，每轮轮换顺序交错测完整Actor＋Critic，避免某组固定占用不同温度/boost阶段。checkpoint、Adam和RNG每次恢复，真实Replay sampler仍执行；恢复/hash/CPU快照不在计时区间内。

`run/actor_execution_v32.json` 包含：

- 每组轨迹/事件/奖励/AMP/软终止/观测与原始Actor梯度的逐项误差。
- 实际Actor/Critic参数更新和梯度、起点SHA、模型未变及恢复状态。
- 完整更新median/P90、Actor/Critic耗时、峰值显存、独立warmup、稳态编译计数。
- 单次带trace语义rollout的forward/BPTT分项，仅作辅助，不当作稳态估计。
- 等待→发射、返回等待、截断覆盖；缺失的事件类型另补原生场景验证，不能自动认定全部通过。

`completed`、`numeric_passed`、`performance_usable`（稳态区间无重编译）及恢复检查任一失败，脚本非零退出并保留报告；各组独立结果仍可审查。即使全部通过也不自动启用正式训练，需要审查实际收益。若packed carry慢或数值超差，可只保留gather，反之亦然；无收益的开关不进入正式配置。

GPU确认胜出组后，再对该组使用V31完整Profiler核对小kernel/索引/反向计数变化；不预先生成四份巨大trace，不自动扩大H32编译范围，不运行半小时训练。
