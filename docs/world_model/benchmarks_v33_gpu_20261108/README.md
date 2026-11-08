# V33 GPU诊断原始结果

2026-11-08，4090/PhysX，4096真实环境，Actor2048/H32，FP32，V32执行基线。没有正式训练或新策略效果结论。

- `execution_study_v33.json`：整体诊断门控，completed/numeric/attribution/timings通过。
- `actor_full_v31.json`：完整更新、轨迹/梯度/Adam验证及源码点归属。
- `actor_full_trace_summary.json`：真实设备活动统计，不混入GPU用户标记。
- `phases.json`：20次连续基线与20次连续粗阶段记录，无每轮snapshot恢复。
- `native_phases.json`：补充实际事件step区间；事件区间与actor_forward嵌套，不能相加。基线完整更新中位1.5630035秒，记录组1.5661971秒，全部梯度/参数max_abs=0、编译delta=0。
- `gpu.csv`：主诊断遥测。

完整trace留服务器`/tmp/reflex_v33_gpu_20261108/run/full/actor_full_trace.json`。补充分项driver退出0、JSON已取回本地，后续可离线分析。排队的可选No-FMA实验取消，没有其GPU结果。最后确认GPU92MiB/0%，没有继续训练。

解释及V34下一步见`../V34_EXECUTION_ORGANIZATION_ZH.md`。本轮无新增加速，不拿跨测量方法的中位数变化算收益。
