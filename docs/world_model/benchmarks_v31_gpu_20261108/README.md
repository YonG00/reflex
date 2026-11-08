# V31 GPU 原始数据

环境、解释、失败项及下一步见 [结果报告](../V31_GPU_RESULTS_20261108_ZH.md)。

- `actor_full_v31.json`：完整链路20次无Profiler更新、Profiler关联work、轨迹/梯度/Adam检查。**初版 `attribution.device_timeline` 混入GPU用户范围，应弃用该子字段。**
- `actor_full_trace_summary.json`：同一原始trace按真实kernel/memcpy/memset重新分析的活动并集、小kernel、同步和拷贝统计，是timeline的修正版。
- `reflex_minimal_r2.json`：Reflex最小链路，编译组严格数值检查未通过，不能视为可合入优化。
- `tracker_minimal_r2.json`：实际服务器第三方原生网络的最小链路，三组通过；合成输入/权重，不是完整任务复现。无Git元数据，报告含源码SHA。
- `gpu.csv`：首轮driver遥测；R2在其退出后单独运行，不包含R2。

完整源码40f3f80，诊断修正ceb15e1。R2沿用同一个Reflex capsule，未重新采集或替换起点。大型trace与45MB capsule留在服务器 `/tmp/reflex_v31_gpu_20261108/full/`，不加入Git。

首轮driver退出1：Reflex新编译边界数值未通过且第三方缺少Git元数据。修正后的两个R2 CLI均退出0，但CLI完成不代表数值通过；以报告的 `numeric_passed` 为准（Reflex false、TrackerLab true）。
