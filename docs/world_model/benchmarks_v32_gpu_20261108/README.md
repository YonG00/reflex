# V32 GPU原始数据

实现版本 `d9e3a3b`，RTX4090 / Torch2.7.0+cu128 / FP32 TF32 off。解释与结论见 [结果报告](../V32_GPU_RESULTS_20261108_ZH.md)。

- `actor_execution_v32.json`：四组20轮交错配对，轨迹/原始梯度/实际更新最大差0，组合完整调用中位1.6424秒、同组基线1.8246秒；稳态无新编译。
- `gpu.csv`：该四组driver的遥测，未覆盖后续独立Profiler进程。
- `combined_profile.json`：组合完整Profiler及5次独立计时，不能用该进程更小的1.5695秒替代四组配对结果计算加速比。
- `combined_trace_summary.json`：排除GPU用户标记的真实设备活动统计。
- `combined_profile_config.json`：该额外诊断使用的实际配置，只新增启用两个V32开关。

组合Profiler与V31基线归档的起点/Actor/WM/RNG SHA一致；两份小型报告都包含源文件SHA。原始大trace留服务器 `/tmp/reflex_v32_gpu_20261108/combined_profile/actor_full_trace.json`（约248MB），不加入Git。

首次四组driver退出0且JSON验收通过；组合Profiler进程退出0，`completed/numeric_passed/attribution_usable`均true。未进行WM更新或正式长训练，没有覆盖checkpoint。
