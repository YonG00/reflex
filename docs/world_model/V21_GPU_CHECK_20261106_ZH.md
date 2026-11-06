# V2.1 GPU 验证与首次联合训练记录

服务器 RTX 4090 24GB，Torch 2.7 / Isaac Lab 2.3.1。时间以客户端 2026-11-06 为本轮日期。

## 已完成

- 真实环境 parity：64 环境 × 128 步，Actor/AMP/Critic 最大误差约 1.75e-6/2.21e-6/2.76e-6。
- 冒烟：一次 Actor、10 次 WM、10 次终止头更新，保存 v3 checkpoint。
- Dynamics/FK 独立 CUDA 编译输出及梯度一致性通过。
- 按用户要求关闭新增箱体接触监控；保留真实物理碰撞、原接触奖励与传感器。

## 首轮半小时尝试：中断，不能称为完成

目录 `outputs/world_model/20261106_v21_joint_30min`。1024 环境，从冒烟 v3 存档恢复。Actor batch2048、H32；首次 Actor 更新含编译 177.706 秒，后续大多约 5 秒（有一次 6.691 秒）。10 次 WM 更新约 0.49 秒。

最后完整日志 iteration166，累计 Actor50、WM500、终止头500；本次约531.6秒。3072+结束episode，255结束throw，CLEAN2几何拦截率53.33%，严格抱持0%，原宽松catch59.61%。这些是训练过程统计，不能替代固定种子独立评测，不能声称接箱效果达标。

在下一次更新前的梯度诊断出错：Torch2.7 compiled donated buffers 不支持 retain_graph=True。不是 Actor 正常单次反向传播出错。仅保留 `checkpoint_00100.pt`，没有 final 存档；恢复会回到该存档，而不是最后日志的第50次Actor更新。

## 修复与扩大编译验证

- 只读梯度诊断临时走 eager，finally 恢复训练编译函数；不更新优化器。
- 候选新增单步 WM+FK+Reward+Termination 融合，以及单独 Actor/Critic Observation 编译；维持真实 timestep、初始参考缓存和历史更新。
- `compile_step_check` 在同 replay 上比较输出/Actor梯度，并分别 warmup 后测2048起点、H32完整BPTT；不执行optimizer.step。
- CPU 全套44项通过（含 expanded aot_eager 轨迹/Actor梯度一致性）；服务器 CUDA 实测仍待完成。未确认提速前不将候选作为正式配置。
