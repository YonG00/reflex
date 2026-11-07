# V29 从零：500轮WM预热 + 30分钟联合训练

## 用户授权与状态

2026-11-07用户明确授权从0开始，随后联合训练半小时。AMP与Critic批处理均开启。训练已于服务器Asia/Shanghai 14:19:37启动；初始 `actor_initialization.kind=scratch`，没有checkpoint/warm-start参数，Actor/WM/Critic/AMP判别器重新初始化。

本记录仅确认启动与预热正常，尚未确认训练完成或站立/接箱效果。最新现场快照：采样迭代393、WM轮141/500、Actor更新0；正式联合计时尚未开始。

## 有效设置

- 代码70b5f2e；配置 `configs/world_model/joint_efficiency_v29_compact_analytic305_batching_512_trial.json`。
- PhysX/RTX4090，4096真实环境，Replay512步/环境，Actor2048/H32，WM3×2048/H8，每轮WM梯度更新10次，UTD0.4。
- 精简WM context127/首层246维；Actor179维、Critic305维，隐藏结构和原奖励维持当前候选。
- `batch_imagination_amp=true`、`batch_imagination_critic=true`、`analytic_soft_termination=true`；无Learned Head/BCE更新。
- 前500轮WM训练期间不更新Actor/Critic，真实采样与原AMP训练继续；第501轮起允许Actor更新。
- `--joint_max_seconds 1800`：第一轮Actor更新完成后才开始1800秒联合计时，排除WM预热和首次Actor冷编译。整体进程有5400秒超时保护。
- `--save_every 500`按向量采样迭代保存；另外保存 `checkpoint_wm_warmup.pt` 与 `checkpoint_final.pt`。
- `save_replay=false`为薄存档，保留模型、优化器和AMP状态。未来恢复需显式重建Replay，不能将其当成完整仿真状态恢复。

## 输出与完成后评测

训练实时目录：`/tmp/reflex_v29_scratch_joint30_20261107/`。

- TensorBoard：`/tmp/reflex_v29_scratch_joint30_20261107/tensorboard/`
- 逐采样迭代指标：`metrics.jsonl`
- 阶段计时：`training_phases.json`，含预热结束、首轮Actor后计时起点和最终联合秒数。
- 启动/控制台日志：`/root/gpufree-data/reflex-v29-test/outputs/world_model/20261107_v29_scratch_joint30/train.log`

完成后启动器自动将最终模型、预热模型、指标、TensorBoard和manifest复制到上述 `outputs/world_model/20261107_v29_scratch_joint30/` 归档目录。训练成功后另有评测等待脚本录制最终模型15秒视频，seed42、单环境、隐藏坐标轴；目录 `final_play15s/`。必须核对train.exit、failure.txt、training_phases与视频解码结果，不能把“进程启动/文件存在”当作训练/评测通过。

该实例释放后数据可能丢失，正式结束应继续下载关键模型、TensorBoard及视频到本地。当前没有自动关机或释放实例操作。

## 测试与估计

独立联合计时的边界测试及原有效率调度测试14项通过。现有完整回归在追加计时前122项通过。

近期预热约1.12秒/WM轮（含按UTD采集及定期验证），正式联合预计430～500次Actor更新/30分钟，实际由训练结束计数确认。旧H32半小时约340次更新的记录包含冷启动，非严格同计时口径；更新次数增加不等于站立或接箱效果已经改善。
