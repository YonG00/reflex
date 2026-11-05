# 按复现仓库默认调度进行联合试训

来源：https://github.com/CuboiLeo/SuperMimic-Reproduced ，当前main与本地固定提交均为`ba6105349069bf040f1fdad60873d13138d505a2`。本地独立副本 `/Users/mac/Documents/skills/仓库/SuperMimic-Reproduced`，已登记外层仓库索引，未纳入skills Git。

核对位置：`src/supermimic/config.py`、`world_model/wm_trainer.py`、`imagination/policy_trainer.py`、`training/loop.py`及`training/schedule.py`。

## 对齐的配置

- WM三层2048 SiLU，自回归窗口8，每轮10个batch，每batch2048窗口，Adam lr2e-4、norm clip1。
- 40个真实向量环境步采样之后按UTD0.4更新：控制步42/44/47/49…执行联合更新。没有几百轮WM初始化预训练。
- 每轮一次Actor Adam更新，batch4096，想象H32，lr2e-4、β=(0.9,0.95)，loss按H缩放，norm clip1之后value clip0.1。
- Critic lr2e-4，5 epochs、4 minibatches、norm clip1，γ=.99、λ=.95。
- 每个真实环境步采样一次，动作探索噪声std=.1。学习率按320000步总调度、250000步衰减终点、最终比例.03的cosine更新。TF32开启。
- 联合模式显式通过`--enable_actor`启用；不再用原预测RMSE阈值、阶段预热轮数或动作排序门控阻止Actor。有限数与物理接口检查仍会显式报错。

配置入口：`configs/world_model/joint_reproduction.json`；原guarded模式与旧默认配置保留。

## 本项目必要适配与范围

24GB 4090先使用1024个真实并行环境，而参考默认4096，因此真实交互预算不同。全局Actor batch仍为4096；按microbatch2048分两块，加权累积梯度后只进行一次clip/Adam。没有缩减全局batch或变成两次Actor更新。24GB不足时可调分块，不调整统计批量。

本项目保留Reflex接箱任务、Oracle观测/Actor网络和AMP先验；不是原MuJoCo motion-tracking框架的完整复现。纯状态82维误差均值、解析终止/奖励及数据的env-ID留出也与原复现不同。当前实现未使用torch.compile，运行速度不能直接对标原复现。critic网络对齐为256/128，Actor仍是Oracle 256/256。

真实数据的人工悬停不是物理动力学，仍排除悬停样本及外部事件跨界窗口；40步之后如没有有效8步窗口，仅继续采样并记录，不进行额外WM预训练。

## 实现检查

新增WM梯度步计数，按更新轮数验证，避免UTD时点与真实采样步模10永远错开。初次联合更新与其后每10轮进行预测验证。分块CPU测试覆盖global4、micro1/2且Adam实际step只有1；40步/UTD调度、cosine终点和验证时点均有测试。

首轮micro128试运行停在第42步首轮Actor backward，py-spy确认耗时为反向传播，未完成Actor Adam step；停止该性能诊断试运行，半批梯度未当作结果。其checkpoint40为未更新Actor的起点。修正为micro2048并开启TF32后从零重新试跑。

GPU结果以实际metrics.jsonl、manifest.json及完整checkpoint为准，不能据CPU测试宣称联合训练成功或接箱成功。

## GPU联合更新结果

`outputs/world_model/20261105_joint_repro_60steps_r2/` 已正常结束（exit0），60个真实向量步×1024环境=61,440条转移；从零的WM累计80次梯度更新，Actor累计8次，比例10:1。首个联合更新发生在控制步42。完整`checkpoint_final.pt`约528MiB，含WM/Actor/Critic/AMP/优化器/replay/解析任务状态。

约237.8秒训练墙钟（不计Isaac启动），每轮WM10次更新约1.3秒，Actor全局4096约26秒；显存观测约14.2GB。24项CPU测试通过，实际Actor梯度及损失有限。没有把首轮micro128中止尝试计为完成更新。

真实即时任务回报没有显示明确上升；8次Actor更新只证明这套联合链路运行、保存成功，不足以说明策略已学会接箱。TensorBoard目录为该产物目录下的`tensorboard/`，同时保存`metrics.jsonl`。

训练前`checkpoint_00040.pt`（Actor更新0）与训练后`checkpoint_final.pt`（Actor更新8）进行同初始seed314159、16环境×500步（每环境10秒）的真实play与录像。加载存档之后重置评测RNG，避免存档训练RNG覆盖指定评测种子。终止后自动reset，统计的是固定时间窗口，不是16个固定完整episode的成功率；策略导致的reset时间不同仍会影响后续随机数分配，不能冒充严格配对episode benchmark。

play_summary统计原始strict persistence达到30帧（0.6秒）的环境数/步数，和物理终止总数，未用宽松距离奖励冒充成功。

| 真实play指标（16环境各10秒） | 训练前/Actor0 | 训练后/Actor8 |
|---|---:|---:|
| 平均每步任务回报 | 0.0600103 | 0.0617999 |
| 物理终止总数 | 192 | 195 |
| 出现严格成功的环境数 | 0 | 0 |
| 严格成功环境步数 | 0 | 0 |

短试训没有观察到明确接箱改善；微小回报变化不能认作学习成功。此结论只针对8次Actor更新的烟测，不能据此否定长期联合训练。训练后视频：`outputs/world_model/20261105_joint_play_compare/after/videos/rl-video-step-0.mp4`；基线为同路径before目录。完整对比summary位于各自play_summary.json。
