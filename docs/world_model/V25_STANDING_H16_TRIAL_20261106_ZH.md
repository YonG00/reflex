# V25：站稳优先奖励 + H16 半小时续训

用户明确授权提高站立奖励/跌倒惩罚、降低动作跟踪权重、H32改H16，并从此前模型继续。
独立消融，不宣称仍是原Reflex奖励或完全对齐第三方配置。

## 配置

configs/world_model/joint_standing_priority_h16_v25.json。
- 新standing_balance权重5，所有阶段生效。直立分数与相对参考anchor高度分数相乘，
  height_std=0.12m，upright_std=0.2；真实环境与想象共享同一Torch函数。
- 六项motion_*跟踪奖励乘0.25：anchor pos/ori 0.5→0.125，body pos/ori/lin_vel/ang_vel 1→0.25。
- fall_terminated -50→-150：dt0.02时一次性成本-1→-3；纯软想象仍用C*p_done及存活率加权。
- torso_uprightness_penalty 0→2；soft_fall_shaping -2→-4。
- Actor2048起点、H16（0.32秒）、WM3×2048/H8/每轮10次更新，AMP/碰撞/终止阈值不变。
- 4096真实环境，updates_per_env_step=0.4，已验证采集优化启用，Replay容量256。

## 恢复与监督边界

源是V24纯软半小时最终模型：Actor340、WM/Termination3400次更新。
明确--migrate_standing_profile仅允许原H32→standing_h16_v1/H16，校验具体奖励修改，
不允许顺便修改物理任务、终止或事件规则；Actor/WM/Critic/AMP和所有优化器保留。
重建WM Replay，避免旧权重奖励列混入；新profile经验统计重新计数。

当前磁盘不适合反复保存大Replay，save_replay=false仅不保存WM Replay。
模型、全部优化器、AMP判别器及AMP数据/状态照常保存；无WM Replay的存档若不显式
--reinitialize_replay恢复会拒绝，不静默缺失。原V24存档保留未删除。

## 验证与启动

CPU73项通过（217.02s），含站稳奖励高度梯度方向、倾斜抑制、无Replay恢复拒绝/显式重建。
4096真实环境预检32步通过：观测1.56e-5、AMP2.43e-5、Critic2.51e-5最大差，
standing_balance真实/解析奖励最大差2.01e-7；H8有效窗口101139，合法起点104832。
预检未更新参数，preflight.pt作为相同权重/优化器的启动源。

已启动outputs/world_model/20261106_v25_standing_h16_joint_30min/，进程95243（启动PID）。
预算1800秒，含训练循环冷编译；每500采集迭代与结束保存模型。tensorboard/、metrics.jsonl、run.log
记录真实奖励、回合长度、终止原因及实际更新时间。没有运行8192环境。
这只记录已启动；是否站稳必须检查训练后真实回合长度和独立play，不能用新奖励总值直接比旧奖励。
