# Stage 3 独立消融：准备完成，上卡验证待执行

工作副本：`ppo-stage3-ablations`，分支 `ppo/stage3-independent-ablations`。
固定原基线提交 `4138039`，原 `ppo-box-info` 工作副本未改动。实验不合并到基线。
规格来源：用户提供的 `reflex_stage3_three_independent_ablations_codex.md`。
本次授权范围是代码和本地验证；未连接 GPU、未开始短训或长训，也没有产生新的效果结论。

## 参数和开关

所有实验要求 `BOX_BUFFER_STAGE=3`，开关只接受 0/1，同时开两个立即报错。
独立实验要求原 intake=.75/retain=.5 配置，避免继承环境变量导致串组。

|组|开关|intake 权重|retain 权重|肩部动作|臂姿偏好|
|---|---|---:|---:|---|---|
|S3|三个都关闭|0.75|0.5|0.02|原样|
|A|BOX_RETAIN_X10=1|0.75|5.0|0.02|原样|
|B|BOX_DYNAMIC_BUFFER=1|6.0|0.5|0.02|原样|
|C|BOX_FLEX_ARMS=1|0.75|0.5|接球阶段 0.02→0.20|接球阶段乘 0.1|

A 仅改 retain 权重，原状态机/公式/安全门控不变。

B 独立计算 TTC（相对根部位置/速度的最近接近时间），TTC ∈ (0,.50]、预测 miss<.75m、
接近速度>.5m/s、箱子世界速度>1m/s 时，接触前最多付费 .50s。
首次有效机器人接触后最多 .35s，按首次接触年龄和独立累计预算双重限制。
接触由原 Stage3 的机器人过滤接触信号提供，不改传感器物理配置；不是 proximity 伪接触。
前后窗口均要求实际发射、非 HOLD_INIT、安全站立、未落地、未失控。
短暂脱离不会刷新接触时钟/预算；原 reset/rethrow/HOLD_INIT 钩子同时清理新状态。
同一步 context 只更新一次。dt=.02 时，后窗实际付费最多 .34s，不超预算。

B 原始奖励 = `.35*lean + .40*yield + .25*drop`，每项截断到 [0,1]：

- lean：前方来球时，躯干局部 y 轴后仰角速度 /1.2rad/s；侧方不奖励后仰。
- yield：骨盆平面速度沿箱子入射平面方向的投影 / .75m/s；反向不奖励。
- drop：骨盆下降速度 / .5m/s。下降只是屈膝代理，必须结合膝角和视频判断。
- 后窗使用首次接触前最后一帧的入射速度/来球方向，避免把碰后反向弹飞当作新的入射方向。
- 不调用旧 intake 的 dist<.9 门控。姿态恢复沿用原控制/稳定恢复逻辑，只将 B 的接球窗口接入放宽门控。

C 肘部始终 .35；肩部采用启动 5s × 每次进入接球窗口 .25s 的渐进 ramp，达到 .20。
等待/未发射阶段肩部仍 .02，保留原 two-stage 准备动作、参考动作及 gimbal 控制。
启动 ramp 同样适用于评测起始阶段，比较结果需保留该配置，不能偷偷关闭。
C 的动作阶段查询读取上一控制步状态并核对当前发射/相对位置，不提前调用任何环境事件。
新增肩部目标物理限位裁剪；记录目标变化/裁剪量、实际肩肘活动范围。
接球阶段将 `cradle_pose/cradle_ready/cradle_ready_dev/box_on_bed` 乘 .1（权重原值不改）；
`hold_still` 保留原相对速度与 held 门控，仅解除接球阶段 arm-freeze 乘子。
等待阶段这些函数保持原样。没有新增挥臂幅度奖励。

## 不变量与已有约束

- Actor183/Critic305、PPO/AMP、观测、训练入口和训练主循环保持原样。
- `buffer_control_core.py` 未改；所有初始化、hover、赋速、发射、重投、HOLD_INIT 回调及顺序未改。
- 原 strict 成功判据、终止条件、站立基础安全、运动跟踪、AMP 均未关闭。
- `anchor_pos` 终止和参考运动/AMP 仍可能限制后退、迈步、大幅屈膝；C 的 two-stage/gimbal 仍是动作先验。
  本轮只检验各独立假设，不能预先将这些限制全部取消。
- `escape_penalty/impact_buffer` 在实际配置保持未注册；不恢复旧速度/冲击惩罚。

## 共享评估口径

`--physical_hold_metrics` 仅开启评估统计，四组均通过同一入口使用；不反馈到训练奖励。
原 strict 不改，另保存每投 `caught_loose`。

`physical_hold_success` 要求连续 .6s 满足：

1. 实际发射且非 HOLD_INIT，有真实手/腕/肘/前臂接触；最近支持接触不超过 .12s。
2. 箱子底面离地>.08m，无地面接触（首次落地本投锁存失效）；相对躯干半径<.75m。
3. 箱子相对躯干速度<.5m/s（包含躯干旋转速度），相对角速度<2rad/s。
4. 根部>.45m、躯干>.55m、躯干 up_z>.25；当前没有头/肩过滤器接触。
5. 接触后逃出1.2m范围，本投锁存失效；不能仅靠近身体、仅胸部撞击或夹住落地箱子成功。

允许托举、双侧夹持、胸臂协作，不使用双腕固定几何结构。
每投输出受控时间、最长连续受控时间、真实臂接触、双侧臂接触时间、头肩接触时间、
最大后仰角、首次接触姿态/骨盆速度/膝角、肩肘活动范围、ground/escape。
双侧臂接触时间只是夹持代理，不等于已经学会主动夹持。

汇总固定 seed8/9，每 seed320 环境，仅 idx=1 首投计分；缺任何首投直接报错，重投数量单独报告。
strict/loose/physical_hold/contact/fall/ground/escape 都用相同首投分母。
历史71/648与109/648只作参考，不能与本次640首投直接比较。

## 日志

- `Episode_Reward/*`：原 RewardManager/TensorBoard episode 统计。
- `Buffer/*_raw_mean`：当前控制步全环境原始均值。
- `Buffer/*_reward_mean`：上值乘实际配置权重，未乘 dt。A 的日志使用5.0，而不是误记成.5。
- JSONL：throw_end、B首次接触记录、窗口累计秒数/投掷和触发比例、子奖励均值/非零比例、C有效姿态权重。
- `invalid_positive_intake/retain`：无效安全/地面状态仍付费的审计计数，短测要求为0。
- 视频复用已有 `disable_debug_visualization`，关闭坐标轴/接触调试标记。

## 已执行的本地验证

2026-11-10，macOS CPU，`/private/tmp/reflex-escape-venv/bin/python`：

```bash
PYTHONPATH=source/reflex /private/tmp/reflex-escape-venv/bin/python -m pytest tests -q
```

118项通过（原73项＋新增45项）。环境未安装NumPy，Torch发出一次初始化警告，不影响本次纯Tensor测试。
包括四组实际配置隔离、双开关拒绝、B预算/缓存/重投/HOLD_INIT/侧方方向、
C等待动作与肘部不变/肩部ramp/限位/姿态衰减/hold_still速度保留、共享评估支持接触/落地/头肩/跌倒/重投。

关闭开关时，从固定提交4138039加载原 `buffer_control.py`，与新适配器运行130步相同轨迹，
包含接触、落地、重投、HOLD_INIT，逐项 `torch.equal` 比较所有输出及累计计数，无差异。
另外检查原 core 和 teacher/train.py 字节不变。
此证据是CPU函数/状态兼容性，**不代表GPU物理轨迹/训练结果已经零差异验证**。

新增脚本 bash -n、Python编译及 git diff --check 通过。
GPU物理、真实RewardManager装载、C热启动目标跳变/步态稳定、实际学习效果均待开卡短回归。

## 开卡后的执行顺序（尚未执行）

先确认服务器代码分支/提交、Isaac Lab依赖、数据/URDF/AMP文件、GPU/磁盘、baseline599路径与SHA256。
本地Mac不能替代服务器的Isaac Sim回归；不伪造服务器checkpoint路径。

```bash
# 在此工作副本根目录，设置实际的Isaac Lab Python和数据路径
export ISAACLAB_PY=/actual/isaaclab/python
export REFLEX_DATA_DIR=/actual/reflex/data
bash scripts/run_s3_ablations.sh short /actual/model_599.pt outputs/s3_short_unique
```

串行 S3/A/B/C，每组都从同一599独立 native resume，128env×12次额外更新。
checkpoint预检强制iter599、actor183/critic305和PPO优化器状态，不进行padding或静默跳过不兼容权重。
调用原runner.load，不修改AMP恢复行为；AMP历史状态是否由当前runner序列化须上机核对并记录，
不能把原warm_start（只复制策略）与native resume混作同一训练协议。
因此本轮同时新跑S3对照，不直接借用历史S3评价实验优劣。

实际迭代日志必须恰为599..610（12次）；正式训练要求599..798（200次）。
若服务器rsl_rl语义不同，检查器拒绝验收，先核对版本再继续，不为凑轮数改训练主循环。
结果检查还要求有效奖励/缩放触发、审计为0；进程异常立即停止队列。
GPU短测后人工检查关节目标突变、倒地、奖励投机、初始化事件与实际配置，确认后再单独调用：

```bash
bash scripts/run_s3_ablations.sh full /actual/model_599.pt outputs/s3_full_unique
```

正式入口重新从同一599运行四组4096env×200更新，依次各做seed8/9评测和无坐标轴视频。
不会接着short的权重训，也不会用A权重训B。
完整实验输出 `comparison.json`、各组checkpoint路径、训练/评测日志与视频。
有待处理的短测异常时不要调用full。若B权重6导致异常，保留失败记录，再独立建立降权版本。

## 回滚与采用

关闭三个开关回到S3；从原`ppo-box-info`也可直接复现未改基线。
任何组合都需要新的消融，不采用尚无两seed一致收益的方案。
本次只有本地实现/CPU验证完成；GPU短回归、长训、评测结果和行为审阅均为待执行。
