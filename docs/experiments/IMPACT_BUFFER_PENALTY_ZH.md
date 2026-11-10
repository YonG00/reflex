> 历史第一版：当前PPO teacher入口已退役该连续惩罚。新模式见 [ESCAPE_PENALTY_ZH.md](ESCAPE_PENALTY_ZH.md)，不要将两者叠加。

# PPO 接箱碰撞缓冲惩罚：第一版

## 范围与当前状态

已迁移到独立副本 `reflex/ppo-box-info` 的 `ppo/box-info-19d` 分支，基于官方代码和已验证的箱子信息扩展（actor 183 / critic 305）。未带入主仓库的 WM 改造，未改变网络架构、其他奖励或训练参数，没有启动长训练。本次按用户后续指示只做本地实现和短测，等开卡再验证真实 PhysX/PPO。

新增 `impact_buffer_penalty`，默认关闭。启用后注册一个 `RewTerm(weight=+1.0)`；函数内部返回三项非正值之和，**外层不要再用负权重**。原来的 `hold_still_reward`、`box_on_bed`、其他奖励、动作和终止条件均保留。

## 三项计算

1. **接触前**：仅真实发射、未首次手臂接触、箱子与最近手/前臂表面估计间距 <0.8m 且相对接近速度 >0 时生效。候选部位是双侧肘→腕连线的最近点和腕端手部区域；速度取对应刚体在该点的速度，包含角速度贡献。间距考虑当前箱子姿态、半尺寸和手臂厚度。惩罚为 `-w_pre * proximity * (approach_speed/8)^2`，临近时渐强，不要求托举或夹持姿态。
2. **接触后**：首次过滤力超过 1N 的箱子—手臂碰撞后，使用 `v_box - (v_torso_com + omega_torso × (p_box-p_torso_com))`，再旋转到躯干坐标系。惩罚为 `-w_damp * ramp * (relative_speed/8)^2`；ramp 在 0.1s 内由 0→1，0.5s 后关闭。
3. **弹离**：首次碰撞时存储躯干坐标下的箱子位置。允许半径为 `max(first_radius+0.08, 0.65)`，超出部分除以 0.6m 后平方，乘 `-w_escape`；只在碰撞后 0.5s 内有效。仅计算向外增大的径向距离，不限制手臂姿态。机器人与箱子共同平移/旋转不改变该项；如果机器人主动远离箱子而相对距离增大，仍会罚，这是相对失控而非世界坐标漂移。

三项不做速度上限饱和，避免超过归一化尺度后惩罚不再增加。实际总奖励仍遵循原 RewardManager 的 `dt` 缩放。

## 接触与生命周期

现有 `contact_forces` 是机器人总接触力，不能准确识别碰撞对象，因此不能用它判断首次手臂接触。新增一个箱子作为唯一 reporting body 的 ContactSensor，过滤目标仅为双侧 elbow、wrist_roll、wrist_pitch、wrist_yaw，共 8 个刚体；不包括胸部、肩膀或地面。箱子仅开启 contact reporting，不改碰撞形状、质量、材质或求解器。

`ImpactPairContactSensor` 保留每个 physics substep 的**过滤力**历史，长度等于 decimation，避免只采最后子步漏掉高速反弹。没有过滤矩阵时明确报错，不退回总接触力。真实 Isaac Lab 的 callback/传感器匹配与子步更新行为仍待 GPU 验证。

不新增、不重排事件，也不包装或跳过原事件。只在原 `reset_box_throw` 的全部物理写入之后清理对应环境的状态/接触历史（同时覆盖初始 reset、realign 与重新投掷）；在原 `hold_init_apply` 写入之后标记 HOLD_INIT。HOLD_INIT 不算发射/首次有效碰撞、不产生本项惩罚，直到下次真实重投重新启用。每投只记录一次，部分环境 reset 不清掉其他环境，重复读取同一个 control step 不重复计时或计数。

首次接触速度使用**前一控制步**箱子相对预计手臂区域的速度，避免把冲量后已减小的速度称为冲击前速度。这是控制频率近似，不是精确碰撞瞬间的子步速度。首次没有前样本时使用当前样本。

## 开关与参数

关闭（默认，baseline）：

```bash
BOX_IMPACT_BUFFER=0 bash scripts/train_stage1_box_info.sh --num_envs 64 --max_iterations 2 --seed 42 --run_name impact_off_short
```

开卡后的短测示例（**本次没有执行**）：

```bash
BOX_IMPACT_BUFFER=1 \
BOX_IMPACT_W_PRE=0.2 BOX_IMPACT_W_DAMP=0.5 BOX_IMPACT_W_ESCAPE=1.0 \
BOX_IMPACT_WINDOW=0.5 BOX_IMPACT_LOG_EVERY=1 \
BOX_IMPACT_LOG_PATH="$PWD/outputs/impact_on_short.jsonl" \
bash scripts/train_stage1_box_info.sh --num_envs 64 --max_iterations 2 --seed 42 --run_name impact_on_short
```

两条短测命令显式指定同样的环境数、轮数和 seed，只切换新项；未改配置文件中的默认训练参数。本项在共享 teacher 配置中启用，box-info 任务继承该入口。默认关闭保留之前的 183/305 维 PPO 对照；开启只增加碰撞缓冲项和过滤接触传感器。原版 170 维任务也可通过原启动脚本选择，但对比时应固定同一个任务和观测维度。

其他可调环境变量：`BOX_IMPACT_RAMP`=0.1、`BOX_IMPACT_PRE_RANGE`=0.8、`BOX_IMPACT_SPEED_SCALE`=8、`BOX_IMPACT_ESCAPE_SCALE`=0.6、`BOX_IMPACT_CONTROL_RADIUS`=0.65、`BOX_IMPACT_FREE_DISPLACEMENT`=0.08、`BOX_IMPACT_CONTACT_FORCE`=1。所有权重非负、尺度和窗口必须 >0。

## 独立日志与规避检查

默认 JSONL：`outputs/impact_buffer/impact_<PID>.jsonl`，每 50 control steps 写快照；`BOX_IMPACT_LOG_PATH`、`BOX_IMPACT_LOG_EVERY` 可覆盖。同时 stdout 标记 `[impact_buffer]`，并写入 `env.extras['log']` 的 `Impact/*` 命名空间。JSONL 为保底可复查来源；TensorBoard 对 extras 的聚合行为需上机确认。

- pre_approach_speed：当前 pre_gate 内的平均相对接近速度；记录有效样本数量 pre_active。
- first_contact_speed：当前已接触投掷的首次相对速度均值；每投关闭时另存原始速度。
- post_relative_speed：当前 post_gate 内相对躯干速度均值；记录 post_active。
- max_escape_distance：本进程观察到的最大向外位移；mean_throw_max_escape 为当前已接触投掷的最大位移均值。
- penalty_pre / penalty_damp / penalty_escape / penalty_total：全环境步平均，包含未启用环境的零；是内部加权、RewardManager `dt` 缩放前数值。
- throws / contacts / strict_success 及 first_arm_contact_rate / strict_catch_rate：累计真实投掷分母，HOLD_INIT 排除；严格接箱使用现有 `strict_all` 连续 0.6s，每投只成功计数一次。在飞投掷也进入分母，因此训练中是实时、未完全结算的比例，不能当独立 CLEAN2 最终成绩。
- closed_throws / missed / near_missed / early_terminated / post_window_truncated：漏接、曾靠近躯干但未手臂接触、投掷期间终止、接触窗口尚未结束就 reset/rethrow 的诊断计数。
- reset/rethrow/HOLD_INIT 时写 `event=throw_end` 的逐投记录，保留接触状态、首次速度、最大弹离位移、接触年龄和 episode_step，避免短交互完全落在两个快照之间。

**已确认的激励边界**：三项都是负奖励且要求临近/接触，故完全躲开或提前终止可能减少惩罚。CPU 短测显式覆盖并记录该现象，没有偷偷增加漏接奖励、存活奖励或调整终止条件。是否真的诱发策略规避，必须在 GPU 对照中比较原版成功率、首次手臂接触率、near_missed、终止类型、episode 长度与视频；短测不能证明不存在 reward hacking。如果接触率/成功率下降而新惩罚变小，不能称作缓冲改善。

## 本地验证

```bash
PYTHONPATH=source/reflex python -m pytest tests/test_impact_buffer.py -q
python scripts/check_impact_buffer.py --out outputs/impact_buffer_cpu_short_20261109
```

2026-11-09：16 项 impact CPU 测试通过；迁移到 PPO 分支后，与既有 5 项箱子观测测试联合运行共 21 项通过；Python 编译与 `git diff --check` 通过。覆盖权重/尺度校验、接近速度/距离单调性、窗口和渐强、首次接触计数、前样本冲击速度、尺寸/方向相关区域速度、躯干旋转及刚体共同运动不变性、短暂过滤力历史、部分 reset、同一步复用、HOLD_INIT、重投、严格成功持续判据、独立日志和已有事件/奖励保留。

短测为 6 条人工指定轨迹、100 步、dt=0.02s：缓冲接住、高速反弹、躲避漏接、提前终止、HOLD_INIT、仅地面/胸部碰撞（不进入箱子—手臂过滤力）。合成场景中 5 次投掷、2 次有效手臂接触、1 次严格成功、1 次提前终止；最大弹离 1.25m，三项均值约 -0.02953/-0.01078/-0.04392，总均值 -0.08424。这些是**测试夹具统计，不是机器人训练成绩**。

产物：`outputs/impact_buffer_cpu_short_20261109/samples.jsonl` 和 `summary.json`。未运行真实 Isaac Lab/PhysX 短训、未验证物理过滤匹配或学习效果，按用户要求等待开卡。

## 文件与回滚

- `impact_buffer_core.py`：纯张量三项计算、每投状态和统计。
- `impact_buffer.py`：机器人运动学、PPO reward 适配、生命周期记账及独立日志。
- `impact_buffer_sensor.py`：过滤接触力子步历史。
- `impact_buffer_config.py`：可选传感器与单个奖励项配置。
- `teacher_env_cfg.py`：默认关闭的配置入口。
- `throw.py`：两个原函数中的少量状态记账。
- `tests/test_impact_buffer.py`、`scripts/check_impact_buffer.py`：本地测试与短测入口。

最快关闭：`BOX_IMPACT_BUFFER=0` 或 unset。此时不注册新传感器/奖励、记账分支不执行；原有默认运行不受影响。排查入口为独立 JSONL、throw_end 记录及配置中的 impact_box_arm 过滤列表。
