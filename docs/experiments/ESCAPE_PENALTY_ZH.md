# 严重弹飞失败惩罚 escape_penalty（每投掷一次）

当前实现位于 `ppo/box-info-19d`，保留官方 PPO+AMP 与183/305维箱子信息扩展。旧连续 impact_buffer 项已从 teacher 入口退役，即使继承 BOX_IMPACT_BUFFER=1 也不注册旧奖励；旧文件仅留作历史复现。新模式不叠加接近速度、减速、接触力或固定姿态惩罚。

## 触发与实际奖励

必须真实发射、非 HOLD_INIT，并在当前投掷中检测到箱子与机器人刚体的物理接触。之后距离躯干 >1.0m 且径向远离速度 >1.0m/s，连续0.15s才触发（50Hz下8步=0.16s）。条件中断立即清零计时，允许短暂反弹、脱手再接和切向运动。

首次触发实际 task reward 扣 **-100**，当前投掷 `env._escape_state.escaped=True`，后续不重复扣，重新接住不退款。按用户最新指示，**每次投掷最多扣一次，不是每个 episode 一次**；同episode真实重投后清除 escaped/contact/dwell/catch 状态，新投掷再次碰撞后失控可以再扣100。episode reset同样通过原reset_box_throw回调清理；HOLD_INIT排除，直到下一次真实重投。

RewardManager会对每项乘dt，因此函数返回 `-100/dt`，外层 RewTerm weight=+1，确保实际单次扣100而非2。原 AMP_W_TASK=1；若未来单独改AMP任务混合权重，最终混合奖励也会相应缩放。

相对速度考虑躯干COM线速度和 `omega × (p_box-p_com)`，再转入躯干坐标。相对距离与径向速度不受机器人和箱子共同平移/旋转影响，不限制姿态、手臂方向或缓冲动作。判定失控的是相对运动，而非箱子世界系速度。

## 接触与原有事件

复用已验证的过滤接触力子步历史传感器。新传感器在初始化时枚举实际机器人RigidBodyAPI刚体，每个刚体独立filter，避免正反接触力合并抵消。箱子作为唯一reporter，不包含地面、其他箱子或无刚体的mesh；胸部、腿和手臂碰箱子都属于真实机器人接触。1N只用于检测接触，不是接触力约束或惩罚。

不新增、重排、包装或跳过原有事件，不改变箱子初始化、悬停、赋速或发射。只扩展之前的记账回调，按当前模式路由到escape_throw_reset；真实重投时先记录旧投掷，再清理新投掷状态和接触历史。

## 原机制检查（仅报告，未修改）

- AMP风格权重1、任务权重1；特征含关节/速度、根部高度/姿态/速度、腕/踝相对位置，偏离示范可能降低风格奖励。
- 位置/姿态/身体速度跟踪奖励仍存在；如anchor位置std=0.3、body线速度std=1.0等，会给自由走动或大幅缓冲施加代价。
- BOX_ARM_SCALE=0.02、BOX_WRISTLOCK=1、BOX_GIMBAL=1、两阶段手臂控制均保留，确实限制完全自由的自主手臂动作。
- BOX_POSTURE=1、BOX_V19C=1：原torso_asym项后仰阈值约18°，权重-0.5。
- 根部/脚踝z偏差0.25m、projected-gravity姿态偏差阈值0.8等终止条件仍可能打断大幅倾斜/下蹲；它们不是平面行走的直接硬限制。
- 原fall_terminated项权重-50，但dt=0.02下该项实际单次为-1；不代表全部跌倒总代价只有1。新的实际-100比该单项更大，提前跌倒/提前终止规避惩罚是必须检查的风险。本次未改AMP、姿态项、终止条件、动作控制、观测或训练主流程。

## 开关与系数

新模式入口：`bash scripts/train_stage1_escape.sh [原一阶段参数]`。脚本设置BOX_ESCAPE_PENALTY=1、BOX_IMPACT_BUFFER=0，再调用原箱子信息PPO入口。

- BOX_ESCAPE_AMOUNT=100
- BOX_ESCAPE_DISTANCE=1.0（m）
- BOX_ESCAPE_RADIAL_SPEED=1.0（m/s）
- BOX_ESCAPE_PERSISTENCE=0.15（s）
- BOX_ESCAPE_CONTACT_FORCE=1.0（N，仅接触检测阈值）
- BOX_ESCAPE_LOG_EVERY=50
- BOX_ESCAPE_LOG_PATH（默认outputs/escape_penalty/escape_<PID>.jsonl）

关闭新项、保留其他原奖励：`BOX_ESCAPE_PENALTY=0 BOX_IMPACT_BUFFER=0 bash scripts/train_stage1_box_info.sh ...`。旧pre×20分支是历史实验，不自动改写其历史结果。

## 独立日志与规避检查

Escape/escape_penalty为实际应用前的每环境事件奖励均值（已按单次-100口径记录，函数返回率则补偿dt）；Escape/escape_rate、contact_rate、catch_success_rate按真实投掷计数，HOLD_INIT排除，在飞投掷也进入分母。严格接住使用原strict_all连续0.6s。

独立JSONL周期快照包含上述指标和throws/contacts/escapes/catches计数；触发时立即写event=escape，包含env_id、throw_id、episode_step、实际-100、距离、径向速度、持续时间，不因周期日志错过一次性事件。重投前写throw_end，保留escaped/contacted/caught与终止信息。

记录no_contact_terminated、contact_terminated_before_escape、pending_escape_terminated和terminated_without_recorded_launch，检查躲避、尚未发射就终止、接触后但确认前结束等路径。只记录，不偷偷添加新惩罚或修改终止条件。该机制无法数学上排除所有规避；若接触/接箱率下降、提前终止增多而escape_rate降低，不能称作改善。catch后仍可escape、escape后仍可catch，两种事件率不互斥，catch不会撤销已扣分。

## 本地验证及文件

50项联合CPU测试通过（新项29项+原impact16项+观测5项）；包含每投一次、同episode重投再次扣分、重新接住不退款、悬停/无接触不触发、短暂反弹、条件中断计时清零、dt补偿、部分reset、HOLD_INIT、全机器人过滤、原reward/event保留和规避诊断。Python编译、shell语法、git diff空白检查通过。CPU夹具不等价于真实物理训练。

入口：`python scripts/check_escape_penalty.py --out outputs/escape_penalty_cpu_short_20261110`，只运行人工轨迹。本地环境为临时CPU PyTorch环境，不使用GPU。

核心文件：escape_penalty_core.py（状态机）、escape_penalty.py（运动学/回调/日志）、escape_penalty_sensor.py（复用子步历史并枚举机器人刚体）、escape_penalty_config.py（只注册新reward）；teacher_env_cfg.py/throw.py/impact_buffer.py为小范围模式与记账改动。tests/test_escape_penalty.py与两个scripts/check/train入口用于验证和使用。

GPU验证：按用户最新授权，待在空闲RTX4090上运行128环境、20轮有界短训，从昨晚model_599.pt warm start；不自动扩成长训。结果完成后在本文件补录。
