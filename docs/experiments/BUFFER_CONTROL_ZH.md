# 接箱正向缓冲奖励：三个独立阶段

## 范围和当前验证状态

当前 `ppo/box-info-19d` 分支，原actor183/critic305不变，不修改PPO/AMP主流程。以正向缓冲设计取代旧impact及-100 escape项。完成本地代码、73项联合CPU测试和三阶段真实GPU短回归，另完成地面碰撞过滤审计。此前SSH连接中断，用户重新开卡后已恢复执行；未将人工轨迹当物理验证。随后用户要求正式开始训练，目前只启动阶段1的200轮正式对照，阶段2/3尚未启动长训。

用户引用的42%接触率、0.31%成功率来自训练累计日志，不是CLEAN2最终持续接住率13.3%；本版本继续区分训练诊断与独立评测，不混用分母和成功判据。

## 检查到的原有约束（保留，未一次性关闭）

- AMP_SLIM=1确实将compliant_intake清零；本配置在teacher的所有父类和AMP_SLIM/反应门控处理结束后应用，避免再次被清零。
- GATE_ALL的gbx起止值-0.3/-0.15，floor0.1，约17°后仰时就压至约10%；upright_hold偏好gbx=-0.15（约8.6°）。本次只对有效缓冲窗口放宽这些姿态门控。
- torso_asym约18°后仰、12°前倾、20°侧倾阈值，权重-0.5。新包装保持前倾/侧倾代价，后仰项仅在稳定持有后的恢复阶段渐入。
- anchor_pos是根部z偏差0.25m终止，不是平面走动距离终止；但限制大幅下蹲。脚踝z偏差和姿态projected-gravity终止亦保留。
- 位置、姿态、身体速度跟踪奖励会压制偏离参考的大幅缓冲/迈步；AMP仍偏好示范动作。未改相关权重或std。
- 肩肘动作缩放0.02、两阶段手臂目标、腕部锁定和gimbal仍限制完全自由双臂动作，未改动作层。
- 原三点稳定奖励仍要求箱子较静、脚较稳等；原步态/姿态项仍保留。新增retain专门覆盖这些稳定奖励之前的过渡，未改原稳定成功定义。

因此本次只放宽指定奖励层的后仰阻力，**不声称整个机器人已没有其他动作限制**。

## 分阶段开关

```bash
# 阶段1：放宽后仰门控；intake仍为0，不加入retain
bash scripts/train_stage1_buffer.sh 1 [原训练参数]
# 阶段2：阶段1 + compliant_intake权重0.75
bash scripts/train_stage1_buffer.sh 2 [原训练参数]
# 阶段3：阶段2 + retain_reward权重0.5
bash scripts/train_stage1_buffer.sh 3 [原训练参数]
```

入口设置BOX_BUFFER_STAGE=1/2/3、BOX_ESCAPE_PENALTY=0、BOX_IMPACT_BUFFER=0。当前teacher入口不再注册escape_penalty或impact_buffer，即使旧环境变量残留为1也只提示退役；不是把-100改为0后仍放在RewardManager里计算。

BOX_BUFFER_STAGE=0或unset则不启用本实验；原始其他奖励仍存在，但被退役的两个负奖励不会恢复。旧历史分支可用于复现之前的实验，本次不回写其结果。

可调初始权重：BOX_BUFFER_INTAKE_WEIGHT=0.75、BOX_BUFFER_RETAIN_WEIGHT=0.5（要求非负有限）。权重之外的窗口阈值集中在BufferConfig，第一版不一起调动作/AMP/跟踪参数。未额外添加按角度奖励后仰的项，也没有新目标后仰角。

## 1. 接近与缓冲阶段的后仰自由

复用原torso_asym前倾/侧倾公式与权重，暂时移除其后仰部分。有效接近窗口或接触后控制窗口内，upright gate按 `safe * [relax + (1-relax)*old_gate]` 混合：relax=1时正常后仰不再因原gbx门控丢失接箱奖励；GATE_ALL与bed_gated_hold_sustained内的upright子门控均接入。

upright_hold在该过渡期暂时不评价站姿，直立/后仰一视同仁；稳定持有后再渐进恢复原奖励。箱子在身体附近、相对速度<0.5m/s持续0.5s后，进入1s姿态恢复渐变。稳定判定一旦达到就锁存，不能靠抖动箱子重置恢复时钟。原hold_still函数和权重不改，只变其已有外层upright gate在过渡期的因子。

安全门要求骨盆离地>0.45m、躯干离地>0.55m、躯干up轴z>0.25，允许明显后仰/屈膝但排除近水平倒地。越界会停止正向门控放宽，不新增终止条件。

## 2. compliant_intake：原公式 + 严格时窗

复用原 `0.6*yield_v + 0.4*squat_v`，不是新写固定后仰姿态。yield_v沿箱子世界水平飞行方向的骨盆速度投影、squat_v为向下骨盆速度，各以0.5m/s归一化并上限1。

新增门控要求真实发射、非HOLD_INIT、箱子仍在空中、世界速度>1m/s、相对接近速度>0.5m/s、距离<1.2m、预计到达时间<0.6s、预测最近经过距离<0.75m。原函数更窄的0.9m近距门仍保留。首次有效接触后关闭此接近项，转入retain；每投掷最多累计0.6s接近奖励窗口，远离再回来不会重置预算，静止箱子附近持续蹲起也不会付费。

默认权重0.75，raw最大1；RewardManager dt=0.02时每环境每步最大+0.015。例如原公式raw0.4对应加权rate0.3，实际每步+0.006。

## 3. retain_reward：接触后的过渡控制

复用box->机器人每刚体过滤接触力子步历史，额外独立box->ground过滤；箱子最低点高度由当前四元数和半尺寸计算。地面接触与机器人接触分开，不把箱子落地误当接住。

只有有效、安全、离地的机器人-箱子接触才启动。随后须在躯干0.9m可控半径内、最近有效接触距今<=0.3s、未向外快速脱离（径向速度<2m/s）、箱子底面离地>8cm、机器人未倒地。**不要求立即静止**，允许3m/s的瞬时相对运动、短暂脱手、后退和迈步；只在明显向外失控时停止付费。

奖励按靠近程度与首次接触后0.1s渐入计算，不按手臂姿势计分。每投掷最多2s过渡窗口；稳定持有后随站姿恢复逐渐退场，交给原hold_still等。落地或距离超过1.2m会锁存该投掷的失控，直到真正重投，不允许夹着落地箱子反复刷新窗口。

默认权重0.5、raw范围[0,1]，dt=0.02时每环境每步上限+0.01，低于原hold_still权重2的最大+0.04，不是大系数替换主要目标。

## 生命周期与日志

所有初始化、悬停、赋速、发射、realign、重投、HOLD_INIT等原事件及顺序全部保留。复用原记账入口路由到buffer_throw_reset，清理本投掷接触/预算/计时/失控；不添加或跳过事件。新增状态不进入actor/critic观测或checkpoint网络。

默认独立JSONL：outputs/buffer_control/buffer_<PID>.jsonl，可用BOX_BUFFER_LOG_PATH覆盖、BOX_BUFFER_LOG_EVERY（默认50）调整快照频率；同时stdout与Buffer/*命名空间日志。

- raw_contact_rate为真实机器人接触率；contact_rate为过滤落地/倒地等无效接触后的有效接触率，二者分开，避免与历史42%口径误比。
- catch_success_rate使用原strict_all连续0.6s，每投掷只记一次，不是retain候选率。
- mean_control_seconds为累计有效控制时间/投掷数；throw_end另外记录每投总控制时间和最长连续控制时间。
- intake_raw_mean/retain_raw_mean为未加权候选均值；intake_reward_mean/retain_reward_mean为本阶段实际加权rate均值（dt前），阶段未开启的项为0。
- back_lean_degrees、knee_flexion、root_height、root_ang_speed、safe_fraction用于比较后仰、屈膝和站立稳定性。

## 测试与待执行回归

73项联合CPU测试通过（新项21+此前52）；Python编译、shell语法与Git空白检查通过。测试包括真实原奖励公式的方向投影、正常后仰不罚/前倾侧倾仍罚、稳定后恢复、不能靠抖动逃避恢复、接近预算、接触新鲜度、短暂脱手、高速但可控状态、地面与倒地、最大2s时窗、部分reset/HOLD_INIT、分阶段权重及旧负项不注册。Isaac接口在本地测试中使用替身，不是物理仿真。

人工轨迹短测：`python scripts/check_buffer_control.py --out outputs/buffer_cpu_short_20261110`，生成三阶段JSONL和summary.json。闲蹲、落地、倒地、失控均无新增正奖励，移动控制和短暂脱手可获有限过渡奖励。这些统计是夹具结果，不能证明策略不会reward hacking。

真实仿真回归入口（各阶段128环境×12轮，同seed和同一起点，共3组）：

```bash
bash scripts/regress_stage1_buffer.sh /实际路径/model_599.pt /新的输出目录
```

本次已依次执行并检查：RewardManager没有escape/impact项，阶段1的intake=0且无retain，阶段2的intake确实非零，阶段3的retain确实非零，日志有限，箱子落地/倒地后不继续付费，全部原事件照常执行，输入维度仍183/305。短回归中尚未见到的触发场景不能标成验证通过。

## 修改文件

buffer_control_core.py：纯状态机；buffer_control.py：运动学、原reward复用、阶段门控与日志；buffer_control_config.py：三阶段配置；escape_penalty_sensor.py：接触过滤复用/地面独立列；teacher_env_cfg.py：在AMP_SLIM之后应用及退役负项；throw.py：两个已有upright门及生命周期开关；impact_buffer.py：复用记账路由。两个train/regress脚本、CPU短测脚本与测试/本文档提供复查入口。


## 2026-11-10 GPU回归实测

服务器RTX4090；同一昨晚model_599.pt warm start、seed42；每组128环境×12更新=36,864步，全部完成，模型/日志无NaN或Inf。主回归代码e1a8ffc，审计日志9763693，实际地面碰撞叶节点过滤修复db094aa。

| 指标 | 阶段1 | 阶段2 | 阶段3 |
|---|---:|---:|---:|
| intake权重 | 0 | 0.75 | 0.75 |
| retain权重 | 不注册 | 不注册 | 0.5 |
| 加权intake步均值峰值（dt前） | 0 | 0.01778 | 0.01770 |
| 加权retain步均值峰值（dt前） | 0 | 0 | 0.02437 |
| 有效接触/投掷 | 23/119 | 23/124 | 22/122 |
| 平均累计控制时间/投掷（秒） | 0.0504 | 0.0482 | 0.0502 |
| 末帧平均后仰（度） | 2.01 | 2.19 | 2.22 |
| 末帧平均膝关节屈曲（rad） | 0.650 | 0.627 | 0.649 |
| 末帧safe比例 | 99.2% | 97.7% | 99.2% |
| 严格接箱事件 | 0 | 0 | 0 |

以上只是12更新的触发/数值回归，不能据此判断阶段优劣或成功率提升。平均姿态是全环境均值，不是有效接球时的条件均值。

实际RewardManager中三个阶段都没有escape_penalty或impact_buffer行；阶段2的compliant_intake实际权重0.75且非零，阶段3额外retain_reward实际权重0.5且非零。YAML配置与昨晚基线比较：三个阶段的events内容及键顺序、actions、terminations、observations均完全一致。输入维度保持183/305。

额外审计发现最初过滤/World/ground没有命中静态碰撞。箱底几何高度保底原已阻止落地奖励；现已修复为枚举UsdPhysics.CollisionAPI叶节点，真实路径为/World/ground/terrain/GroundPlane/CollisionPlane。修复后补跑阶段3共288快照，地面接触最多45环境同时命中，grounded最多53；invalid_positive_intake和invalid_positive_retain全部为0。出现过最大后仰56.71°，这不意味着该时刻一定有奖励，仍受有效时窗/离地/安全条件检查。

保留补充审计记录：第一份审计因同步未成功仍是旧日志版本，不计为审计通过；v2审计定位到地面过滤问题，ground_audit为最终修复验证。没有删除这些记录或将旧结果冒充修复后验证。

服务器及本地输出：outputs/buffer_stages_short_20261110/（stage1/2/3日志与verified_summary.json），outputs/buffer_stage3_audit_v2_20261110/，outputs/buffer_stage3_ground_audit_20261110/。完整原始日志已下载到本地副本。

## 后续正式训练状态

用户问“怎么还没开始训练”后，按其建议的对照顺序先启动阶段1：BOX_BUFFER_STAGE=1，只放宽后仰相关奖励；从昨晚model_599.pt恢复PPO权重与优化器，4096环境、seed42，追加200更新。不使用小批量短回归的权重作为起点。输出outputs/buffer_stage1_resume200_20261110/；训练后脚本自动CLEAN2两seed各320次投掷及无坐标轴视频。阶段2/3正式训练留待阶段1结果后比较，不把三个改动直接混成一组。
