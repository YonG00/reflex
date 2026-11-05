# 联合训练 V2.1 本地修复与开卡验证清单

审查基线：`1baab50`。规格来源：用户提供的《Reflex × SuperMimic 联合训练修复方案 V2.1》。本轮只修改本地代码、运行CPU检查，未连接服务器或启动GPU训练。此前半小时训练并未启动。

新配置：`configs/world_model/joint_repaired_v21.json`。旧`joint_reproduction.json`保留，不能把新配置当作旧试训的无缝续跑。

## 已实现与CPU检查

| 项目 | 原语义 → 新语义 | 文件 | CPU验证 | GPU待验证 |
|---|---|---|---|---|
| 网络 | Actor256/256、Critic256/128 → 两者512/256/128 ELU | trainer.py、新配置 | 层宽/输入维度/激活测试 | 实测参数与导入路径 |
| 初始化 | Actor末层零权重 → PyTorch Linear默认初始化 | trainer.py | 与本地安装RSL MLP同种子逐参数一致 | 服务器RSL版本及起点行为 |
| 特权Critic | 与Actor共用179 → 独立292维、独立训练集归一化 | observation.py、trainer.py、joint.py | 独立旋转矩阵核对身体位置/姿态、梯度与292拼接 | 与真实环境critic观测逐项parity，必须通过 |
| 终止 | Actor使用解析0/1 → 独立64/64 SiLU头输出物理终止概率 | models.py、task.py、trainer.py | BCE可下降、Brier、概率输入梯度、冻结检查 | 真实标签正例覆盖、概率校准、分布外行为 |
| 梯度诊断 | 分块路径只记总范数 → 独立只读诊断 | diagnostics.py、train.py | 不更新参数/Adam、不消耗训练RNG、零奖励识别、梯度分解 | 各真实接箱项有效梯度与开销 |
| 效率 | 每步重复FK → 当前/预测FK供Actor、Critic、AMP、Task共享 | physics.py、observation.py、task.py、trainer.py | 原回归、前向与反向一致 | 真实耗时和显存，不承诺倍数 |
| 编译 | 无 → 可选纯Dynamics和FK fullgraph编译 | trainer.py、check_wm_compile.py | CPU aot_eager输出及参数/输入梯度一致 | CUDA/Inductor与服务器Torch版本再验证 |
| 性能剖析 | 仅总时间 → 组件区间与独立trace | joint.py、trainer.py | CPU profiler实际导出测试 | WM/FK/Reward/AMP/Obs/BPTT及同步占比 |
| 真实数据统计 | 步均回报/终止总数 → 完整episode、throw、原因、接触、拦截、抱持 | experience.py、isaac_adapter.py | 计数/持续帧/未知接触/恢复边界测试 | 传感器过滤、子步历史、投掷边界 |
| 存档 | v2 → 修复配置v3 | trainer.py、inference.py | 保存/恢复/推理与旧配置拒绝测试 | 实际train-save-load-play |

### 292维观测合同

沿用原Reflex critic顺序：command58、motion anchor位置3/姿态6、14个跟踪身体的anchor局部位置42/姿态84、base速度6、相对关节58、上次动作29、物体位置/速度6，总计292。Actor仍为Oracle179维，新增物体姿态/角速度/尺寸/质量只属于当前Actor接口，未擅自扩大Critic维度。身体顺序来自实际环境contract，必须恰好14个；真实parity失败会明确报错。

归一化只使用训练环境数据，与留出环境隔离；Critic不再借用Actor的179维normalizer。保留原有一次拟合策略，未声称等同PPO的所有归一化训练过程。

### 独立终止头

Dynamics仍只输出82维物理残差，只优化状态loss。终止头另有优化器，输入是当前末状态相对任务目标的几何特征：anchor位置差3、anchor投影重力差3、14身体参考位置差42，标准化后拼接signed/absolute特征，64/64 SiLU→logit。

真实标签在physics之后、auto-reset及interval事件之前捕获，使用TerminationManager的**物理terminated**，不把timeout或外部事件当作物理正例。几何特征使用与真实终止一致的参考缓存，包含目标依赖。独立BCE从真实replay特征学习，不把分类loss反传给Dynamics。每轮Dynamics更新10次、终止头另更新10次，各自计数；不能把这20次优化器step都计成Dynamics更新。

Actor中冻结Dynamics、终止头、Critic、AMP判别器参数，保留对输入的梯度；存活权重与λ-return使用同一预测概率。真实环境仍使用原硬终止。timeout、参考结束、重投/推扰等继续独立截断并bootstrap，不用预测概率覆盖这些边界。外部跳变未被动力学模拟，边界价值仍是预事件近似。

校准指标包括BCE、Brier、预测率/真实率及10个概率区间的ECE/计数。CPU可学分类测试不代表真实终止头已校准，不自动宣称可用。

### 梯度诊断

每50次Actor更新执行一次，首轮也执行；独立batch16、完整H32。各任务项、AMP都记录直接奖励梯度，以及包含survival路径的目标梯度范数和对总梯度投影。额外记录Critic bootstrap贡献和完整目标的survival-only梯度（奖励及Critic值均detach，仅保留hazard导数）。task+AMP+bootstrap的梯度重构误差可验证统计口径。

该诊断使用autograd.grad和独立RNG作用域，不step优化器，不改变正式采样序列。`gradient_audit_every`仍为0，避免误触旧分块禁用路径；新增入口是`diagnostic_every`。对硬抱持计数出现零直接梯度会如实记录，不为产生梯度而修改原奖励。

### 接触、拦截与覆盖

新增单箱体传感器，只过滤机器人刚体，独立于机器人—地面的net contact force。法向接触力阈值1N，覆盖4个物理子步；统计接触区段以20ms控制步为单位，不冒充接触点总数。未提供/过滤错误/历史缺失/非有限值不会被静默称为“零碰撞”。真实传感器数据仍须GPU核对。

采用one-to-many配置，依据[Isaac Lab 2.3 ContactSensor API](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/api/lab/isaaclab.sensors.html)：单个箱体对机器人多刚体，读取force_matrix_w_history；未改任务碰撞几何和成功奖励。

分别记录：

- 实际发射事件、完成episode回报/长度、物理终止与timeout原因。
- 只对观察到起点且已结束的throw计算投掷指标，开放throw暂不纳入分母；恢复训练只继承累计已完成统计，不把旧物理episode与新仿真拼接。
- 真实箱体—机器人接触率，与CLEAN2几何拦截率分开。
- CLEAN2拦截沿用`per_throw_eval.py`：距torso小于r+0.45m、世界z>0.30m、速度<1.0m/s，连续至少5控制帧。它不是严格成功。
- 原catch_success_bonus宽松判据与原严格persist成功分别记录，未用宽松项冒充成功。
- 有效8步监督窗口、Actor可用起点；实际Actor批次的几何分组、真实接触前/中/后、参考相位10桶。几何stage仍明确是诊断分组，不是接触证据。
- 1/4/8/16/32步窗口/末态物理误差、机器人/箱子拆分、输入截断比例，以及按真实接触阶段分组的误差。

未加入阶段均衡、困难初始化或RSI课程。必须先看真实覆盖统计，再作决定。

### 性能与batch

新配置Actor batch2048、microbatch2048，一次优化更新；相对旧配置4096是**显式采样预算折衷**，不声称完全等同参考默认，也不把减半batch的收益冒充编译提速。WM10×2048窗口、UTD0.4、H32和任务/AMP机制保持。

默认`compile_dynamics=false, compile_fk=false`，直到GPU编译等价检查和profiling确认后再启用。CPU验证后仍不能保证CUDA/Inductor速度或服务器Torch2.7兼容；检查脚本不更新任何模型。

## 存档兼容性

修复配置为新基线。旧v1/v2 Actor/Critic、优化器和179维Critic normalizer不能完整续训到新结构；加载会明确拒绝，不静默跳过权重。旧179维策略的独立推理加载器仍支持，旧配置可用于新运行。当前Adapter增加了诊断合同和Replay字段，历史完整存档没有自动迁移这些字段；如需原样恢复旧完整实验，应使用对应旧提交，不能把独立Actor推理兼容误解为完整Trainer恢复兼容。新v3含独立Critic normalizer、终止头/优化器、Dynamics/Actor/Critic/AMP/各优化器/replay/任务参考与RNG。旧纯物理WM潜在可复用，但本轮未自动导入，避免混合基线；新基线默认全部从零。

## 开卡后执行

本轮未执行以下命令，只有语法/CPU检查：

```bash
source scripts/local/env.sh
# 真实Actor179/Critic292、奖励/终止/接触接口预检，再短时联合回归和profile
bash scripts/preflight_joint_v21.sh outputs/world_model/NEW_v21_check

# 对短训新v3存档检查GPU编译输出和梯度；不改变存档
python scripts/check_wm_compile.py \
 --checkpoint outputs/world_model/NEW_v21_check/smoke/checkpoint_final.pt \
 --urdf source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf \
 --output outputs/world_model/NEW_v21_check/compile_equivalence.json

# 真实检查通过后，30分钟联合训练，预算到时完成当前update并保存
bash scripts/train_joint_v21_30min.sh \
 outputs/world_model/NEW_v21_check/smoke/checkpoint_final.pt \
 outputs/world_model/NEW_v21_30min
```

30分钟预算不含Isaac启动；在更新边界检查，允许末轮更新和存档造成少量超时，不用外部timeout杀进程。比较应使用同任务Oracle PPO，不拿原感知受限PPO混作公平基线；固定真实评测条件，报告实际episode/throw分母。

放行条件是接口/梯度/存档/有限数检查通过；没有恢复原先以人为WM RMSE门槛阻止联合训练的流程。是否有接箱改善必须由后续真实回放和严格评测决定。

## 本轮验证状态

- 完整CPU回归：38 passed（包含编译前后想象输出和Actor梯度、RSL初始化、soft survival/GAE、校准、只读诊断、v3存档、接触历史及CLEAN2持续帧计数）。
- 静态检查：ruff E9/F、Python编译、两个shell入口语法检查通过。
- Spec/Standards两项代码审查完成。已关闭：存活梯度遗漏bootstrap、critic-only可选存档自加载失败、CLEAN2拦截率缺失及起点分布不足。
- 尚未执行：真实Isaac/GPU parity、传感器过滤与子步数据、真实终止校准、CUDA编译/性能、30分钟训练、Oracle PPO对照。
- 本轮没有启动GPU，也没有声称接箱效果改善。开卡后先预检，再决定启用编译和运行半小时训练。
