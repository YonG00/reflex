# V33：H32整体执行诊断与事件重写可行性

## 研究范围

基于V32结果 `011f0d7`。本阶段不继续把局部clone优化作为主线，不改奖励、随机规则、事件顺序、梯度或网络结构。之前本地准备的“证明无效果后省略初始化回调”候选已撤出，当前代码没有省略事件或固定计时器优化开关。

当前交付是完整链路测量入口，用于决定是否值得实现固定Batch事件程序。没有实现或启用GPU事件重写，没有新的提速结果，也未启动GPU或正式训练。保留V32约1.64秒作为已验证基线。

## 源码事实与结构性机会

`teacher_terms.v38_walking_hover_and_launch` 已对整批计算发射速度，通过 `where` 选择悬停、发射和当前状态，并调用原生写入。仅把发射公式改成batch并不是新优化。

值得研究的是整个执行边界：

| 环节 | 现状 | 固定形状研究机会与限制 |
|---|---|---|
| scheduler时钟 | 每项筛选、获取动态长度、随机重置，再执行回调 | 非随机mask/count可集中计算；随机重置仍须保持与回调随机数交错的原顺序 |
| 随机游走 | `bool(redo.any())`、`int(redo.sum())`、`rand(k)`、布尔索引 | 稠密数学可融合；直接换为`rand(B)`再mask会改变生成器序列 |
| 发射连续计算 | 已是batch，依赖实时预测机器人位置、p0及投掷参数 | 适合研究连续张量段＋确定性写入融合；不能代入Replay未来真值 |
| 重投 | 根据预测状态判坏，动态ids调用reset采样 | 需要保留全套reset与随机调用，不可只处理首发或把重投删掉 |
| view / WritePlan | 构建影子asset、控制状态、写入记录，再合并 | 可能可改为固定Tensor状态与mask，但要保留回调间“读到前次写入”的顺序 |
| H32图组织 | 已有单步编译、AMP/Critic批处理；仍逐步调度 | 先量化编译单元/Autograd调用成本，再选分段边界；不预设H32巨图最快 |

原生源文件：`envs/teacher_terms.py` 的hover/launch、cradle perturb；`envs/throw.py` 的hold-init、first realign和reset；时钟与回写在 `events_v24/runtime.py` / `execution_v30.py` / `writes.py`。本轮没有改这些事件规则文件。

## 不能由计数直接推出的结论

V32约81,310个kernel、1608次stream同步。后者在Profiler下累计CPU时间约13.7ms，不能凭次数就称它占了1.6秒的大头。Profiler使CPU范围膨胀，不能把kernel work、范围时长和无Profiler wall相加减后当作因果收益。

需要回答：同步发生在哪个Python调用及其反向来源？等待本身有多长？是不是大量CPU dispatch、视图装配或Autograd节点使GPU等CPU？GC尾延迟是否来自基准中的快照恢复，而不是正常连续更新？

## 新增测量

### A. 连续更新的粗粒度时间线

在完整Profiler之前运行，避免Profiler事件树延迟GC污染计时。两组从相同参数/Adam/RNG出发，分别连续更新20次；**组内不恢复快照、不逐次复制参数到CPU**。最后比较实际Actor/Critic参数和梯度，再恢复原状态。

一组不加记录，一组记录：

- Actor forward（含H32、批量AMP/Critic估值及return）。
- Actor BPTT。
- 裁剪与Adam。
- 原有指标中的完整Actor、Critic及完整调用时间。
- GC generation、停顿时长及所在阶段。

CUDA Events只在上述粗阶段记录，没有逐模块synchronize；使用原更新结尾的同步后读取事件。CUDA时间跨度包含设备空闲，与host跨度重叠，不能相加，也不等于kernel忙碌时间。

每组计时前仅在区间外清理预热/快照垃圾；组内GC保持启用。它是冻结Replay/WM/AMP的连续Actor诊断，不是完整持续联合训练吞吐。两组顺序不同的温度/GC因素仍需解释，不能将差值全部视为记录开销。

### B. 完整链路的调用点Profiler

保留V31整链检查和真实sampler，20次无Profiler完整更新另测稳态，再捕获一次预热后的完整trace。额外范围区分：

- 每个原生回调与invoke。
- view准备、WritePlan record/select/merge、asset refresh。
- scheduler中的Tensor `nonzero`、标量读取、布尔索引的文件/行号。
- 按forward sequence/thread匹配的反向来源；无法唯一匹配的单列。

源码点hook只在eager scheduler执行期间临时安装，退出即恢复；不包裹 `_view`、不破坏V32的函数身份判断，不在编译的plant/FK/heads或BPTT中替换Tensor方法。脚本化/C++内部操作可能没有Python行号，这些保留回调级或unmatched，不能猜测来源。捕获期间重编译则报告不可用。

修改 `joint.py` 仅添加默认关闭的粗阶段记录边界；`fast_view.py` 仅增加可关闭的诊断范围，原函数身份和数学保留。Runtime、WritePlan和原生事件函数未改；默认训练路径没有新Tensor运算或随机数。

## 初步可行性判断

值得研究固定Tensor事件状态和mask程序，但不承诺它一定是最大瓶颈。第一步优先选择箱子完整等待→发射路径，保留其随机游走、重投和与其他事件的写入先后关系。

实际可选边界：

1. 保留原随机采样调用及长度/顺序，将稠密连续计算与mask写入组织为更大的纯张量段。这保真风险较低，但仍可能保留少量CPU边界。
2. 若数据证明这些边界仍限制性能，再研究固定容量索引/批量count回传，或设备侧随机状态。必须正确处理容量溢出及完整分支，不可截断事件。
3. 设备侧RNG或H32 Graph不是简单打开开关：需要对齐PyTorch随机算法、offset推进、各分支采样形状，并保留launch对机器人位置梯度，工程量要由潜在收益支持。

是否做上述重写、选哪条边界，等新GPU数据量化后决定。没有预先认定事件一定是主因，也不会通过省略回调、缩短Horizon或改变精度达标。

最终“WM＋Actor整轮1秒”还须在统一配置的同一进程测真实采集、WM10次、Actor/Critic与实际调度。历史WM约0.25秒来自独立Graph基准，本工具专注Actor诊断，不能将两个独立最小值相加当作端到端成绩。

## 使用和验收

```bash
export ISAACLAB_PY=/opt/conda/envs/isaaclab/bin/python
export REFLEX_DATA_DIR=/root/gpufree-data/reflex/data
bash scripts/study_actor_v33.sh \
  /tmp/reflex_v29_scratch_joint30_20261107/checkpoint_final.pt \
  /tmp/新的V33诊断目录
```

固定V32 verified、4096环境、2048/H32、FP32。输出 `run/phases.json`、`run/full/actor_full_v31.json`（新增`execution_sites`）、完整trace及排除GPU用户标记的trace summary。

脚本拒绝覆盖输出，设置超时；检查完成、数值、无重编译/归属可用状态，不只看退出码。沿用默认dtype容差；检查原始轨迹、Actor梯度、实际参数更新和恢复。完整CPU回归215 passed、1条原有弃用警告（715.36秒）；随后连续测量器最终6项复测通过（68.58秒）。GPU诊断已完成，门控通过；没有新增执行加速。原始结果见 [GPU记录](benchmarks_v33_gpu_20261108/README.md)，解释及下一步见 [V34](V34_EXECUTION_ORGANIZATION_ZH.md)。CPU日志见 [CPU记录](benchmarks_v33_cpu/README.md)。

回滚：继续V32入口，不选择 `execution_study_v33`；不会替换正式训练的事件执行器。
