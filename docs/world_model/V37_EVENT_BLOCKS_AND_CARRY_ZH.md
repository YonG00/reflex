# V37：事件分块写入与小范围状态携带融合

## 范围与状态

本地实现与GPU四组对照完成：严格数值及单次实际更新一致性通过，但无可靠加速，两个开关继续关闭。详见V37_GPU_RESULTS_20261108_ZH.md。两个候选默认关闭，不修改正式训练配置、奖励、Actor/Critic/WM网络、H32/H8、软存活或随机数调用。V36完整Profiler结果作为研究依据，不根据代码变化宣称收益。

## 两个独立候选

### BlockWritePlan

`events_v24/blocks_v37.py` 将原84维状态分成7/29/6/29/7/6六块，对应机器人pose、q、根速度、dq、箱子pose、箱子速度。完整batch且列为已知完整块时，仅对该块执行functional where及mask更新；需要整块targets/mask时才拼接并缓存。目标是减少每次根部写入在整块84维targets上的in-place索引写入及其反向节点。

它与V34 ContiguousWritePlan不同：V34是整块targets上的连续列写入，本候选改变的是写入图的状态组织。部分、乱序、重复行或不规则关节列继续使用原WritePlan录入，select/merge保留独立存储和原始index_copy语义。写入值即便等于旧值仍保留mask，常量写入仍正确替换原物理状态的梯度。保留原四元数、世界/局部坐标转换及PlannedAsset规则，不碰事件函数、计时器、随机抽样或回调顺序。

构造和输出物化可能增加cat开销，收益待GPU测量；不能预先认定更快。旧runtime.py/writes.py及其source-pinned迁移规则未改。

### fused_actor_carry

`actor_carry_v37.py` 在原packed姿态携带基础上，将物理状态、上一动作、位置及四元数的where携带组合为一个独立torch.compile单元，仍在每步末尾执行。不融合事件回调、不编译整个H32、不执行新的浮点姿态计算，也不detach；phase、事件历史、合法性和终止更新仍保持原顺序。

两个配置键 `block_event_writes`、`fused_actor_carry` 默认false；实验候选可显式打开。检查旧checkpoint缺少这两个键时恢复Actor/Adam，执行开关不触发模型权重迁移；恢复不会静默丢弃优化器。carry要求state WM与packed_pose_carry，block与contiguous计划工厂不能同时配置。

## GPU验证入口（当前未执行）

```bash
source /root/gpufree-data/reflex_v36_env.sh
V37_REPEATS=20 V37_TIMEOUT=600 bash scripts/study_event_blocks_v37.sh \
  checkpoint_final.pt /root/gpufree-data/新的V37输出目录
```

保持V32验证配置、4096真实环境、Actor2048/H32、FP32/TF32关闭；真实采样40步后固定Replay、同权重/Adam/RNG，比较reference、block_writes、fused_carry、combined四组。每组预热2次，交替配对实际Actor/Critic更新；每次更新后恢复快照，正式训练不启动。新结果目录，不覆盖旧实验。

输出严格轨迹/Reward/AMP/Termination/事件/梯度/实际参数更新、风险指标、事件转换覆盖、样本SHA、median/p90及显存，记录新模块源码hash。稳态新增编译单独检查。正常结束、失败或超时均恢复或清理对应诊断进程；不使用定时faulthandler线程。实际GPU数值不通过或收益不明确时，不进入正式训练。

## CPU验证口径

新增24项测试覆盖完整/部分/乱序/重复/空行写入、重复列fallback、intervention mask、保留中间状态后的梯度、select/merge、常量等值写入、carry的值/VJP、完整H32四候选实际Actor/Critic更新和RNG/Adam恢复、等待发射与重投/机器人推力事件fixture、实际reset_box_throw原生函数AST（采样器stub）及旧checkpoint恢复、异常恢复。

原V32/V34回归58项通过（119.46秒）。新增CPU测试使用aot_eager验证编译计算图及梯度，不等于CUDA/Inductor验证，不等于全部真实接触情形覆盖。真实PhysX接触/重投、实际CUDA误差、冷编译成本和稳态提速仍需开卡。

不调整容差，不以减少Horizon/batch、删奖励或跳过事件冒充加速。GPU取得完整结果之前保留基线，不合入正式训练路径。

## GPU验收更新

四组各20轮完成，退出0；median为1.585129/1.631642/1.583892/1.595805s，全部数值差0、无稳态新增编译。没有值得启用的提速，未进入正式训练。上文GPU验证入口与风险描述保留作实验复查。
