# WM稳定编码与H8 Graph正式接线

## 状态

模型/训练入口/配置/checkpoint合同/配对联合验证入口已接线。CPU验证通过；当前服务器是无卡模式（torch.cuda.is_available=False，device_count=0），新组合尚未做CUDA/Inductor验收，不能宣称已取得旧基准速度或已可替换正在使用的模型。原训练配置不静默改变，历史checkpoint可继续使用。

## 接线

- `TrainConfig.wm_encoding` 默认legacy。`stable_heading_v1`在CompactContextWorldModel中持久选择稳定canonical，第3列de-yawed R[1,0]精确置零；其他维度、物理积分、全量解析context与原生事件不改。仅支持已审计的compact stage1物理context。
- 稳定编码使用同一纯函数，原benchmark临时替换也复用该函数，避免两套数学规则。
- `compile_wm_rollout=true`编译整个H8条件化监督递推与物理损失，真实外部事件指令按已接受的recorded_commands协议推进，不跳过发射/悬停等数据。
- 新`wm_training_compile_mode=reduce-overhead`仅控制WM训练编译。`wm_compile_mode=default`保持Actor动力学预测原编译边界，避免启用WM Graph时顺带启用Actor Graph。
- 每次WM梯度更新前统一设置Graph迭代边界；H8内前后依赖、反向与Adam顺序不改，优化器在Graph外。Standardizer.fit及load_state_dict维持mean/std/参数存储地址，CPU测试检查地址与值。

新候选配置：`configs/world_model/joint_h16_batch4096_wm_graph.json`，H16、WM H8、双batch4096、4096真实环境、Replay512、WM3×2048、原奖励/AMP/软存活，WM-only500轮。它是下一轮待GPU验收配置，未自动用旧模型启训。

## checkpoint规则

旧配置缺少wm_encoding时按legacy加载；不增加legacy learning_contract字段，因此旧checkpoint合同保持兼容。稳定编码checkpoint明确保存wm_encoding及exact_zero_column，WM、normalizer、Adam保持整套恢复。相同编码可切换训练编译开关，不丢优化器。

legacy与stable编码不能直接互载，加载在修改权重前给出明确错误。没有伪造旧存档字段、静默改normalizer或清空Adam。原误差源是浮点噪声，不能承诺从旧编码向稳定编码无损转换；本次选择新WM/normalizer/Adam从零开始，旧模型保留旧编码。minimal capsule导出/恢复也保留编码，评测脚本从checkpoint重建配置。

## GPU验收入口

```bash
source /root/gpufree-data/reflex_v36_env.sh
bash scripts/check_wm_graph_training.sh /root/gpufree-data/全新WMGraph验证目录
```

此入口从零初始化稳定编码模型，采集真实Replay后，在相同模型、优化器、AMP、RNG和Replay下对照：原WM训练执行、H8 Graph训练执行。两组都接完整Actor/Critic更新；不跳过事件、奖励、AMP、终止或梯度。研究入口显式临时绕过500轮Actor门槛，结束恢复，仅用于短诊断，不更改正常训练预热流程，不保存试验模型。

每组预热2次，5轮交替配对；输出WM10+AMP、Actor+Critic与完整更新耗时、峰值显存、实际WM/Actor/Critic参数和梯度、WM更新后H16轨迹/事件/Reward/AMP/终止与原始Actor梯度严格对照。另采Profiler确认实际cudaGraphLaunch、捕获期间无新增编译；不以compile开关代替Graph launch证据。冷阶段原子进度/超时清理保留，不启用曾触发Python崩溃的定时faulthandler线程。

900秒硬上限，不覆盖旧结果，不进行半小时训练。成功条件包含completed、numeric_passed、performance_usable、Graph launch>0、profile无新增编译及模型/Actor恢复；实际吞吐改善和满Replay显存仍需检查。失败不放宽容差、不自动提升候选。

## 本地检查

原legacy/compact/宏事件/Actor/容量覆盖等相关回归91项通过。新增测试覆盖：稳定编码存档与normalizer/Adam往返、legacy双向拒绝与拒绝前不改权重、连续WM→Actor更新、Graph仅作用训练而不改变Actor模式、验证器退出时模型/AMP/RNG/Actor门槛恢复、稳定minimal capsule编码。精确零恒等式FP64值和切空间梯度测试保留。日志见benchmarks_wm_graph_integration_cpu，CPU不证明CUDA数值/速度。

## 分支整理原则

各开发/历史分支独有提交先核对为当前历史祖先，合入main并推送后才删除origin与本地旧分支。其他worktree只解除分支绑定，目录和未跟踪文件保留，不删除checkpoint、数据、视频或源码目录。upstream为原作者远程引用，不删除其服务器分支。
