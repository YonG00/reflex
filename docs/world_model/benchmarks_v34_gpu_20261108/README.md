# V34 GPU原始JSON

条件、解释及失败项见 [GPU报告](../V34_GPU_RESULTS_20261108_ZH.md)。

- native_event_probe.json：64真实环境，24步原生事件一致性、正例扰动、H32发射及位置梯度，通过。
- actor_execution_v34.json：4096真实环境、2048/H32，20轮交替真实Actor/Critic更新；数值/无新增编译通过，收益小，未放入正式配置。
- fusion_boundary_probe.json：初版探针的原始结果。single_step同输入字段因requires_grad元数据改变不适用于正常H32解释，不使用其中第2步巨大误差。
- fusion_boundary_probe_r2.json：c0123d5修正元数据后重跑；程序完成，但分段/单步融合的状态和实际Adam未通过。该报告是最终融合定位依据。

原生/完整配对服务器输出`/tmp/reflex_v34_gpu_20261108_r1`，driver退出0；最小链R2为`/tmp/reflex_v34_fusion_20261108_r2`，退出0。两次退出码0不等于融合数值通过。未训练新checkpoint。
