# V35：Physical WM编译数值定位与保留运算边界的修复候选

状态：本地实现/CPU验证，GPU待测；没有宣称数值问题已解决、没有新性能结果、未改正式训练或checkpoint。继承V34同一capsule、FP32/TF32关闭和默认dtype容差。

## 已知问题与假设边界

V34最小链表明：WM单独编译和Actor＋WM融合均在相同参考输入第6步的箱子角速度Y（84维状态索引82）首先超差。动作/控制历史通过，因此先查Physical WM内部。完整H32及实际Actor Adam也失败。这个最小控制不含真实任务事件/FK/Reward，不能把错误归因于原生事件。

目前还不能断言是FMA、归一化除法、激活近似、矩阵乘法或积分。V35不改公式、编码、输入归一化统计、Adam epsilon或奖励，不用放宽容差消除失败。原`stable_canonical`编码试验仍独立，本轮没有静默打开或迁移旧模型。

## 同输入定位

`physical_numerics_v35.py`从原capsule生成eager参考H32，在第1/6/16/32步取完全相同的输入，分别比较：

1. features：canonical状态、Target、精简Context列和事件特征。
2. normalization：原Standardizer，包括原mean/std/clamp。
3. mlp：原网络全部线性层和激活。
4. integrate：原82维残差到84维状态的积分，保留root/box COM、坐标系和四元数处理。
5. encoder：features＋normalization，识别跨边界融合才发生的差异。

各阶段单独比较FP32输出和固定非均匀cotangent的输入VJP。没有用候选状态推进参考序列，不混入前一步误差。梯度针对同一组独立叶输入隔离检查；这是数值探针，不是Actor训练detach。`--mlp-layers`可追加各Linear/SiLU层的同输入/VJP检查。局部通过不证明整个融合图通过，因为返回中间量/分段会改变编译边界。

进度写`physical_components.partial.json`，完成写`physical_components.json`。manifest保存capsule SHA、Torch版本、源码SHA及FMA请求环境。模型/normalizer不更新；RNG与TF32在CLI结束时恢复。无Isaac初始化、无采集、无训练checkpoint写入。

## 修复候选：只改变编译边界

`physical_compile_v35.py`直接调用原features/Standardizer/net/integrate，支持：

| driver | 编译区域 | 用途 |
|---|---|---|
| wm_features | features | 隔离canonical/Context装配 |
| wm_normalization | Standardizer | 隔离除法和clamp |
| wm_encoder | features＋normalization | 检查跨编码边界融合 |
| wm_mlp | 原MLP | 首选候选：物理编码、归一化和积分保留原执行方式 |
| wm_layerwise | 各Linear独立，激活保留原执行方式 | 对照激活/epilogue融合造成的误差 |
| wm_integrate | integrate | 隔离积分和坐标变换 |
| wm_encoder_mlp | encoder＋MLP | 检查编码和网络间融合 |
| wm_mlp_integrate | MLP＋integrate | 检查网络和积分间融合 |

这些只是候选，未GPU验收，不称为已修复。若MLP-only能严格通过并保留足够收益，再接入完整Actor测试；否则根据阶段误差进一步收窄，不推广无收益的layerwise执行。

全部模型权重和网络规模从capsule恢复；不替换激活或矩阵乘法，不增加FP64，不改变FMA以外的精度设置。候选不注册重复模型子模块，不改变state_dict布局。源码不修改models.py/physics.py/trainer.py，也没有新训练配置开关。

## H32、梯度和实际Adam

原`diagnostics_v31.minimal.check`增加这些显式driver，旧默认三组保持不变。对每个候选：相同权重、标准化统计、输入、目标、RNG、FP32、H32、loss、梯度裁剪及Adam初始状态；仍检查轨迹、同输入逐步误差、非零physical BPTT及真实参数更新。

新增`parameter_update_drift`记录最大更新差坐标、两边更新后的权重、该坐标实际应用梯度，便于检验近零梯度的Adam敏感性，不能用全局norm接近替代更新一致性。第一组必须eager，避免拿候选自身当参考。

本工具是独立计算控制：没有原任务的真实事件/FK/Reward/AMP。即使该控制通过，仍须完整Actor真实事件H32、梯度、Adam和实际wall-clock回归；WM H8训练的梯度/Adam也须独立验证。不能用最小链0.077秒替代完整更新成绩。

## GPU测试

复用原capsule，先做默认四边界试验：

```bash
ISAACLAB_PY=/opt/conda/envs/isaaclab/bin/python \
  bash scripts/study_physical_wm_v35.sh \
  /tmp/reflex_v31_gpu_20261108/full/minimal_capsule.pt \
  /tmp/新的V35默认数值定位目录
```

默认阶段为1/6/16/32，driver为eager/segmented/wm_mlp/wm_layerwise，H32、预热2次、稳态10次；超时600秒，不覆盖目录。脚本显示每组numeric gate，完成不等于通过，没有自动启用训练。

可先只测组件/细分MLP，不重复H32：

```bash
PYTHONPATH=source/reflex /opt/conda/envs/isaaclab/bin/python \
  -m reflex.world_model.physical_numerics_v35 \
  --capsule /tmp/reflex_v31_gpu_20261108/full/minimal_capsule.pt \
  --device cuda:0 --output /tmp/新的V35组件目录 \
  --components-only --mlp-layers --steps 6
```

仅在默认组件结果支持该方向时，才选择`V35_NO_FMA=1`做独立请求禁用Triton点运算FMA的对照；脚本在Python导入前设置`TRITON_DEFAULT_FP_FUSION=0`及全新Inductor/Triton缓存，不污染默认试验。该环境变量是否在当前Triton生成代码生效仍需核对；不能把设置环境变量等同于已验证修复，也不会声称它禁用了cuBLAS内部的FMA。没有把CPP编译选项当作GPU开关。

## CPU证据与回滚

最终16项CPU专项通过（25.33秒），40项相关回归通过（95.04秒，包含重叠检查）。MLP-only与layerwise另覆盖真实Replay H8条件化监督的WM梯度及实际WM Adam，保留原mask和梯度裁剪。CPU使用aot_eager，验证图捕获、数学链、VJP与实际更新，不能证明CUDA/Inductor的舍入行为。测试和结果见`benchmarks_v35_cpu/README.md`。未更改旧事件源码哈希，无新增checkpoint迁移。回滚时继续V34/V32入口，不选V35显式driver即可。
