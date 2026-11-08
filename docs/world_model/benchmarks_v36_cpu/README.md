# V36本地验证

macOS/Python3.12/Torch2.15，CUDA不可用，aot_eager检查捕获及数学链。各组有重叠，不相加作独立测试数。

- related_45.log：45通过/115.44秒，V32及既有数值/事件/恢复相关回归。
- final_38.log：38通过/167.83秒，完整Actor融合配对、严格数值失败仍保留风险与性能、WM/Actor分阶段、异常保存/恢复、H8/H32、V31适配等。
- preparation_9.log：最终9通过/61.75秒，补外部终止清理和实际超时，父退出但子忽略信号的进程组清理，阶段门控、checkpoint原子保存、完整融合及恢复。
- risk_5.log：5通过，物理单位/四元数正负号、小角度、实际Adam delta而非post-weight、极小梯度和软存活加权误差（不隐藏未加权误差）。

新模块Ruff、脚本bash -n和git diff --check通过。train.py保留既有无关lint告警，本次不重排Isaac导入顺序。CPU不能证明GPU速度/Inductor舍入或完整任务效果。
