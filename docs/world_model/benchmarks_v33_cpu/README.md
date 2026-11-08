# V33本地验证（2026-11-08）

macOS CPU，Python3.12.15 / Torch2.15.0 / pytest9.2.0。未运行GPU。

| 检查 | 结果 |
|---|---|
| 首次专项 | 6 passed / 70.24秒 |
| 完整回归 | 215 passed / 1条已有TorchScript弃用警告 / 715.36秒 |
| 连续测量器重排/改为组内不恢复后的最终专项 | 6 passed / 68.58秒 |
| 新增模块/专项Ruff、格式 | 通过 |
| 新脚本bash语法/帮助 | 通过 |
| diff空白检查 | 通过 |

最终测量器调整在完整回归运行期间完成，因此单独复测全部6项；没有将完整215项冒称为调整后的再次全跑。该调整仅涉及诊断模式，不改变正常训练数学。

测试验证源位置hook的数值/梯度及异常恢复、view函数身份保留、记录前后实际更新/Adam/RNG一致、GC回调恢复、完整Profile归属入口以及 `aot_eager` plant仍使用原编译路径。CPU通过不证明CUDA2.7/Inductor通过。

原始日志：[完整回归](pytest_full.log)、[初次专项](pytest_initial.log)、[连续测量器复测](pytest_final.log)。
