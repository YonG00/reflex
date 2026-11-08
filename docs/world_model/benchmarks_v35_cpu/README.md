# V35 CPU验证

macOS、Python3.12.15、Torch2.15.0，无CUDA；aot_eager用于完整图捕获和梯度/Adam正确性，不证明CUDA/Inductor舍入一致或提速。

- candidate_final.log：最终16项通过，25.33秒。8种编译边界的前向/输入梯度/模型不变性；组件/VJP；H32及实际Actor Adam；参数漂移坐标；细分MLP与进度；无COM分支；MLP-only和layerwise的真实Replay H8条件化WM监督/原裁剪/实际WM Adam；拒绝候选自身作为参考。
- related.log：40项通过，95.04秒。V35最初专项＋既有V31最小链/第三方适配及V34事件/恢复相关检查；与最终专项重叠，不相加当作独立测试总数。随后增加的细分层、COM和H8等由candidate_final覆盖。

新文件Ruff、shell bash -n、CLI --help和git diff --check通过。正式models.py、physics.py、trainer.py、原奖励/事件未改。GPU尚未测，所有候选未放入正式训练。
