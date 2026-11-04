# Stage 1 公开配置从零训练

状态：已启动，尚未完成。

- 配置：`configs/train/stage1_teacher.env`，4096 环境、600 iterations、seed 42。
- 初始化：随机初始化；没有 `--warm_start`、`--warm_start_ckpt` 或 `--resume` 参数。
- 数据：BONES CSV checksum 通过；参考动作 1500 帧、50 fps；AMP `(6576, 2, 83)`，有限数值检查通过。
- 日志：`outputs/experiments/stage1_scratch_20261105/train.log`。
- 进度：同目录 `status.json`；配置与源码差异保存在 `manifest.json`、`source.diff`。
- 训练完成后：官方教师和从零训练教师分别使用 regular、seed 8/9、每 seed 320 环境评测；新教师使用 seed 8/9/10 录制单环境视频。
- 结果口径：上游 `reflex.eval.summarize` 的 sustained_hold，不将单环境视频的通过率当作论文基准。

上游说明：公开配置是多轮教师训练的最后一轮，单独从零运行不保证复现发布教师模型。结果以本次测量为准。
