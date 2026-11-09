# 一阶段官方 PPO+AMP 扩展箱子信息：从零训练 600 轮

服务器副本：`/root/gpufree-data/reflex-ppo-box-info`，分支 `ppo/box-info-19d`。官方基线 `657c77d`；服务器训练代码提交 `39b290f`（以补丁移植，与 GitHub `63c5ff0` 的运行代码一致；本地版本另有首次 reset 单元测试与说明更新）。原改造仓库保持独立。

训练：4096 环境，seed 42，600 次 PPO iteration（索引 0–599），不加载权重或 resume；actor 183、critic 305；保留官方 PPO+AMP、隐藏层、奖励与物理配置。正式流程 2026-11-09 19:53:53 启动，训练及两组评测于 20:31:30 完成（北京时间）。

最终权重：`logs/rsl_rl/reflex/2026-11-09_19-54-01_box_info_scratch600_s42_20261109/model_599.pt`。已检查 iter=599，actor 首层 `[512,183]`、critic 首层 `[512,305]`，模型 state_dict 中全部张量有限值。

## CLEAN2 评测

标准箱子 regular；seed 8/9 各 320 环境，每环境一次投掷、640 控制步。使用最终权重，原版 summarize 口径。

| seed | 投掷 | 持续接住 | 拦截 | chest strict | 跌倒 |
|---|---:|---:|---:|---:|---:|
| 8 | 320 | 60 | 126 | 40 | 0 |
| 9 | 320 | 56 | 100 | 39 | 0 |
| 合计 | 640 | 116 (18.125%) | 226 (35.3125%) | 79 (12.34375%) | 0 |

持续接住：twin 条件曾成立、之后未掉落、机器人未跌倒，并且回合结束箱子速度 <1 m/s。原 summarize 输出 18.1% ±3.0 百分点（95% CI）。不能将 chest strict 与持续接住混为同一指标。

尚未运行相同 seed/预算的原版 170 维对照，不能据此证明额外箱子信息提升了训练。官方说明中，发布 teacher 本身来自多次 warm start；本次从零 600 轮也不等价于复现发布权重。

服务器输出：`outputs/box_info_run/`，包括 train.log、regular_s8.json、regular_s9.json、eval_s8.log、eval_s9.log、manifest.json、checkpoint.txt、status.txt、run_train_eval.sh。

原始批量评测未开启渲染，没有视频。后续单环境演示通过新增 `--video_dir` 补录，使用同一最终权重；单环境新回合不能视作原 640 个批量回合的原样重放。

```bash
bash scripts/eval.sh teacher_box_info --condition regular --seed 8 --num_envs 1 \
  --ckpt /实际路径/model_599.pt --out outputs/box_info_run/video_s8.json \
  --video_dir "$PWD/outputs/box_info_run/videos_s8"
```

上机发现并解决：尺寸缓存在首次 reset 前不存在（观测维度探测改为读取已生成几何）；CUDA linalg 动态库搜索需在 LD_LIBRARY_PATH 中包含当前 Python 的 torch/lib。后者为启动环境修复，不改变算法。
