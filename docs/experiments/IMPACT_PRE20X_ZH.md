# 碰撞前相对速度惩罚 ×20 对照实验

独立分支 `ppo/impact-pre20x`，由 `ppo/box-info-19d` 的 `11a5910` 克隆。

仅改变新奖励内部的 `w_pre`：0.2 → 4.0（20 倍）。`w_damp=0.5`、`w_escape=1.0`、碰撞后窗口 0.5s、其他奖励、观测、网络与训练参数保持不变。不是将全部速度惩罚或全部奖励放大 20 倍。

与原权重续训实验从同一份昨晚的 `model_599.pt` 恢复 PPO 权重与优化器，4096 环境、seed 42、追加 200 轮。原版 checkpoint 不含 AMP 判别器状态，两组均按原加载流程重新初始化判别器。

入口：

```bash
bash scripts/train_stage1_impact_pre20x.sh --num_envs 4096 --max_iterations 200 --seed 42 \
  --run_name impact_pre20x_resume200 --resume True \
  --load_run 2026-11-09_19-54-01_box_info_scratch600_s42_20261109 --checkpoint model_599.pt
```

在独立副本的 `logs/rsl_rl/reflex/` 下为该基线 run 建只读用途的符号链接；新 run、独立日志和评测输出都写入新副本。原权重实验继续运行。

重点比较：接触前接近速度、首次接触速度、手臂接触率、near_missed、提前终止、严格成功与 CLEAN2 持续接住率，防止把躲避导致的低惩罚误判为缓冲改善。并行训练会争用 CPU/GPU，吞吐以实测为准。
