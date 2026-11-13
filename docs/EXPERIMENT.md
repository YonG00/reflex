# 一阶段 PPO 箱子信息扩展

公开分支：`experiments/box-info-19d`；整理前代码基点：`5586ca7`。本次整理不改变运行算法。

## 改了什么

以官方 `657c77d` 为起点，只增加一阶段 Actor/Critic 的箱子信息。公开分支回到 `5586ca7`，不混入后来碰撞/弹飞/缓冲奖励代码。
Actor170→183，Critic292→305；新增姿态6D、角速度3D、尺寸3D、质量1D。原箱子位置和线速度仍保留。线速度转到机器人坐标系但不减机器人速度；该观测与奖励中的相对速度不是同一定义。
箱子19维不是连续切片：位置160:163、速度163:166、姿态170:176、角速度176:179、尺寸179:182、质量182；166:170仍是原延迟/置信信息。
不改变 PPO+AMP、隐藏层、原奖励、动作缩放和事件。旧170维模型不能直接当183维模型resume。

## 环境与数据

按 [官方安装说明](../README_UPSTREAM.md) 安装依赖并执行资产准备。使用自己的授权动作数据重建 motion 与 AMP dataset。将代码安装进 Isaac Lab 所用的 Python 环境，参阅原安装步骤。

```bash
export ISAACLAB_PY=/path/to/isaaclab/python
export REFLEX_DATA_DIR=/path/to/reflex-data
```

这里的路径是占位符，需替换；不依赖历史服务器的 runtime.env。先确认 GPU、数据和输出位置再启动，以下命令会实际训练。

## 训练与评测

```bash
bash scripts/train_stage1_box_info.sh --num_envs 4096 --max_iterations 600 --seed 42
for seed in 8 9; do
  bash scripts/eval.sh teacher_box_info --ckpt /path/to/model_599.pt \
    --condition regular --seed "$seed" --num_envs 320 --out "outputs/box600_s${seed}.json"
done
```

不传 warm_start/resume 即从零训练。原版对照使用原task的 `scripts/train_stage1.sh` 和 `scripts/eval.sh teacher`，不要混用输入维度不同的权重。
结果116/640最终接稳；[训练记录](experiments/2026-11-09_STAGE1_BOX_INFO_SCRATCH600.md)中“尚无原版对照”是当时状态，后续对照见 [最新汇总](EXPERIMENTS_ZH.md)。
