# 本仓库的训练与录像入口

本仓库基于 [ToberJ/reflex](https://github.com/ToberJ/reflex) 的 `657c77d98f847fb363a3f25c4c1e60c25c03cb65`，保留原提交历史、MIT 许可证和第三方声明。上游 README、REPRODUCE 和 STAGE3 文档仍是算法与复现口径的依据。

## 当前验证边界

2026-11-04：本地启动脚本通过 shell 语法检查，录像修改通过 Python 编译检查；尚未完成训练、闭环评测或视频渲染验证。GPU 运行结果不得由这些语法检查推断。

版本目标为 Isaac Sim 5.1.0、Isaac Lab v2.3.1、PyTorch 2.7.0/CUDA 12.8、rsl-rl-lib 3.0.1。新镜像的 Sim、PyTorch、rsl-rl 版本与目标匹配；Isaac Lab 扩展依赖仍需安装并验收。

## 资源准备（不需要 GPU）

已有备份时优先按 [最小数据恢复说明](DATA_BACKUP_ZH.md) 恢复六个必要文件，无需重新拉取整个 G1 压缩包。

依照上游 README 安装项目，并下载官方 checkpoint、机器人描述和所需动作。数据、权重、日志及视频分别位于被忽略的 `data/`、`logs/`、`outputs/`，不上传 Git。

BONES-SEED 需要使用本人账号接受条款。当前上游只提供 `g1.tar.gz` 压缩包，Reflex 仅使用其中 `g1/csv/210707/walk_forward_grab_normal_003__A006_M.csv`。不要重新分发此数据或生成的参考动作。AMP 还需要 assets/motions/README.md 中指定的 AMASS B10 动作。

## 开卡后的验证顺序

激活已经安装完整依赖的环境，然后执行：

```bash
# 可选：指定环境 Python；默认使用当前 python
export ISAACLAB_PY=/path/to/environment/bin/python
bash scripts/local/build_data.sh
bash scripts/local/play.sh teacher
bash scripts/local/play.sh true_z
bash scripts/local/play.sh visual
```

`build_data.sh` 使用 Isaac Sim 生成参考动作和 AMP 数据。`play.sh` 每种模式使用种子 8、9、10，各运行一个环境，分别保存 JSON、日志和 MP4 至 `outputs/play/<时间>_<模式>/`。这只是效果查看，不是论文的 640 次投掷基准评测。

论文口径评测继续使用上游 `scripts/eval.sh` 的 320 环境、两个种子配置。

## 训练下一阶段

```bash
bash scripts/local/train.sh bc
bash scripts/local/train.sh ppo --num_envs 4096 --max_iterations 600 --run_name stage2_local
# 使用新 checkpoint 录像
bash scripts/local/play.sh true_z --ckpt /path/to/model_599.pt
```

PPO 明确从本次 BC 的 `data/stage2/bc_init.pt` 初始化，避免误用上游脚本默认的已发布 BC 权重。运行前先做短程 train→save→load→play 验证，完整训练时长与效果以实际日志为准。

所有新增入口均需人工启动；资源下载完成不会自动启动 GPU 训练。

## 当前实验范围：Stage 1（2026-11-05）

当前只检查官方 Stage 1 checkpoint，以及公开配置从随机初始化训练的 Stage 1 checkpoint。Stage 2 的两轮功能检查不属于本实验，后续 Stage 2 工作已停止。

```bash
# 从零训练：不加载预训练权重，不恢复旧训练
bash scripts/local/train.sh stage1_scratch --num_envs 4096 --max_iterations 600 --seed 42 --run_name stage1_scratch_20261105
# 从官方教师续训必须明确选 stage1_continue
```

公开配置是教师训练序列最后一轮配置，从零运行不等于完整重放论文教师的训练过程。本次结果将与官方教师在 regular 条件、seed 8/9、每 seed 320 环境下比较，使用上游 sustained_hold 指标。

数据准备已完成：BONES CSV 与 AMASS B10 的 checksum 均通过；参考动作 1500 帧、50 fps；AMP 数据为 `(6576, 2, 83)`，有限数值检查通过。官方教师录像已生成，从零训练结果尚待完成。
