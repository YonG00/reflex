# 本地改动记录

2026-11-05：新增 Stage 1 联合机器人/箱子世界模型实验、可微想象更新、完整存档与 Oracle PPO 对照。CPU 合成训练恢复测试已通过；真实 Isaac 和效果均等待开卡。详细设计及命令见 `docs/world_model/IMPLEMENTATION_ZH.md`。

| 日期 | 改动 | 原因 | 已验证 | 待验证 |
|---|---|---|---|---|
| 2026-11-04 | 评测器新增可选 `--video_path` | 保存教师、潜变量和视觉策略的效果视频 | Python 编译、diff 检查 | GPU 渲染、有效帧、MP4 可播放、评测 JSON 完整 |
| 2026-11-04 | `scripts/local/env.sh`、`build_data.sh` | 统一环境和动作数据生成入口 | shell 语法 | Isaac Sim 动作生成与 AMP 数据完整性 |
| 2026-11-04 | `scripts/local/train.sh` | Stage 2 PPO 显式使用自己的 BC 输出 | shell 语法 | train→save→load→play |
| 2026-11-04 | `scripts/local/play.sh` | 三个独立种子的单环境录像 | shell 语法 | 三段视频与结果文件验收 |

上游基线：`657c77d98f847fb363a3f25c4c1e60c25c03cb65`。原始代码来源和许可证不变。后续修改应记录对应提交、运行配置、checkpoint 和实际结果，分清“实现”“语法通过”“GPU 跑通”和“效果通过”。

| 日期 | 后续修正 | 验证证据 | 边界 |
|---|---|---|---|
| 2026-11-05 | 启动时加入当前 Python 的 torch/lib 动态库目录 | 修复 CUDA linalg 的 dlopen 错误；参考动作和 AMP 数据生成完成 | 未修改训练算法 |
| 2026-11-05 | 录像关闭 motion/contact 调试标记并拉近视角 | 官方教师 seed 10 录像已生成 | 仅调整画面，不改变评测规则 |
| 2026-11-05 | 明确区分 stage1_scratch 与 stage1_continue | 从零训练实际命令没有 warm_start/resume 参数 | 结果待验收 |

2026-11-05 后续验收：Stage 1 从零完成 600 iterations，`model_599.pt` 重新加载通过；同条件 640 次投掷接物成功率为 15.3%，官方教师为 87.7%。训练、评测结果与最终权重校验记录见 `docs/experiments/2026-11-05_stage1_scratch.md`。

2026-11-05 续训：新增教师训练参数 `--resume_total_iterations`，用于明确累计目标、避免重跑最后一个已完成索引，并恢复 PPO 自适应学习率标量。旧 checkpoint 缺少 AMP 判别器状态的限制已在 `docs/experiments/2026-11-05_stage1_resume600_to2000.md` 记录；本次为 600 → 2000，结果待验证。

2026-11-05 用户追加要求：完整 AMP 存档已实现；CPU 下一步更新一致性和实际 GPU 保存/恢复验证通过。旧进程已暂停，新格式从 `model_652.pt` 起包含 AMP。后续正式模型使用 4096 验证分支的 `model_656.pt`，继续至累计 2000。详见 `docs/AMP_CHECKPOINT_ZH.md` 和 `docs/LOCAL_PENDING_CHANGES.md`。
