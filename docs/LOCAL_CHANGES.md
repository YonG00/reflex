# 本地改动记录

| 日期 | 改动 | 原因 | 已验证 | 待验证 |
|---|---|---|---|---|
| 2026-11-04 | 评测器新增可选 `--video_path` | 保存教师、潜变量和视觉策略的效果视频 | Python 编译、diff 检查 | GPU 渲染、有效帧、MP4 可播放、评测 JSON 完整 |
| 2026-11-04 | `scripts/local/env.sh`、`build_data.sh` | 统一环境和动作数据生成入口 | shell 语法 | Isaac Sim 动作生成与 AMP 数据完整性 |
| 2026-11-04 | `scripts/local/train.sh` | Stage 2 PPO 显式使用自己的 BC 输出 | shell 语法 | train→save→load→play |
| 2026-11-04 | `scripts/local/play.sh` | 三个独立种子的单环境录像 | shell 语法 | 三段视频与结果文件验收 |

上游基线：`657c77d98f847fb363a3f25c4c1e60c25c03cb65`。原始代码来源和许可证不变。后续修改应记录对应提交、运行配置、checkpoint 和实际结果，分清“实现”“语法通过”“GPU 跑通”和“效果通过”。
