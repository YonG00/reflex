# Reflex 实验分享

这是基于 Reflex 官方开源代码的个人实验仓库，**不是官方项目，也不是 SuperMimic 的完整复现**。保留上游作者署名、MIT 许可证及第三方声明。上游代码基线为 `ToberJ/reflex@657c77d`。

## 三条实验线

| 分支 | 内容 | 阅读入口 |
|---|---|---|
| [experiments/supermimic-wm](https://github.com/YonG00/reflex/tree/experiments/supermimic-wm) | 借鉴 SuperMimic 的世界模型、可微想象与解析软终止；包含多轮实现和效率探索 | `docs/EXPERIMENT.md` |
| [experiments/box-info-19d](https://github.com/YonG00/reflex/tree/experiments/box-info-19d) | 从官方基线仅增加箱子输入，保留 PPO+AMP、原奖励和隐藏层 | `docs/EXPERIMENT.md` |
| [experiments/reward-hold](https://github.com/YonG00/reflex/tree/experiments/reward-hold) | 在增维 PPO 上修改缓冲和持续抱持机制；包含 C 版动作缩放和参考跟踪调整，不是纯单项奖励消融 | `docs/EXPERIMENT.md` |

`ppo/*` 是历史开发分支，保留追溯；公开阅读优先使用以上三个入口。三个分支各自完整，不需要把另一个分支的目录复制进来。`main` 保留世界模型开发历史并提供导航，不代表最优策略。

## 已完成的 600 更新对照

| 方案 | 最终持续接住（主要指标） | chest strict（曾达到严格条件） |
|---|---:|---:|
| 官方代码从零训练 600 更新，170/292 维 | 98/640 = 15.31% | 73/640 = 11.41% |
| 增加箱子信息，183/305 维 | **116/640 = 18.13%** | 79/640 = 12.34% |
| C 版定时恢复，从零 600 更新 | 101/640 = 15.78% | 108/640 = 16.88% |
| 最终持续抱持版，从零 600 更新 | 97/640 = 15.16% | 110/640 = 17.19% |

最后一版**没有改善最终接稳率**：比增维基线少 19 次，比原版少 1 次。相对原版基本持平，不能包装成提升。增维的观察结果较好，但只有一个训练 seed，尚不足以证明稳定优势。

以上为同预算一阶段 PPO：训练 seed42、4096 环境、600 次更新（0–599，每次24控制步）；CLEAN2 regular，评测 seed8/9 各320环境、640控制步，仅统计首投。600 是 PPO 更新次数，不是600个环境步。官方发布 teacher 使用多轮 warm start，不能将此表称为论文发布模型复现，也不能与世界模型的单回合演示直接比较。

最终持续接住要求曾达到 twin 抱持条件，之后未掉落、机器人未跌倒且结束箱速 <1m/s；strict 是曾连续满足严格几何/速度条件0.6秒的锁存事件，之后仍可能掉落。总训练奖励因奖励定义不同不能直接排名。详细数值与局限见 [实验经验](docs/EXPERIMENTS_ZH.md)。

## 使用与复现

先选择分支，再按 `docs/EXPERIMENT.md` 执行。原官方安装、数据重建、许可证和引用说明保存在 [原版 README](README_UPSTREAM.md)、`docs/REPRODUCE.md` 和 `THIRD_PARTY_NOTICES.md`。

```bash
git clone --branch experiments/box-info-19d https://github.com/YonG00/reflex.git
cd reflex
```

环境沿用 Isaac Sim5.1、Isaac Lab2.3.1、rsl-rl-lib3.0.1、PyTorch2.7/CUDA12.8。依赖、机器人资产及授权动作数据需要自行安装或重建；本仓库不重新分发受限动作数据、训练权重、视频和服务器环境。历史文档中的本机/服务器路径是当时的证据位置，不是公开下载地址。

公开整理未改变算法或启动训练。参见 [公开范围与发布检查](docs/PUBLIC_RELEASE.md)。

当前分支：`experiments/reward-hold` — 接触后持续抱持奖励实验。
