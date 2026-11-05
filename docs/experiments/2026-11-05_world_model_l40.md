# L40 世界模型工程验证与诊断

服务器：用户经网页终端核验的 L40 实例；项目 `/root/gpufree-data/reflex`。不要把无卡 CPU 测试等同于本记录的 GPU 检查。

## 首轮：随机 Actor，64 环境

代码 `1ef087c`，原 `configs/world_model/default.json`，Actor 随机初始化（输出层为零）、WM 拟合窗口 4、想象窗口 32。

- 新服务器 CPU 全部 10 项测试通过。
- 512 控制步预检通过，32768 条真实交互，32 步有效发射后窗口 1135 个。
- 真实 Oracle observation 最大误差 `1.7508864e-6`；AMP 特征最大误差 `2.2053719e-6`；逐步真实动作目标校验通过。
- 小规模试训本来上限 60 轮，因多步误差不再明显改善，已停止；保留 `outputs/world_model/20261105_l40_smoke/checkpoint_00040.pt`。不要把未完成的 60 轮或被停止前未存档的更新当作持久存档。
- 持久 checkpoint 的 Actor 更新次数为 **0**。门限没有被绕过，也没有正式新策略效果结论。
- 离线 GPU 梯度检查通过：真实 replay、32 步、batch32，task/style Actor 梯度非零；Actor/Critic 更新可计算，WM/判别器权重未变。该检查修改的 Actor 随进程丢弃，没写回实验 checkpoint、没在仿真中执行。

## 已确认的问题

连续性检查覆盖 14134 条相邻有效边：`next_state[t]` 与 `state[t+1]` 最大差 `9.536743e-7`，memory 差为零；没有发现跨 reset/发射的时间错位。

| 可用窗口长度 | 训练集几何阶段计数 0/1/2/3 | 验证集计数 |
|---|---|---|
| 4 | 278 / 2751 / 1255 / 6131 | 38 / 812 / 352 / 1616 |
| 16 | 117 / 1019 / 145 / 5072 | 22 / 337 / 38 / 1310 |
| 32 | 15 / 230 / 5 / 1535 | 0 / 96 / 2 / 400 |

阶段是距离/速度几何分箱，不是真实接触标签。长窗口几乎被低速近距离状态占据；其中可能包含机器人/箱子已经倒地，不能统称“成功抱持”。64 环境下当前帧的有效排序样本也稀少，初期多次未达至少 16 对的要求。

第 30 轮存档固定种子 held-out 审计：16 步 box position RMSE 约 0.099m，32 步约 0.244m（每坐标分量 RMSE）；箱子角速度误差也较大。原 4 步拟合不足以证明 32 步长期预测可靠。

## 针对性修正：工程热启动

新增 `configs/world_model/contact_bootstrap.json`：16 步 WM 拟合、阶段均衡训练/想象采样、256 个环境用于后续 GPU 检查。验证集仍按 env ID 隔离、保持原分布；无样本阶段不会被伪造。原配置保留，不改成功标准，不放宽模型或排序门限。

新增 `--warm_start_actor`：从原 167/170 维 PPO Actor 复制公共前 166 维输入及后续层，blind/trust 输入按理想观测固定为零，新 Oracle 列零初始化但可训练。Actor 隐层配置显式匹配旧 Teacher。新 WM、Critic、优化器、replay、AMP 均重新初始化；这属于工程 bootstrap，**不是完整 resume，也不是从零效率对照**。加入 observation normalizer 时，对第一层做对应仿射变换，保留未发生 clipping 的热启动策略输出，并有 CPU 回归。

热启动来源 SHA、初始化类别会写入 manifest 和 checkpoint，不能在结果中省略。第一轮数据不会混入新实验 replay。

```bash
source scripts/local/env.sh
bash scripts/train_stage1_world_model.sh --mode preflight --num_envs 256 \
  --config configs/world_model/contact_bootstrap.json \
  --warm_start_actor data/checkpoints/teacher_stage1.pt --preflight_steps 512 \
  --output outputs/world_model/20261105_l40_contact_preflight
```

本节新增配置在提交时仅完成 CPU 检查，GPU 结果应在实际运行后补充，不得从首轮成功预检推断本轮已通过。
