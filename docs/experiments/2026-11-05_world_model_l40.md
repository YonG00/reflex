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

## 曾准备的可选 bootstrap（未执行，当前不采用）

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

## 当前任务：只验证 WM，固定随机数据

用户明确当前先做随机采样的 WM 预测检查。上述热启动与阶段均衡仅为可选代码，未执行，也不用于当前对照。Actor 不更新。

核对 SuperMimic-Reproduced：`WMTrainerCfg.window=8`，`PolicyTrainerCfg.horizon=32`。前者是记录动作驱动的自回归训练窗口，后者是 Actor 更新的想象长度，两者不同是原算法的设计，不能据此认定错误。首轮使用 4 是本实现自选配置，不是参考默认；当前默认已改为 8。

新增 `reflex.world_model.offline` 独立入口，不构造 Actor、Critic 或 AMP。读取第 40 轮固定 replay，所有对照使用相同初始 WM/Adam、同一随机起点序列和固定 held-out batch：

```bash
python -m reflex.world_model.offline \
  --checkpoint outputs/world_model/20261105_l40_smoke/checkpoint_00040.pt \
  --urdf source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf \
  --updates 64 --batch 128 --cases 4:1,8:1,8:82 \
  --output outputs/world_model/20261105_l40_wm_only
```

`state_weight=82` 对应当前 82 维归一化状态残差的维度求和，而不是求均值；参考代码也是逐状态分量求和。该对照用来检验当前状态损失相对 reward/memory/termination 辅助项的权重影响，并非已经证明的修复。不同窗口下模型计算量不同，报告单独列出优化步数、模型转移计算量和时间；不把它们视为等计算量实验。
