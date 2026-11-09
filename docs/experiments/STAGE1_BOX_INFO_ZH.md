# 一阶段 PPO 箱子信息扩展实验

## 基线与范围

副本分支：`ppo/box-info-19d`。官方基线分支：`official-baseline`，提交 `657c77d`（原仓库本地保存的 `upstream/main`；本次未联网确认是否有更新）。副本由本地 Git 克隆，不包含主仓库未提交改动、数据和训练输出。

新增独立任务 `Reflex-Teacher-BoxInfo-G1-v0`。原版任务、原版启动脚本、二三阶段不变。继承官方 teacher，只追加观测；保留 PPO+AMP、网络隐藏层/激活、奖励、动作、终止、随机化与全部原版训练超参数。网络首层输入宽度由 observation manager 和原版 runner 自动确定，不硬编码新的网络。

## 输入合同

官方配置 `BOX_V38_LAND_OBS=0` 下，actor 预期从 170 变为 183，critic 从 292 变为 305；实际总维度仍需 Isaac Lab 启动日志确认。两个组都新增 13 维，加上已有箱子位置/线速度 6 维，箱子信息共 19 维。保留原版列顺序便于对照和官方 warm start，因此图片的连续 `160:179` 布局不直接适用。

| 信息 | 维度 | actor 索引 | critic 索引 |
|---|---:|---|---|
| 原有箱子位置 | 3 | `160:163` | `286:289` |
| 原有箱子线速度 | 3 | `163:166` | `289:292` |
| 原有 blind_age / trust | 4 | `166:170` | 无 |
| 新增箱子姿态 | 6 | `170:176` | `292:298` |
| 新增箱子角速度 | 3 | `176:179` | `298:301` |
| 新增箱子完整尺寸 | 3 | `179:182` | `301:304` |
| 新增箱子质量 | 1 | `182` | `304` |

姿态：机器人 root 相对旋转 `R_root^T R_box` 的前两列，按行展开为 `[r00,r01,r10,r11,r20,r21]`；恒等姿态为 `[1,0,0,1,0,0]`。角速度：箱子世界系角速度旋转到机器人 root 坐标系（rad/s），不减机器人角速度。尺寸：现有 `_box_half_extents` 缓存乘二，箱子局部 x/y/z 完整边长（m），不使用随朝向变化的世界 AABB。质量：读取当前 PhysX 质量（kg），包括质量随机化或课程更新，不按型号猜质量。

原版位置/线速度完全保留；无感知噪声时，线速度仅旋转坐标系，不减机器人线速度。官方 stage1 配置 `BOX_DP_EST=1`，actor 原有位置/速度仍经过原有模拟感知与滤波，而 critic 使用真值。新增 13 维在两组都使用仿真真值且不加噪声。这是额外真值信息实验，并非硬件端观测能力已就绪。

## 运行

在已经安装 Isaac Lab、机器人资源和训练数据的环境中，从副本根目录运行：

```bash
bash scripts/train_stage1_box_info.sh --num_envs 4096 --max_iterations 600 --seed 42 --run_name stage1_box_info_s42
```

数据默认位于副本的 `data/`；如另有数据目录，用 `REFLEX_DATA_DIR=/实际数据目录` 指定。克隆不会复制被 Git 忽略的数据与机器人资源，需按原版 README 准备。

原版对照：同一环境、同一 seed/训练预算运行 `bash scripts/train_stage1.sh`。不切换到 world model 或其他 PPO 实现。

优先从零训练进行对照。若需要官方 teacher 权重 warm start，使用原版脚本支持的 `--warm_start /path/model.pt`：原版加载器会为扩大的 actor/critic 首层追加零列。不要用普通 resume 直接加载旧维度 checkpoint；旧 checkpoint 与新增维度不兼容。已有二阶段代码有固定 170 维 teacher 假设，本实验的 183 维 checkpoint 尚未适配二阶段。

## 已验证与待验证

- CPU：4 项单元测试通过，覆盖姿态参考系/6D 顺序、角速度旋转、混合尺寸/动态质量、两组追加顺序和 13 维形状。Isaac Lab 数学与配置接口使用替身，不等价于真实仿真验证。
- shell 语法、Python 编译、Git diff 空白检查通过。
- 本机未安装 Isaac Lab；未启动 GPU 训练，未验证真实场景初始化、manager 最终维度、PhysX 调用性能或训练收益。
- 上机先以少量环境短跑，核对 policy=183、critic=305、观测有限值、首次 reset/后续 reset 均正常，再进行相同预算对照。结论需基于训练与评测结果。

CPU 检查命令（需 PyTorch）：`python -m unittest discover -s tests -v`。
