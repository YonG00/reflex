# 2026-11-11 Stage3 独立消融上机记录

服务器：183.147.142.40:31788，RTX4090 24GB，IsaacSim5.1、IsaacLab2.3.1、torch2.7.0+cu128、rsl_rl3.0.1。
工作副本：`/root/gpufree-data/reflex-s3-ablations`。
旧机器通过无卡模式读取原始权重，不使用其他模型替代。

## 输入一致性

|文件|SHA256（新旧机器一致）|
|---|---|
|model_599.pt|9a595d778ff6fd114efe5e0adc09daacb574e2caf985aa94b7c570abea0a2ab6|
|walk_then_catch_v17_30s.npz|10aeef1894b46d4f01d8a3ad1d1e077b8c463a2ce3652ec93bda8bf745014655|
|amp_dataset.pt|4db94680833df13d0cc7f6ed0983cc7387f5bc0ca75bc6caa1c6c50def4c61b9|
|main_hands.urdf|e91bc63b1550b874262b1c73b2e56cc64dd57b88576cb8b41c991bb4c2c850b5|

权重iter599、actor[512,183]、critic[512,305]，原PPO优化器可恢复。
不存在AMP判别器历史状态，按原版流程初始化，不声称恢复了未保存的状态。

## 短训实测

四组都从同一原599独立native resume，seed42，128环境×12更新，iteration599..610。
每组288控制步，36864环境步，无效安全/地面状态下的intake/retain付费审计计数均为0。

|组|追加更新|intake raw步均值最大值|retain raw步均值最大值|有效接触/发射累计|
|---|---:|---:|---:|---:|
|S3|12|0.023283|0.076468|29/126|
|A|12|0.023283|0.076904|30/140|
|B|12|0.038031|0.076347|31/132|
|C|12|0.023040|0.070947|29/139|

这些是短训中的累计状态和奖励触发诊断，不能当作matched评测成功率。
B前窗累计17.680s、后窗8.760s；后仰/顺势移动/下降子项非零样本分别250/471/758。
C肩部scale跨环境均值最大0.026467（多数环境不在接球窗）；目标裁剪最大0。
C目标步间变化最大0.513383rad，包含原two-stage参考变化，缺少同口径baseline目标变化量，
因此不能仅凭这个数字宣称动作跳变已经完全安全；正式训练须继续观察站立/终止统计。

S3/A/B证据：`/root/gpufree-data/s3_short_20261111_r2/`。
C修复后证据：`/root/gpufree-data/s3_C_short_20261111_r3/`。

## 发现与修复

- 新机初次仿真因动态库搜索路径缺torch/lib而退出，补充运行环境路径后正常启动。
- C真实RewardManager验证拒绝遗漏的原`inner`参数；新包装显式转发原trust/reactive参数。
  添加该真实签名场景的测试，本地与服务器均119项通过，C重新从原599短训成功。
- 原S3/A/B奖励逻辑未因C签名修复改变。
- S3的小样本共享评测已产出，A评测进行中时用户要求停止评测直接正式训练。
  剩余评测已停止；这些结果不作为消融效果结论。

## 正式训练授权与状态

用户已授权直接串行正式训练，不再等待额外评测。
计划S3/A/B/C各4096环境×200次追加更新，均从原599开始，seed42，最后目标迭代798附近。
本轮采用`train`模式，不自动评测、不将前一组checkpoint作为后一组起点。
正式队列已启动（2026-11-11，服务器本地日志约09:13）：

- 训练源码提交：`c7d6ac9`，服务器工作区干净。
- 队列PID：`11018`；首组训练Python PID：`11061`。
- 输出目录：`/root/gpufree-data/s3_formal_200_20261111/`。
- 队列日志：`/root/gpufree-data/s3_formal_200_20261111.launch.log`。
- 查看当前组：`cat /root/gpufree-data/s3_formal_200_20261111/status.txt`。
- 各组日志：目录内`S3/A/B/C.train.log`；最终checkpoint路径由`*.checkpoint.txt`单独记录。
- 运行方式为nohup，SSH断开不影响串行队列；异常退出即停止，不跳过失败组。

这是启动记录，不能当作200更新已经完成。具体完成情况以迭代日志、check.json和status.txt为准。

## C从零600更新实验

用户另行授权C从零训练600更新，单独运行，不加载model_599或任何续训权重。
服务器启动脚本：`/root/gpufree-data/run_C_scratch600_20261111.sh`；启动队列PID61207。
结果目录：`/root/gpufree-data/C_scratch600_20261111/`。
采用原单分支Actor183/Critic305，C奖励与肩部ramp，与双分支Actor实验分开。
这是启动记录，尚未核验最终600更新完成。

可复现入口：`bash scripts/train_C_scratch600.sh`。默认4096环境、seed42、600次更新；
拒绝resume/warm-start参数，保留原初始化、发射、重投及全部事件。
