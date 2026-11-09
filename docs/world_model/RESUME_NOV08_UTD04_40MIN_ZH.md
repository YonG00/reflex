# 昨晚 H16 模型续训：UTD 0.4、Return Reset、40分钟

状态：已在31428启动40分钟续训；来源SHA校验通过，完整GPU恢复通过，已观察Actor805→828、WM13050→13280，暂无OOM。

- 来源：`reflex_scratch_h16_b4096_joint30_20261108_r1/checkpoint_final.pt`，应为 Actor805、WM1305轮。禁止误用今天 Actor249 的最终模型。
- 4096真实环境，Replay512，WM/Actor batch3584，H8/H16。每步采集，UTD0.4（平均每2.5向量步一次WM10＋Actor1），没有连续10步采集设置。
- 用户明确要求启用 Return-based Reset Sampling：50%均匀＋40%当前低回报＋10%历史权重。旧模型没有该统计，必须显式`--initialize_return_reset --reinitialize_replay`；仅新建采样器统计和Replay，完整恢复权重、优化器、AMP、归一化与训练进度，不做500轮WM预热。
- 运行40分钟，`--max_seconds 2400`从训练循环开始，包含新Replay采集/期间更新冷启动；不使用首次Actor后重新计时的joint预算。Isaac启动/模型加载与最终保存单列为额外墙钟开销。
- 昨晚WM编码为legacy，严格保持编码，不能套stable_heading_v1 Graph检查点。保留昨晚WM编译配置（不启用新H8 Graph）；启用Critic4096行分块与激活重算以控制显存。不能承诺今天stable模型约0.426秒的WM速度直接适用于此模型。
- 原奖励、网络、事件、软终止不变。新进程不恢复旧PhysX瞬时状态，参考起点正常重采样，Replay不得跨进程拼接。

## 开卡后入口

先确认来源checkpoint真实存在、GPU/进程状态和最新导入代码，再执行（输出目录必须不存在）：

```bash
source /root/gpufree-data/reflex_v36_env.sh
bash scripts/resume_nov08_h16_utd04_40min.sh \
 /root/gpufree-data/reflex_nov08_source/checkpoint_final.pt \
 /root/gpufree-data/reflex_resume_nov08_utd04_return40_NEW
```

脚本先核对Actor805及关键源配置，保存独立batch迁移副本，不覆盖源模型；训练加载器再做完整契约校验。缺模型/契约不匹配即停止，不静默变成从零训练。用户允许从零作为备选：如果旧模型找不到或不能严格恢复，使用独立从零配置`configs/world_model/joint_scratch_h16_b3584_utd04_return40.json`（max_seconds=2400），40分钟包含500轮WM预热，报告实际Actor时间；不要混用两种结果。

结束检查：final checkpoint、training_phases、TensorBoard、实际配置、每项耗时、增量更新/真实交互量。另做隐藏坐标轴的15秒独立PhysX评测；与昨晚统一评测起点分布，训练Return Reset不作为公平评测分布。

## 本地验证

Return Reset采样/统计、显式恢复迁移、旧模型无授权拒绝迁移、既有统计禁止丢弃、UTD恢复、batch迁移和效率/预热恢复相关CPU测试21项通过；shell语法与配置构造通过。随后在31428完成真实昨晚checkpoint的GPU加载与持续更新验证（见下方启动记录），40分钟最终结果仍待训练结束。

## 开卡迁移记录

旧机30340已重新开启，找到昨晚Actor805/WM13050完整checkpoint，SHA256为`e1cf244e73f98694b4d0eddf891d7b0e5f15e165788436b7bc17fa8477462086`。新机31428原有备用模型实际Actor419/H32，禁止混用。服务器直连超时，改用本机SSH流式中转；最终源模型置于`/root/gpufree-data/reflex_nov08_source/checkpoint_final.pt`，必须完整SHA匹配后启动。

## 已启动实例

输出：`/root/gpufree-data/reflex_resume_nov08_h16_b3584_utd04_return40_20261109_r1`。监督状态为同名`.job.json`，日志`run.log`、TensorBoard在`tensorboard/`。初始WM1305轮/Actor805，`wm_warmup_completed_before_loop=true`，无500轮重复预热；训练循环208.19秒时Actor828、WM梯度13280，GPU显存20622MiB。此为启动验证，不能作为40分钟完成或效果通过。

## 最终结束与独立视频评测

训练循环2400.260秒正常完成，最终Actor1722（新增917），WM梯度22220（新增9170），真实转移23080960（新增9551872）。没有重复500轮预热。末300行稳态中位WM10=0.6712秒、Actor=0.9779秒、Critic=0.0431秒、向量采集步=0.1530秒；legacy编码未启用新H8 Graph，不能套用stable WM的速度数字。

最终checkpoint独立PhysX单环境seed42评测完成：15秒751控制步，物理终止0次；完成1回合650步（13秒），以time_out结束。发射4次，完成观测投掷3次，CLEAN2/严格保持均0。本短录像显示存活改善，但不能据单种子确认普遍站稳；当前评测继承训练Return Reset起点分布，与昨晚均匀/原生起点不是严格对照。

视频750帧、1280×720、50fps、15秒，全帧解码通过，坐标轴已隐藏。视频SHA256：`e9beeb3b021deba1619210f243a2dc40682e719054bb8fa5a65160bf9a4a583b`。本地视频及原始评测JSON：`/Users/mac/Documents/skills/资料/2026-11-09_195045_Reflex续训40分钟最终评测`。远端视频在训练目录同名加`_play15s/model_video.mp4`。
