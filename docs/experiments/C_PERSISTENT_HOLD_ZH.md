# C版持续抱持：取消按接箱时间恢复参考姿态

## 问题与范围

旧C从零600的strict达到108/640，但最终持续接住101/640，低于增维600的116/640。
用户观察到接住后的姿态恢复会挤出箱子。代码确认存在：

- 原retain过渡窗口2秒，settled后1秒恢复姿态门控；
- 躯干anchor及全身motion tracking始终追参考动作；
- two-stage手臂BASE仍在准备/抬臂参考间切换，只有有限条件下的cradled锁存。

这证明存在竞争入口，不等于已经证明每次掉箱都由同一原因造成。
本分支`ppo/c-persistent-hold`以旧C为基线，开关`BOX_PERSISTENT_HOLD=1`，
只允许与C（BOX_BUFFER_STAGE=3、BOX_FLEX_ARMS=1、A/B关闭）组合。
开关关闭时保留原S3/A/B/C行为，buffer_control_core.py不改。

## 持续抱持状态

只在真实发射、非HOLD_INIT、安全姿态、未落地、未逃逸时启用。
入口需要真实手/腕/肘/前臂接触支持，箱子躯干相对距离<.75m、速度<1m/s、
相对角速度<2rad/s，连续有效.12秒。胸部单独碰撞/箱子靠近不能进入。

进入后，只要上述健康状态仍成立、距离<.9m、相对速度<2m/s、角速度<6rad/s，
且最近臂支持接触不超过.3秒，就持续保持；**没有2秒或其他接箱年龄上限**。
短暂脱手允许.3秒重新支持，不能在落地后重新刷持有奖励；落地/严重逃逸沿用原状态锁存。

- 持续retain仍用原权重.5。raw=`(.5+.5*proximity)*exp(-(rel_v/.5)^2-(rel_w/2)^2)*entry_ramp`。
- 低线/角相对速度获得更高奖励，不指定固定接箱姿态；PPO仍按原流程对reward乘dt。
- 有效持有期间旧restore门控保持关闭，不因settle_age或catch_age变大而恢复。
- 真正失去支持/控制后才退出；软退出插值.25秒只为减小目标突变，不是持有后的自动计时退出。
- 落地、跌倒、严重逃逸或reset立即清除新奖励和姿态放宽。

原Buffer日志继续统计原2秒过渡，新增Persistent指标统计长时抱持；二者不混用。

## 参考跟踪与动作

有效持有期间，motion_global_anchor_pos/ori的参考误差逐步不再计分，以原上限1作固定中性得分。
不能将正奖励直接清零，否则会让策略因接住而失去收益，诱发避免抱持。原权重不改。
四个motion_body pos/ori/lin_vel/ang_vel仅评价髋/膝/踝/脚，
不再直接追上半身和骨盆的参考动作。未持有时原函数/权重/参数保持。

手臂two-stage原回调完整执行，随后仅替换其BASE：
进入持有时，以当前实际肩肘位置减去当前策略残差作为该投掷的动作基准。
持续持有时，原准备/抬臂BASE的变化不再带动这组关节；**策略残差仍可连续调整双臂**，
不锁定关节，不强制唯一CRADLE_POSE。真实失控后才交回原BASE。
C肩部.02→.20 ramp、肘部.35和物理关节限位继续保留，腕部gimbal不改。

保留原PPO/AMP及其判别器、Actor183/Critic305、参考运动输入、腿部跟踪、
关节/动作安全、跌倒代理、anchor/ankle终止及所有初始化/hover/赋速/发射/重投/HOLD_INIT事件。
AMP仍可能提供间接动作先验；腿部跟踪也可能间接限制躯干运动。
本版取消的是已确认的定时恢复及直接上半身参考牵引，不能提前声称所有掉落原因都已消除。

## 验证与日志

本地137项CPU测试通过，包含持续控制12秒仍有奖励且不恢复参考、同一步缓存、
短暂脱手、失控/地面/跌倒、HOLD_INIT/reset/rethrow、上半身跟踪梯度取消但腿部保留、
捕获实际手臂基准后不追准备BASE、策略残差仍可改变目标。
既有原S3固定提交逐Tensor对照仍通过。

新增日志：persistent_hold_fraction/mix、reference_scale、hold_raw、最长持有时间、
hold_entries/holding_throws/hold_seconds/lost_after_hold、arm_base_weight，以及每投结束持有秒数。
原invalid_positive_intake/retain审计保留。

上卡计划：先128环境×12更新从零回归RewardManager/原事件链；
再用旧C600权重做只推理的有界诊断，确认真实支持后持续持有和参考门控实际触发。
旧权重只用于诊断，不作正式训练初始化。验证后正式4096环境、seed42从零600更新。

```bash
bash scripts/train_C_persistent_scratch600.sh
```

正式入口拒绝resume/warm-start。没有修改其他训练超参数，也没有新增负冲击惩罚。
GPU验证与正式训练启动状态将在核对后补充。

## 2026-11-11 GPU验证与正式启动

- 服务器及本地137项测试通过。服务器最初继承持有模式环境变量运行对照用例，被配置守卫拒绝；
  清理外部实验开关后137项全部通过，保留原失败日志便于追溯，不修改对照测试来掩盖配置问题。
- 初始实现129f214完成128环境×12更新从零GPU回归（iteration0..11），新持有入口未触发；
  invalid_positive_intake/retain均0，证明原RewardManager/事件链和PPO更新可运行。
- 最终95c5749修正了正anchor奖励不能直接清零的问题，并通过同样137项CPU测试。
- 使用旧C600权重在最终实现上做128环境×640控制步只推理诊断。
  旧权重**仅用于诊断，不作为正式训练初始化**。
- 真实连续控制最长10.08004秒；mix达到1，动作基准替换实际触发；
  invalid_positive_intake=0、invalid_positive_retain=0。它是机制验证，不当作新策略训练效果。
- GPU诊断目录：/root/gpufree-data/C_persistent_regress_20261111/。

正式训练已启动：原单分支Actor183/Critic305、4096环境、seed42、从零600更新。
训练源码95c5749；队列PID103579。
输出：/root/gpufree-data/C_persistent_scratch600_20261111/，train.log/train.jsonl/status.txt。
驱动脚本在启动前检查诊断实际持续控制超过2秒及审计为0；结束后验证完整0..599迭代、
无checkpoint载入、最终输入维度及所有参数有限，并写checkpoint.txt/check.json。
SSH断开不影响nohup任务。当前是启动记录，不能视作600更新已完成或效果已提高。
