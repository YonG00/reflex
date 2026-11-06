# 指定世界模型checkpoint生成视频

适用于本仓库WM联合训练checkpoint（含config/actor/model），自动读取H16/H32、奖励配置和
训练时BOX/AMP/M5环境标志。不是原Reflex PPO的model_*.pt入口，不会绕过物理/任务契约检查。
仅加载模型执行真实play，不训练、不改原checkpoint。

在GPU服务器：

```bash
conda activate isaaclab
cd /root/gpufree-data/reflex
bash scripts/local/eval_wm_video.sh /完整路径/checkpoint.pt
```

默认单环境、seed42、约15秒、隐藏坐标轴/参考及接触调试绘图。
例如当前最新可用的站稳H16存档：

```bash
bash scripts/local/eval_wm_video.sh \
  outputs/world_model/20261106_v25_standing_h16_joint_30min/checkpoint_00500.pt
```

调整时长、随机种子和输出目录：

```bash
bash scripts/local/eval_wm_video.sh /完整路径/checkpoint.pt \
  --seconds 15 --seed 8 --output outputs/world_model/videos/my_model_seed8
```

输出目录必须为空。默认会自动创建含上海时间和模型文件名的独立目录：
`outputs/world_model/videos/YYYYMMDD_HHMMSS_微秒_模型名/`。
结束会打印视频的完整路径：
- model_video.mp4：可直接看的视频；完整FFmpeg解码验证后才显示完成。
- play_summary.json：真实环境评测指标。
- video_validation.json：解码帧数/视频时长。
- run.log：完整日志。
- checkpoint_config.json、checkpoint_flags.env、evaluation_request.json：配置和选定模型记录。
- command.sh：对应底层评测命令。单独运行它的视频在videos/，不包含入口后的验证/复制步骤。

`--num-envs 8`可并行测试8环境，镜头仍跟随env0。默认单环境便于观看，不把一段视频当作成功率。
`--dry-run`只导出配置与底层命令，不启动Isaac；查看帮助用`--help`。
渲染和Isaac初始化耗时通常长于视频时长，等终端显示“评测完成”再打开视频。

## 2026-11-07 验证

3项CPU测试通过：精确选择带空格路径、时长/隐藏轴参数、checkpoint配置导出、环境标志安全引用。
GPU实际执行最新checkpoint_00500.pt，自动配置、单环境、无坐标轴，生成1秒测试录像并完整解码通过，
输出outputs/world_model/20261107_eval_video_cli_smoke/model_video.mp4。
训练/其他诊断仍至少6环境，只有play允许1环境；未更改训练算法或模型权重。
