# V37新服务器准备记录

新实例：183.147.142.40:30340（不记录认证信息）。代码 `/root/gpufree-data/reflex`，独立分支experiment/v37-gpu-check；基线58f2662301172591706ae7f0ea796c3651d9ce91加本地V37研究补丁，关键源码SHA与本地一致。

GPU RTX4090 24GB，14核配额、50GB内存。Torch2.7.0+cu128、Triton3.3.0、Isaac Sim5.1.0、rsl-rl3.0.1。明确通过PYTHONPATH使用Isaac Lab2.3.1源码，而非预装2.3.2.post1包；NumPy对齐1.26.4。恢复缺失的kit pip archive、URDF importer、机器人资源，未重新下载整个Isaac SDK。

6个动作/AMP数据文件从本地最小数据包恢复并验证。V29 checkpoint位于 `/root/gpufree-data/reflex/checkpoint_final.pt`，它是指向migration_v37内文件的软链接，因此不得仅清理迁移目录。checkpoint SHA256为f693022b0252e031caaf96bfa6ff282a4eda3294eea3ee67d468567e54ea293f。机器人与两份SDK扩展包SHA均匹配本地备份。未迁移不需要的最小capsule及旧5GB编译缓存。

使用：

```bash
source /root/gpufree-data/reflex_v36_env.sh
```

有界120秒环境探针实际17.66秒完成，退出0；Isaac App启动、实际任务模块导入、CUDA分块写入梯度及小型carry的Inductor前向/VJP精确对照通过。探针结束GPU92MiB/0%，未运行正式训练。

注意：只证明环境和小型计算可用，尚未证明新机4096场景、完整H32事件/梯度、稳态速度或训练效果。下一步显式运行scripts/study_event_blocks_v37.sh，以新目录保存四组真实数据对照。环境探针证据与源文件hash见benchmarks_v37_cpu/new_server_*json。

完整代码与资源来自本地已保存备份，不依赖被占用的旧服务器；认证信息未写入仓库、脚本或迁移清单。
