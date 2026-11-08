# V34 CPU证据

本地macOS，Python3.12.15，Torch2.15.0，无CUDA。CPU验证不证明CUDA/Inductor或加速收益。

- related_initial.log：相关回归109通过、2失败。tuple状态探针已修复；事件源码哈希迁移的失败另行修复复测，不隐去原始失败。
- minimal_fixed.log：修复tuple输入后，V31全部8项与当时V34专项15项，共23通过（107.24秒），含第三方fixture的tuple状态。
- candidate_final.log：最终V34专项21通过（10.98秒），包含严格状态/mask/梯度、H32发射、机器人扰动/关节写入、aot_eager完整图前反向、实际Actor/Critic更新、异常恢复、旧checkpoint及源码固定的显式迁移。不是完整仓库全部测试重跑。

初始回归发现的旧事件迁移全链测试，在修复后独立复测1通过（142.81秒），结果见migration_fixed.log。初始两项失败均有通过的修复复测证据；未声称完整仓库全套重跑。新增文件Ruff通过，shell入口bash -n及git diff --check通过。旧train.py/trainer.py已有其他Ruff告警，本轮不改无关规则和导入顺序。
