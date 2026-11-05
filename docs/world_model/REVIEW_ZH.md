# 首版审查与修正

比较基点 `8f9a139`，首版实现 `e274515`。按 implement → code-review 流程分别执行 Standards 与 Spec 轴只读审查。以下是修正前发现，不能当作仍未处理的问题。

## Standards

无工作区目录违规。发现 3 项：

1. P1：reward head 未接收 raw action history。腕锁会使动作映射不可逆，原 `action_rate_l2` 不可表示。
2. P2：排序探针的真实仿真交互没有计入总预算。
3. P2：ring 填满后，采样每个字段先拼接整个 replay，造成不必要复制。

修正：补齐奖励模型输入并加反例测试；计数探针所有真实步并存档；直接将时间索引映射为 ring 物理索引后 gather minibatch。

## Spec

发现 3 项：

1. P1：缺少 raw action history，同时 canonical state 去掉了固定参考所需的 anchor 相对位置信息。
2. P1：探针漏计使“相同真实交互量”比较失真。
3. P2：只有总 wall time，缺少模块耗时。

修正：模型额外接收重建 Oracle observation 与当前 raw action；总预算包含 collect + probe；增加同步边界的分项计时。全学习奖励头、几何阶段标签、PhysX warm-start 限制均在实施说明披露，仍需 GPU 实测。

两轴合计各 3 项，最严重问题分别都是奖励条件缺失；共同的阻断问题已修正并进入 CPU 回归。CPU 通过不等于真实仿真或效果通过。
