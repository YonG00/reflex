# 最小数据备份与恢复

本实验实际只需要两个原始片段，以及生成的三个参考动作和一个 AMP 文件。它们可打包为几 MB 的本地备份，下次恢复即可训练和 play，无需再次下载 G1 大包，也无需重新启动 Isaac Sim 构建动作。

## 数据与许可

当前 GitHub 仓库是公开的。BONES-SEED 许可第 6 节禁止上传数据到公开仓库，允许仅供内部使用且不可被第三方访问的私有仓库。AMASS 原始许可禁止向第三方分发，允许归档备份。因此本仓库仅提交工具和 checksum 清单，不提交这些第三方原始或衍生数据，也不把 AMASS 下载页面的 CC BY 标记当作覆盖原始条款的授权。

- BONES-SEED 原始许可：https://huggingface.co/datasets/bones-studio/seed/blob/main/LICENSE.md
- AMASS 原始许可：https://amass.is.tue.mpg.de/license.html
- 上游数据来源：`assets/motions/README.md`、`THIRD_PARTY_NOTICES.md`。

## 备份内容

`assets/data_bundle_manifest.json` 记录以下六个文件的相对路径、字节数和 SHA-256：

1. BONES 的指定 G1 CSV 原始片段。
2. AMASS B10 的 G1 重定向原始片段。
3. `motions/walk_then_catch_v17_30s.npz`。
4. `motions/male2_walk_v2.npz`。
5. `motions/male2_walk_turn_left_45.npz`。
6. `amp/amp_dataset.pt`。

数据包只作为同一获授权用户的内部备份保存。上游自制的 catch pose 和 ready pose 已随源代码提供，不必再次打包。

## 创建备份

```bash
python tools/motion/local_data_bundle.py pack --data-dir data \
  --output /your/private/storage/reflex_minimal_data_v1.tar.gz \
  --manifest-output assets/data_bundle_manifest.json
```

打包时检查两个原始文件是否与上游 SHA-256 一致，再记录全部六个文件的 checksum。不要把压缩包加入公开 Git 或 Release。

## 恢复到新机器

先克隆代码，把本地备份传到新机器，再执行：

```bash
python tools/motion/local_data_bundle.py restore \
  --bundle /your/private/storage/reflex_minimal_data_v1.tar.gz --data-dir data
```

恢复只用 Python 标准库，不需要 GPU。工具先校验全部六个文件，再写入目标目录；已有相同文件会保留，已有不同内容会报错，不会覆盖。项目运行仍需安装 Isaac Sim 等依赖，checkpoint 继续由官方脚本下载或从自己的存档恢复。
