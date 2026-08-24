# V4十二格冻结参考

本目录保存V5报告所需的三份历史参考JSON。它们是V4已完成十二格重叠实验的按字节冻结副本，用于生成15单元背景视图；V5运行时不再访问HFLSnF_KG_v4目录。

| 文件 | SHA-256 |
| --- | --- |
| `overlap_ablation_summary.json` | `ca3032bae7cadb67cbc10d4981394a21dac2b597b8a85a6f5602108e32e3d6a4` |
| `official12_summary.json` | `d419863ac8953087b072486ff37527932ca20ae827546add7cd040cec3b80028` |
| `batch_summary.json` | `4029e345ea9889edb9c34a6493356662147dbf959094a43694ce189cd138bad3` |

不得重新格式化、修改换行符或改写历史路径字段。任何字节变化都会触发配置合同失败。正式训练仍只新增三组V5图语义实验，不会重新运行V4实验。