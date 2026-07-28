# 四臂二乘二与无需重训方向诊断

## 一、这一步完成了什么

三臂结果已经说明：

- B比A好，行级聚合有帮助；
- C比B好，FedE式本地目标组合有帮助。

但只有A、B、C时，还无法判断这两项改进是简单相加，还是必须组合在一起才有效。因此现在补上D：`dense+FedE-fair`，形成完整二乘二：

|  | margin目标 | FedE公平预算目标 |
|---|---|---|
| 稠密聚合 | A：`dense_margin` | D：`dense_fede_fair` |
| 行级聚合 | B：`masked_margin` | C：`masked_fede_fair` |

代码还增加了一个只读诊断入口。它不重新训练模型，只读取最佳检查点，分别统计头预测、尾预测、逐关系指标和逐查询胜负。

## 二、D臂保持了哪些内容不变

D使用配置`configs/server_fb15k237_hflsnf_dynamic_mat_dense_fede_fair_cuda.yaml`。

它与C保持相同的：

- 37个客户端及头实体均衡划分；
- 200行MAT动态参与和动态分组；
- 256维嵌入；
- L1距离；
- 每个正样本5个尾负样本；
- FedE式自对抗逻辑损失；
- 批次4096、学习率0.001；
- 每轮1个本地epoch、共200轮；
- 全局候选、全局filtered、头尾双向评估。

D相对C只把聚合从`row_mask_presence`改回`dense_triple_weighted`。配置沿用历史`ablation_suite: v2_same_mat_three_arm_v1`，这是为了让已经完成的A、B、C结果可以直接复用，不代表D仍属于三臂。

## 三、如何运行D

### 3.1 先在任何电脑上校验

从项目根目录运行：

```powershell
& 'D:\Anaconda3\envs\py37\python.exe' -m HFLSnF_KG_v2.run_four_arm_ablation --action validate
```

这一步只校验四份配置、数据文件、MAT文件和公平合同，不启动训练。

### 3.2 在CUDA服务器只补跑D

如果已有A、B、C仍放在`HFLSnF_KG_v2/results/三种/`：

```bash
python -m HFLSnF_KG_v2.run_four_arm_ablation \
  --action run-d \
  --existing-root HFLSnF_KG_v2/results/三种
```

程序只训练D，不重复训练A、B、C。D完成后会自动读取旧三臂结果，检查以下指纹是否一致：

- 客户端划分哈希；
- MAT调度哈希；
- 初始模型哈希；
- 通信轮、本地epoch和有效数据遍历预算。

校验通过后，管理目录会生成：

- `factorial_contract.json`：四臂公平合同；
- `factorial_summary.json`：四臂指标、条件效应和交互项；
- `factorial_metrics.csv`：四臂表格；
- `factorial_report.md`：大白话报告；
- `dense_fede_fair.log`：D臂完整终端日志；
- `suite_status.json`：任务状态和四个实际结果目录。

### 3.3 从IDE一键运行

打开`run_four_arm_ablation_from_ide.py`。

本机默认：

```python
DEFAULT_ACTION = "validate"
```

放到CUDA服务器后改为：

```python
DEFAULT_ACTION = "run-d"
```

然后直接点击PyCharm或VS Code的“运行Python文件”。

## 四、D出来后看哪四个差值

| 差值 | 回答的问题 |
|---|---|
| B-A | margin目标下，行级聚合是否有效 |
| D-A | 稠密聚合下，FedE目标是否有效 |
| C-B | 行级聚合下，FedE目标是否有效 |
| C-D | FedE目标下，行级聚合是否有效 |

交互项使用：

```text
(C-B) - (D-A)
```

它也等价于：

```text
(C-D) - (B-A)
```

大白话解释：

- 接近0：两项改进大致各做各的，效果近似相加；
- 明显大于0：行级聚合和FedE目标放在一起有额外收益；
- 明显小于0：两项改进解决了部分相同问题，组合收益发生重叠或抵消。

当前程序把绝对值`0.003`作为描述性参考线，但这不是显著性检验。只有种子42时，不能把交互写成稳定统计结论。

## 五、无需重训方向诊断怎么运行

### 5.1 本机CPU冒烟

默认会从`results/三种`自动发现A、B、C，只评估8条测试事实：

```powershell
& 'D:\Anaconda3\envs\py37\python.exe' -m HFLSnF_KG_v2.run_directional_diagnostics
```

少量事实只用于验证代码链路，不能判断模型优劣。

### 5.2 CUDA服务器完整评估

D训练结束后，把D结果目录显式加入：

```bash
python -m HFLSnF_KG_v2.run_directional_diagnostics \
  --result-root HFLSnF_KG_v2/results/三种 \
  --result dense_fede_fair=/path/to/dense_fede_fair_result \
  --using-gpu \
  --require-cuda \
  --max-triples 0 \
  --query-batch-size 16 \
  --candidate-batch-size 4096
```

`--max-triples 0`表示完整官方测试集。正式模式没有CUDA会立即报错，不会偷偷退回CPU。

### 5.3 从IDE一键运行

打开`run_directional_diagnostics_from_ide.py`。

本机保持：

```python
FULL_CUDA = False
```

服务器正式运行时：

```python
FULL_CUDA = True
D_RESULT_DIR = "results/你的D臂结果目录"
```

然后直接点击运行。

## 六、方向诊断输出怎么读

每次诊断输出：

- `directional_summary.json`：头、尾和综合指标及检查点指纹；
- `query_ranks.csv`：每条事实的头、尾排名；
- `relation_metrics.csv`：每个关系的头、尾和综合指标；
- `pairwise_query_outcomes.csv`：A到B、A到D、B到C、D到C的逐查询胜负；
- `pairwise_summary.csv`：逐查询胜负汇总；
- `directional_report.md`：大白话报告；
- `diagnostic_config.json`：本次实际参数和结果目录。

重点判断：

1. C相对B的提升是否只出现在尾预测；
2. B的Hits@1下降主要来自头预测还是尾预测；
3. 哪些关系反复受益，哪些关系反而退化；
4. 平均MRR提高是大量查询小幅改善，还是少数查询大幅改善。

如果C只明显改善尾预测，说明尾负采样方向可能是主要来源之一。如果头、尾都提高，则FedE式目标学到的共享实体和关系表示也在帮助头预测。

## 七、本机已经做过的验证

当前已经完成：

- 四臂配置公平合同校验；
- 四臂条件效应和交互项手工数值测试；
- 无CUDA时D臂正式训练快速失败测试；
- 固定种子测试事实抽样复现测试；
- 逐关系头、尾、综合指标测试；
- 逐查询胜、负、平统计测试；
- 无CUDA时正式方向诊断快速失败测试；
- 使用真实A、B、C最佳检查点完成2条FB15k-237测试事实的CPU全链路冒烟。

真实2条事实的输出仅证明检查点读取、排名和文件落盘能够完成，不用于报告模型性能。

## 八、真正的下一步

现在服务器上只需要做两件事：

1. 只补跑D臂；
2. D结束后对A、B、C、D执行完整方向诊断。

拿到D和完整方向诊断结果后，再决定优先补多随机种子，还是进一步拆分L1、尾负采样和自对抗损失。现阶段不要先改学习率，也不要先运行256负样本、3本地epoch的高开销论文配置。
