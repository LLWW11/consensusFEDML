# 无需重训评估桥接说明（大白话版）

## 这一步到底在做什么

一句话说明：

> 不重新训练模型，只把已经训练好的几个模型拿到同一套“考卷”和同一套“判分规则”下重新考试。

FedE日志里的MRR约为0.4299，当前项目里的集中式MRR约为0.208，联邦MRR约为0.155。表面看FedE高很多，但这几个数字不是同一种考试：

- FedE只预测尾实体；
- FedE会先屏蔽当前客户端训练中没有出现过的实体；
- 当前项目要在全部14,541个实体中同时预测头和尾；
- FedE与当前项目重新划分了训练、验证和测试集。

因此，直接把0.4299和0.155放在一起比较，就像一个人做了只含本地知识点的选择题，另一个人做了全知识点的双向问答题。分数高低不能直接说明训练算法谁更好。

评估桥接就是把这些不同一点一点拆开，最后让所有检查点做同一批题。

## 它不会做什么

这个入口不会：

- 启动任何训练；
- 创建优化器；
- 修改模型参数；
- 覆盖FedE检查点；
- 覆盖当前项目检查点；
- 把结果写回旧的`HFLSnF_KG/results/`。

所有新结果只会写到：

```text
HFLSnF_KG_v2/results/evaluation_bridge_*/
```

结果中的`training_performed`固定为`false`。

## 四种评估分别是什么意思

### E0：照着FedE原来的规则再考一次

E0尽量原样复现FedE：

- 每个客户端单独评估；
- 只预测尾实体；
- 候选实体主要是这个客户端训练时见过的实体；
- 使用客户端自己的训练、验证和测试真三元组做filtered过滤；
- 使用原代码的双重`argsort`排名方式；
- 三个客户端按测试三元组数量合并。

完整正式评估应该复现：

```text
MRR约为0.4299
Hits@1约为0.3031
Hits@5约为0.5828
Hits@10约为0.6711
```

E0的作用不是得到新结论，而是确认：

> 新评估代码确实读对了FedE检查点、实体表、三个客户端的私有关系表和原始评估规则。

### E1a：取消本地候选屏蔽，只预测尾实体

E1a仍然使用FedE检查点和FedE测试集，但改成：

- 在全部14,541个实体中找尾实体；
- 使用全局310,116条已知真三元组做filtered过滤。

E0与E1a之间的差值，主要说明“缩小候选实体范围”给FedE分数带来了多大帮助。

### E1b：使用全局候选，只预测头实体

E1b回答：

> 同一个FedE模型在预测头实体时表现怎么样？

原FedE日志没有评估头预测，所以这个数字是补充信息。

### E1c：全局候选，头尾一起算

E1c把E1a的尾排名和E1b的头排名放在一起做微平均。

这已经接近当前项目的评估方式，但它仍使用FedE自己的31,013条测试事实，所以还不能直接和当前项目最终测试MRR比较。

### E2：所有模型做同一套公共测试题

E2是真正的横向校准：

- FedE检查点；
- 集中式快速检查点；
- FLSnF固定25客户端检查点；
- HFLnoSnF固定15客户端检查点；
- 以后生成的V2行级聚合检查点。

它们全部使用：

- 同一批严格公共测试三元组；
- 全部14,541个候选实体；
- 相同的全局真三元组filtered集合；
- 相同的头预测和尾预测；
- 相同的乐观并列排名；
- 相同的MRR、Hits@1、Hits@3、Hits@5和Hits@10公式。

## 为什么公共测试集只有2,048条

当前官方划分和FedE划分虽然包含同样的310,116条三元组，但各自把它们分到了不同位置。

如果直接拿当前官方测试集去评估FedE：

- 20,466条官方测试事实中有16,327条已经进入FedE训练集；
- 泄漏比例约为79.78%。

如果反过来拿FedE测试集评估当前模型：

- 31,013条FedE测试事实中有27,198条已经进入当前官方训练集；
- 泄漏比例约为87.70%。

这种交叉测试相当于把训练时见过的题放进考试，得分会虚高。

当前代码只保留“双方都放在测试集，而且双方训练集和验证集都没有见过”的事实，最终得到：

```text
严格公共验证集：1,743条
严格公共测试集：2,048条
公共测试查询：4,096个，即2,048个头预测加2,048个尾预测
```

公共测试集哈希固定为：

```text
72359a1db962f6a2ec724c5237a58654123d7e8541eb7aecb542a5247f07e422
```

需要注意：2,048条只占官方测试集约10%。它适合回答“同一批题上谁更好”，但不能替代完整官方测试集，也不能单独代表模型在全部关系上的总体精度。

## 两边的实体关系编号为什么能够接上

FedE pickle里没有实体名称字典，只有整数编号。代码做了下面这层保险：

1. 当前项目按`train.txt → valid.txt → test.txt`重新生成实体和关系编号；
2. FedE使用`edge_index_ori`和`edge_type_ori`恢复全局编号；
3. 对比双方310,116条三元组全集；
4. 只有每一条都完全相同时才允许继续评估。

当前核验结果为完全一致，所以FedE实体表第`i`行可以和当前项目实体编号`i`对接。

三个FedE客户端的关系表分别是：

```text
客户端0：78行
客户端1：78行
客户端2：81行
```

它们拥有互不重复的关系。代码根据pickle中的局部关系到全局关系映射，把三张表拼回237行完整关系表，不会做平均。

如果以后误换成Fed5、Fed10或另一份数据，三元组全集、文件哈希或关系映射不一致时会直接报错，不会勉强继续。

## 本机CPU轻量验证

从项目根目录运行：

```powershell
& 'D:\Anaconda3\envs\py37\python.exe' `
  -m HFLSnF_KG_v2.run_evaluation_bridge `
  --cf HFLSnF_KG_v2/configs/evaluation_bridge_smoke_cpu.yaml
```

这份配置只抽取：

- 8条FedE测试事实；
- 4条严格公共测试事实；
- 50次bootstrap。

它只检查以下内容：

- 数据和检查点能否读取；
- 三张关系表能否正确拼接；
- E0、E1和E2是否都能完成；
- CPU指标是否为有限值；
- 所有输出文件是否齐全。

冒烟输出中的MRR来自极少量样本，波动会很大，不能写进论文，也不能拿来判断模型优劣。

## CUDA服务器完整评估

服务器正式命令：

```bash
python -m HFLSnF_KG_v2.run_evaluation_bridge \
  --cf HFLSnF_KG_v2/configs/evaluation_bridge_full_cuda.yaml
```

完整配置会运行：

- E0：FedE完整31,013条测试事实，局部候选尾预测；
- E1：同样31,013条事实的全局头预测和尾预测；
- E2：完整2,048条严格公共测试事实；
- FedE、集中式、FLSnF和HFLnoSnF四个检查点；
- 1,000次bootstrap和相对集中式快速基线的配对差值区间。

正式配置包含：

```yaml
using_gpu: true
require_cuda: true
gpu_id: 0
```

如果服务器没有CUDA，程序会在读取FB15k-237和检查点前直接报错，不会静默退回CPU。

这不是训练，但完整排名需要让很多查询依次和14,541个候选实体比较，仍然会花费一定时间。它通常比重新训练便宜得多，但不会像读取JSON一样瞬间完成。

## 在PyCharm或VS Code里直接运行

最简单的方式是打开：

```text
HFLSnF_KG_v2/run_evaluation_bridge_from_ide.py
```

直接点击“运行Python文件”。

文件顶部默认为：

```python
DEFAULT_BRIDGE_PROFILE = "evaluation_bridge_smoke_cpu"
```

所以第一次点击只运行安全的CPU冒烟。

放到CUDA服务器后，把它改成：

```python
DEFAULT_BRIDGE_PROFILE = "evaluation_bridge_full_cuda"
```

再次点击运行即可。

也可以继续使用统一IDE入口，通过环境变量选择：

```powershell
$env:HFLSNF_KG_V2_IDE_PROFILE = "evaluation_bridge_smoke_cpu"
python HFLSnF_KG_v2\run_from_ide.py
```

正式CUDA方案把值改为：

```text
evaluation_bridge_full_cuda
```

## 只做数据审计或单独跑某个阶段

终端参数`--stage`可以临时覆盖YAML：

```powershell
python -m HFLSnF_KG_v2.run_evaluation_bridge `
  --cf HFLSnF_KG_v2/configs/evaluation_bridge_smoke_cpu.yaml `
  --stage data
```

可选值：

| 值 | 实际工作 |
|---|---|
| `data` | 只核对数据、哈希、泄漏和公共集，不做模型排名 |
| `fede` | 运行E0和E1 |
| `common` | 只运行E2公共测试 |
| `all` | 依次运行数据审计、E0、E1和E2 |

## 如何加入新的V2检查点

在YAML的`project_checkpoints`中增加：

```yaml
project_checkpoints:
  - name: "dynamic_mat_masked"
    path: "results/你的V2结果目录/model_best.pt"
```

加载器会自动检查：

- 文件是不是标准TransE检查点；
- 是否包含完整实体和关系表；
- 实体数是不是14,541；
- 关系数是不是237；
- 实体和关系映射是否与公共数据完全一致；
- 同目录`config_snapshot.json`是否明确记录L1或L2距离；
- 配置维数是否和检查点张量形状一致。

旧检查点本身没有保存`distance_norm`，所以同目录配置缺失时程序会拒绝猜测。如果确实知道检查点使用的距离，可以在YAML对应项显式增加：

```yaml
distance_norm: 2
```

## 每个结果文件是干什么的

每次运行会创建独立时间戳目录，其中包括：

| 文件 | 大白话说明 |
|---|---|
| `config_snapshot.json` | 这次到底用了什么配置 |
| `data_audit.json` | 数据是否同号、有没有泄漏、公共集覆盖多少 |
| `common_valid.tsv` | 1,743条公共验证事实及名称和客户端归属 |
| `common_test.tsv` | 2,048条公共测试事实及名称和客户端归属 |
| `model_manifest.json` | 每个检查点的路径、SHA-256、维数和L1/L2 |
| `protocol_metrics.json` | E0、E1、E2的核心指标 |
| `query_ranks.csv` | 每条事实的头排名、尾排名和倒数排名 |
| `summary.json` | 一次运行的总汇总 |

`query_ranks.csv`保留逐查询结果，是为了后续能够进行配对bootstrap，而不是只比较两个四舍五入后的MRR。

## 正式结果应该先看哪里

建议按这个顺序看：

1. 打开`summary.json`，确认`status`为`completed`且`training_performed`为`false`；
2. 打开`protocol_metrics.json`，确认E0完整MRR接近0.4299；
3. 比较E0与E1a，观察局部候选屏蔽带来的影响；
4. 比较E1a与E1b，观察尾预测和头预测的难度差异；
5. 查看E2中各模型的`combined_metrics.mrr`和`hits_at_3`；
6. 查看`paired_delta_vs_reference`，判断相对集中式快速基线的MRR差值区间；
7. 同时查看`data_audit.json`中的公共测试覆盖率，不把公共子集结果冒充完整测试结果。

## 怎样判断下一步

评估桥接完成后：

- 如果FedE从E0切换到E1后MRR明显下降，说明原高MRR中很大一部分来自较小候选范围和仅尾预测；
- 如果FedE在E2公共测试上仍明显优于其他检查点，才说明FedE训练目标或行级聚合值得继续深入；
- 如果FedE与当前检查点在E2差异很小，就不应先付出高成本复刻论文全部训练预算；
- 下一组训练实验仍应按`dense+margin → masked+margin → masked+FedE-fair`顺序运行。

初筛建议继续使用：

```text
MRR绝对提升至少0.003，并且Hits@3不下降
```

通过后再运行3个训练随机种子和置信区间。

## 安全说明

FedE数据和检查点都使用Python pickle格式。pickle能够执行序列化对象中的代码，因此只能加载可信的本地文件。

正式配置在反序列化前先检查：

- Fed3 pickle SHA-256；
- FedE最佳检查点SHA-256。

文件身份变化时会立即停止。
