# V2同MAT三臂消融运行说明

## 一、这一步到底要做什么

这一步不是继续盲目调参，而是把当前最重要的两个问题拆开回答：

1. 完整参数平均是不是把少数客户端真正更新过的嵌入“冲淡”了？
2. 换成FedE式本地训练目标以后，在相同训练预算下是否还有额外收益？

因此固定运行三组实验：

| 实验臂 | 聚合方式 | 本地训练 | 大白话作用 |
|---|---|---|---|
| A：`dense_margin` | 完整实体、关系表按三元组数平均 | L2、头尾负采样、间隔排序损失 | 当前动态MAT基线 |
| B：`masked_margin` | 只平均客户端拥有的实体、关系行 | 与A完全相同 | 单独检查行级聚合有没有用 |
| C：`masked_fede_fair` | 与B相同的行级聚合 | L1、尾负采样、自对抗逻辑损失 | 检查FedE式本地目标这一整包有没有用 |

A和B之间只改聚合方式，所以这是最干净、最重要的一组对比。

B和C之间同时改了距离、负采样方向和损失函数。因此，C即使更好，也只能说明“FedE式本地目标组合有效”，暂时不能说究竟是哪一个因素有效。

## 二、程序怎样防止不公平比较

三套配置都写入了同一个公平合同，以下内容必须完全一样：

- FB15k-237的训练集、验证集和测试集；
- 37客户端头实体均衡划分；
- 200行MAT文件和逐轮参与客户端、动态分组；
- 随机种子42和初始模型参数；
- 256维嵌入、5个负样本、批次4096；
- 200通信轮、每轮1个本地epoch；
- Adam、学习率0.001；
- 每10轮一次1000条验证事实评估；
- 最终完整验证集、测试集的全局头尾filtered评估。

程序会做三层检查：

1. 训练前检查三份YAML共享字段和原始数据文件哈希；
2. 每次训练构造完客户端和MAT调度后，核对固定的划分哈希与调度哈希；
3. 汇总三组结果时，再核对划分、MAT、初始模型、参与预算和训练预算。

只要其中一项不同，汇总程序就会拒绝比较，而不是勉强给出一个容易误导的表格。

## 三、先做不训练的安全检查

在项目根目录`D:\1\1myworkcode`执行：

```powershell
python -m HFLSnF_KG_v2.run_three_arm_ablation --action validate
```

这条命令只读配置、数据和MAT文件，不创建模型训练进程，也不需要GPU。看到“**三臂配置校验通过**”才说明三份配置满足公平合同。

## 四、在CUDA服务器上一键运行

### 4.1 先跑最重要的B臂

如果服务器时间有限，建议先运行B，因为A已经是当前基线，B能最直接地检验行级聚合：

```bash
python -m HFLSnF_KG_v2.run_three_arm_ablation \
  --action run \
  --arm masked_margin
```

### 4.2 一次连续运行A、B、C

```bash
python -m HFLSnF_KG_v2.run_three_arm_ablation \
  --action run \
  --arm all
```

程序会按A、B、C顺序启动三个独立训练进程。每个通信epoch仍会实时打印：

- 当前实验方案；
- epoch和MAT行号；
- 参与客户端数和动态组数；
- 本地加权损失；
- 本轮耗时；
- 到评估轮时的验证MRR与Hits@3。

如果CUDA不可用，程序会在读取FB15k-237和启动训练进程之前直接报错，不会偷偷切回CPU。

## 五、在PyCharm或VSCode中一键运行

直接打开：

`HFLSnF_KG_v2/run_three_arm_ablation_from_ide.py`

本机默认配置是：

```python
DEFAULT_ACTION = "validate"
DEFAULT_ARM = "all"
```

此时点击“运行Python文件”只做安全检查。把代码复制到CUDA服务器后，将第一行改成：

```python
DEFAULT_ACTION = "run"
DEFAULT_ARM = "all"
```

再次点击运行，就会按A、B、C顺序连续训练并自动汇总。IDE入口和终端入口调用的是同一套代码，不存在两套实验逻辑。

如果只想从IDE跑B，把`DEFAULT_ARM`改成`"masked_margin"`即可。

## 六、三次分开跑完后怎样汇总

如果A、B、C不是一次连续跑完，可以把三个结果目录交给汇总入口：

```powershell
python -m HFLSnF_KG_v2.run_three_arm_ablation `
  --action summarize `
  --result "dense_margin=D:\结果目录\A" `
  --result "masked_margin=D:\结果目录\B" `
  --result "masked_fede_fair=D:\结果目录\C"
```

汇总前程序会重新审计三个结果是否可比。审计通过后生成：

- `comparison_summary.json`：机器可读的完整比较；
- `comparison_metrics.csv`：A、B、C最终指标和耗时表；
- `comparison_report.md`：简体中文大白话结论。

## 七、结果放在哪里

单个训练结果仍写入：

```text
HFLSnF_KG_v2/results/<对应方案名称_时间戳>/
```

一键任务的合同、日志、执行状态和汇总报告写入：

```text
HFLSnF_KG_v2/results/three_arm_ablation_<时间戳>/
```

即使中途失败，`suite_status.json`也会保留已经完成的实验臂和错误信息，不需要猜训练停在了哪里。

## 八、结果应该怎样判断

先看A到B：

- 如果B的测试MRR比A至少高0.003，而且Hits@3不下降，说明行级聚合值得继续；
- 如果差值很小或Hits@3下降，就不能宣称行级聚合有效。

再看B到C：

- 如果C同样达到MRR至少提高0.003且Hits@3不下降，下一步补充D：`dense+FedE-fair`，构成完整二乘二；
- 如果C没有通过初筛，就先不补D，也不运行高开销FedE论文参数。

最后要注意：种子42的一次结果只是初筛。真正写论文结论前，胜出方案至少需要三个随机种子，并报告均值、标准差和置信区间。

## 九、三份正式配置

- A：`configs/server_fb15k237_hflsnf_dynamic_mat_cuda.yaml`
- B：`configs/server_fb15k237_hflsnf_dynamic_mat_masked_cuda.yaml`
- C：`configs/server_fb15k237_hflsnf_dynamic_mat_masked_fede_fair_cuda.yaml`

不建议直接手工复制并修改这三份配置。确实需要改预算时，应当三份一起改，并先重新执行`--action validate`。
