# FB15k-237数据放置说明

请将FB15k-237的三个标准文本文件放到本目录下的 `FB15k-237/`：

```text
HFLSnF_KG/
└── data/
    └── FB15k-237/
        ├── train.txt
        ├── valid.txt
        └── test.txt
```

每一行必须包含三个由制表符或空格分隔的字段：

```text
头实体    关系    尾实体
```

加载器会按照训练集、验证集、测试集中的首次出现顺序建立统一的 `entity2id` 和 `relation2id`。三个数据划分中的已知真三元组会共同用于filtered排名过滤，但只有 `train.txt` 会参与模型参数更新。
