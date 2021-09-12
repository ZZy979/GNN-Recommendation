# 异构图表示学习
## 数据集
[ogbn-mag](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag) - OGB提供的微软学术数据集

| 顶点类型 | 数量 |
| --- | --- |
| author | 1134649 |
| paper | 736389 |
| field_of_study | 59965 |
| institution | 8740 |

| 边类型 | 数量 |
| --- | --- |
| (author, writes, paper) | 7145660 |
| (paper, cites, paper) | 5416271 |
| (paper, has_topic, field_of_study) | 7505078 |
| (author, affiliated_with, institution) | 1043998 |

* 预测任务：顶点分类，预测论文所属期刊
* 类别数：349
* 评价指标：分类准确率

## Baselines
### R-GCN (full batch)
`python -m gnnrec.hge.rgcn.run_ogbn_mag_full`

### HAN
`python -m gnnrec.hge.han.run_ogbn_mag`

### HetGNN
1. 预处理 `python -m gnnrec.hge.hetgnn.preprocess data/word2vec/ogbn_mag.model data/word2vec/ogbn_mag_corpus.txt data/hetgnn`
2. 训练模型（有监督） `python -m gnnrec.hge.hetgnn.run_ogbn_mag data/hetgnn`

### 预训练顶点嵌入
使用metapath2vec（随机游走+word2vec）预训练顶点嵌入，作为GNN模型的顶点输入特征
1. 随机游走 `python -m gnnrec.hge.metapath2vec.random_walk data/word2vec/ogbn_mag_corpus.txt`
2. 训练词向量 `python -m gnnrec.hge.metapath2vec.train_word2vec --size=128 --workers=8 data/word2vec/ogbn_mag_corpus.txt data/word2vec/ogbn_mag.model`

### HGT
#### 邻居平均(average)
`python -m gnnrec.hge.hgt.run_ogbn_mag`

#### 预训练顶点嵌入(pretrained)
`python -m gnnrec.hge.hgt.run_ogbn_mag --node-feat=pretrained --node-embed-path=data/word2vec/ogbn_mag.model --epochs=40`

### HGConv
#### 邻居平均(average)
`python -m gnnrec.hge.hgconv.run_ogbn_mag`

#### 预训练顶点嵌入(pretrained)
`python -m gnnrec.hge.hgconv.run_ogbn_mag --node-feat=pretrained --node-embed-path=data/word2vec/ogbn_mag.model`

### R-HGNN
`python -m gnnrec.hge.rhgnn.run_ogbn_mag data/word2vec/ogbn_mag.model`

### C&S
#### Linear+Smooth+正样本图
`python -m gnnrec.hge.cs.run_ogbn_mag /home/zzy/output/pos_graph_5.bin`

#### R-HGNN+Smooth+正样本图
1. 预训练R-HGNN `python -m gnnrec.hge.rhgnn.run_ogbn_mag --save-path=/home/zzy/output/rhgnn.pt data/word2vec/ogbn_mag.model`
2. Smooth `python -m gnnrec.hge.rhgnn.smooth data/word2vec/ogbn_mag.model /home/zzy/output/rhgnn.pt /home/zzy/output/pos_graph_10.bin`

### HeCo
`python -m gnnrec.hge.heco.run_ogbn_mag data/word2vec/ogbn_mag.model /home/zzy/output/pos_graph_5.bin`

## RHCO
基于对比学习的关系感知异构图神经网络(Relation-aware Heterogeneous Graph Neural Network with Contrastive Learning, RHCO)

在HeCo的基础上改进：
* 网络结构编码器中的注意力向量改为关系的表示（类似于R-HGNN）
* 正样本选择方式由元路径条数改为预训练的R-HGNN计算的注意力权重
* 元路径视图编码器改为正样本图编码器，适配mini-batch训练
* Loss增加分类损失，训练方式由无监督改为半监督
* 在最后增加C&S后处理步骤

1. 预训练R-HGNN `python -m gnnrec.hge.rhgnn.run_ogbn_mag --save-path=/home/zzy/output/rhgnn.pt data/word2vec/ogbn_mag.model`
2. 构造正样本图 `python -m gnnrec.hge.rhco.build_pos_graph --num-samples=5 data/word2vec/ogbn_mag.model /home/zzy/output/rhgnn.pt /home/zzy/output/pos_graph_5.bin`
3. 训练模型 `python -m gnnrec.hge.rhco.run_ogbn_mag --contrast-weight=0.5 --save-path=/home/zzy/output/rhco.pt data/word2vec/ogbn_mag.model /home/zzy/output/pos_graph_5.bin`
4. Smooth `python -m gnnrec.hge.rhco.smooth data/word2vec/ogbn_mag.model /home/zzy/output/pos_graph_5.bin /home/zzy/output/rhco.pt`

## 实验结果
| 模型 | Train Acc | Valid Acc | Test Acc |
| --- | --- | --- | --- |
| R-GCN (full batch) | 0.3500 | 0.4043 | 0.3858 |
| HAN | 0.2154 | 0.2215 | 0.2364 |
| HetGNN | 0.4609 | 0.4093 | 0.4026 |
| HGT+average | 0.5956 | 0.4386 | 0.4160 |
| HGT+pretrained | 0.6507 | 0.4807 | 0.4491 |
| HGConv+average | 0.5032 | 0.4626 | 0.4507 |
| HGConv+pretrained | 0.5669 | 0.5039 | 0.4882 |
| R-HGNN （1层） | 0.5277 | 0.4921 | 0.4793 |
| R-HGNN | 0.5777 | 0.5321 | 0.5142 |
| C&S+正样本图 | 0.2602 | 0.2392 | 0.2453 -> 0.2334 |
| Smooth+正样本图 | 0.2602 | 0.2392 | 0.2453 -> 0.3090 |
| Smooth+引用图 | 0.2602 | 0.2392 | 0.2453 -> 0.2565 |
| HetGNN内容聚集+HGConv | 0.5919 | 0.4347 | 0.4006 |
| HGT注意力+HGConv | 0.5502 | 0.4469 | 0.4218 |
| HeCo+正样本图（无监督） | 0.2649 | 0.2448 | 0.2467 |
| HeCo+正样本图+半监督 | 0.2804 | 0.2618 | 0.2632 |
| HeCo+正样本图+半监督（使用z_sc） | 0.4228 | 0.3783 | 0.3629 |
| HeCo+Smooth+正样本图 | 0.4228 | 0.3783 | 0.3629 -> 0.3775 |
| R-HGNN+Smooth+正样本图 | 0.5777 | 0.5306 | 0.5124 -> 0.5200 |
| RHCO（1层）+旧正样本图 | 0.4320 | 0.3970 | 0.3798 -> 0.3865 |
| RHCO+旧正样本图 | 0.4885 | 0.4492 | 0.4286 -> 0.4301 |
| RHCO+正样本图 (α=0.0) | 0.4872 | 0.4504 | 0.4270 -> 0.4330 |
| RHCO+正样本图 (α=0.5) | 0.4758 | 0.4305 | 0.4000 -> 0.4098 |
