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
`python -m gnnrec.hge.rgcn.run_ogbn_mag --num-hidden=48 --dropout=0.8`

（使用minibatch训练准确率就是只有20%多，不知道为什么）

### HAN
`python -m gnnrec.hge.han.run_ogbn_mag`

### HetGNN
1. 预处理 `python -m gnnrec.hge.hetgnn.preprocess model/word2vec/ogbn_mag.model model/word2vec/ogbn_mag_corpus.txt model/hetgnn`
2. 训练模型（有监督） `python -m gnnrec.hge.hetgnn.run_ogbn_mag model/hetgnn`

### 预训练顶点嵌入
使用metapath2vec（随机游走+word2vec）预训练顶点嵌入，作为GNN模型的顶点输入特征
1. 随机游走 `python -m gnnrec.hge.metapath2vec.random_walk model/word2vec/ogbn_mag_corpus.txt`
2. 训练词向量 `python -m gnnrec.hge.metapath2vec.train_word2vec --size=128 --workers=8 model/word2vec/ogbn_mag_corpus.txt model/word2vec/ogbn_mag.model`

### HGT
#### 邻居平均(average)
`python -m gnnrec.hge.hgt.run_ogbn_mag`

#### 预训练顶点嵌入(pretrained)
`python -m gnnrec.hge.hgt.run_ogbn_mag --node-feat=pretrained --node-embed-path=model/word2vec/ogbn_mag.model --epochs=40`

### HGConv
`python -m gnnrec.hge.hgconv.run_ogbn_mag --node-feat=pretrained --node-embed-path=model/word2vec/ogbn_mag.model`

### R-HGNN
`python -m gnnrec.hge.rhgnn.run_ogbn_mag model/word2vec/ogbn_mag.model`

### C&S
Linear+Smooth+正样本图

`python -m gnnrec.hge.cs.run_ogbn_mag data/graph/pos_graph_5.bin`

### HeCo
`python -m gnnrec.hge.heco.run_ogbn_mag model/word2vec/ogbn_mag.model data/graph/pos_graph_5.bin`

## RHCO
基于对比学习的关系感知异构图神经网络(Relation-aware Heterogeneous Graph Neural Network with Contrastive Learning, RHCO)

在HeCo的基础上改进：
* 网络结构编码器中的注意力向量改为关系的表示（类似于R-HGNN）
* 正样本选择方式由元路径条数改为预训练的HGT计算的注意力权重、训练集使用真实标签
* 元路径视图编码器改为正样本图编码器，适配mini-batch训练
* Loss增加分类损失，训练方式由无监督改为半监督
* 在最后增加C&S后处理步骤

1. 预训练HGT `python -m gnnrec.hge.hgt.run_ogbn_mag --node-feat=pretrained --node-embed-path=model/word2vec/ogbn_mag.model --epochs=40 --save-path=model/hgt_pretrain.pt`
2. 构造正样本图 `python -m gnnrec.hge.rhco.build_pos_graph --num-samples=5 model/word2vec/ogbn_mag.model model/hgt_pretrain.pt data/graph/pos_graph_5_label.bin`
3. 训练模型（如果中断可使用--load-path参数继续训练） `python -m gnnrec.hge.rhco.run_ogbn_mag --contrast-weight=0.5 model/word2vec/ogbn_mag.model data/graph/pos_graph_5_label.bin model/rhco_0.5_label.pt`
4. Smooth `python -m gnnrec.hge.rhco.smooth model/word2vec/ogbn_mag.model data/graph/pos_graph_5_label_c.bin model/rhco_0.5_label.pt`

## 实验结果
| 模型 | Train Acc | Valid Acc | Test Acc |
| --- | --- | --- | --- |
| R-GCN (full batch) | 0.8526 | 0.3906 | 0.3722 |
| HAN | 0.2154 | 0.2215 | 0.2364 |
| HetGNN | 0.4609 | 0.4093 | 0.4026 |
| HGT+average | 0.5956 | 0.4386 | 0.4160 |
| HGT+pretrained | 0.6510 | 0.4804 | 0.4504 |
| HGConv | 0.5653 | 0.5007 | 0.4828 |
| R-HGNN | 0.5907 | 0.5318 | 0.5196 |
| C&S+正样本图 | 0.2602 | 0.2392 | 0.2453 -> 0.2334 |
| Smooth+正样本图 | 0.2602 | 0.2392 | 0.2453 -> 0.3090 |
| HeCo+正样本图（无监督） | 0.2696 | 0.2479 | 0.2501 |
| RHCO+正样本图 (α=0.0) | 0.5751 | 0.5100 | 0.4860 -> 0.5352 |
| RHCO+正样本图 (α=0.2) | 0.6165 | 0.5158 | 0.4871 -> 0.5348 |
| RHCO+正样本图 (α=0.5) | 0.6086 | 0.5159 | 0.4878 -> 0.5346 |
| RHCO+正样本图 (α=0.8) | 0.6091 | 0.5196 | 0.4964 -> 0.5416 |
