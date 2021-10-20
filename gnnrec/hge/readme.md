# 异构图表示学习
## 数据集
| 数据集 | 顶点数 | 边数 | 目标顶点 | 类别数 |
| --- | --- | --- | --- | --- |
| ACM | 11246 | 34852 | paper | 3 |
| DBLP | 26128 | 239566 | author | 4 |
| ogbn-mag | 1939743 | 21111007 | paper | 349 |

## Baselines
### R-GCN (full batch)
```shell
python -m gnnrec.hge.rgcn.train --dataset=acm --epochs=10
python -m gnnrec.hge.rgcn.train --dataset=dblp --epochs=10
python -m gnnrec.hge.rgcn.train --dataset=ogbn-mag --num-hidden=48
```
（使用minibatch训练准确率就是只有20%多，不知道为什么）

### 预训练顶点嵌入
使用metapath2vec（随机游走+word2vec）预训练顶点嵌入，作为GNN模型的顶点输入特征
1. 随机游走
```shell
python -m gnnrec.hge.metapath2vec.random_walk model/word2vec/ogbn_mag_corpus.txt
```

2. 训练词向量
```shell
python -m gnnrec.hge.metapath2vec.train_word2vec --size=128 --workers=8 model/word2vec/ogbn_mag_corpus.txt model/word2vec/ogbn_mag.model
```

### HGT
```shell
python -m gnnrec.hge.hgt.train_full --dataset=acm
python -m gnnrec.hge.hgt.train_full --dataset=dblp
python -m gnnrec.hge.hgt.train --dataset=ogbn-mag --node-embed-path=model/word2vec/ogbn_mag.model --epochs=40
```

### HGConv
```shell
python -m gnnrec.hge.hgconv.train_full --dataset=acm --epochs=5
python -m gnnrec.hge.hgconv.train_full --dataset=dblp --epochs=20
python -m gnnrec.hge.hgconv.train --dataset=ogbn-mag --node-embed-path=model/word2vec/ogbn_mag.model
```

### R-HGNN
```shell
python -m gnnrec.hge.rhgnn.train_full --dataset=acm --epochs=30
python -m gnnrec.hge.rhgnn.train_full --dataset=dblp --epochs=20
python -m gnnrec.hge.rhgnn.train --dataset=ogbn-mag model/word2vec/ogbn_mag.model
```

### C&S
```shell
python -m gnnrec.hge.cs.train --dataset=acm --epochs=5
python -m gnnrec.hge.cs.train --dataset=dblp --epochs=5
python -m gnnrec.hge.cs.train --dataset=ogbn-mag --prop-graph=data/graph/pos_graph_5.bin
```

### HeCo
```shell
python -m gnnrec.hge.heco.train --dataset=ogbn-mag model/word2vec/ogbn_mag.model data/graph/pos_graph_5.bin
```
（ACM和DBLP的数据来自 https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/heco ，准确率和Micro-F1相等）

## RHCO
基于对比学习的关系感知异构图神经网络(Relation-aware Heterogeneous Graph Neural Network with Contrastive Learning, RHCO)

在HeCo的基础上改进：
* 网络结构编码器中的注意力向量改为关系的表示（类似于R-HGNN）
* 正样本选择方式由元路径条数改为预训练的HGT计算的注意力权重、训练集使用真实标签
* 元路径视图编码器改为正样本图编码器，适配mini-batch训练
* Loss增加分类损失，训练方式由无监督改为半监督
* 在最后增加C&S后处理步骤

ACM和DBLP
```shell
python -m gnnrec.hge.rhco.train_full --dataset=acm
python -m gnnrec.hge.rhco.train_full --dataset=dblp
```

ogbn-mag
```shell
python -m gnnrec.hge.hgt.train --dataset=ogbn-mag --node-feat=pretrained --node-embed-path=model/word2vec/ogbn_mag.model --epochs=40 --save-path=model/hgt_ogbn_mag.pt
python -m gnnrec.hge.rhco.build_pos_graph --dataset=ogbn-mag --num-samples=5 --use-label model/word2vec/ogbn_mag.model model/hgt_ogbn_mag.pt data/graph/pos_graph_5_label.bin
python -m gnnrec.hge.rhco.train --dataset=ogbn-mag --contrast-weight=0.5 model/word2vec/ogbn_mag.model data/graph/pos_graph_5_label.bin model/rhco_0.5_label.pt
python -m gnnrec.hge.rhco.smooth --dataset=ogbn-mag model/word2vec/ogbn_mag.model data/graph/pos_graph_5_label_c.bin model/rhco_0.5_label.pt
```

## 实验结果
| 模型 | ACM | DBLP | ogbn-mag |
| --- | --- | --- | --- |
| R-GCN | 0.7750 | 0.9490 | 0.3722 |
| HGT | 0.7660 | 0.7860 | 0.4504 |
| HGConv | 0.7550 | 0.9060 | 0.4828 |
| R-HGNN | 0.6890 | 0.8680 | 0.5196 |
| C&S | 0.7420 | 0.7970 | 0.3090 |
| HeCo | 0.8850 | 0.9070 | 0.2501 |
| RHCO | 0.8280 | 0.8890 | 0.5416 |
