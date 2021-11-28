# 异构图表示学习
## 数据集
* [ACM](https://github.com/liun-online/HeCo/tree/main/data/acm) - ACM学术网络数据集
* [DBLP](https://github.com/liun-online/HeCo/tree/main/data/dblp) - DBLP学术网络数据集
* [ogbn-mag](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag) - OGB提供的微软学术数据集
* [oag-venue](../kgrec/data/venue.py) - oag-cs期刊分类数据集

| 数据集 | 顶点数 | 边数 | 目标顶点 | 类别数 |
| --- | --- | --- | --- | --- |
| ACM | 11246 | 34852 | paper | 3 |
| DBLP | 26128 | 239566 | author | 4 |
| ogbn-mag | 1939743 | 21111007 | paper | 349 |
| oag-venue | 4235169 | 34520417 | paper | 360 |

## Baselines
* [R-GCN](https://arxiv.org/pdf/1703.06103)
* [HGT](https://arxiv.org/pdf/2003.01332)
* [HGConv](https://arxiv.org/pdf/2012.14722)
* [R-HGNN](https://arxiv.org/pdf/2105.11122)
* [C&S](https://arxiv.org/pdf/2010.13993)
* [HeCo](https://arxiv.org/pdf/2105.09111)

### R-GCN (full batch)
```shell
python -m gnnrec.hge.rgcn.train --dataset=acm --epochs=10
python -m gnnrec.hge.rgcn.train --dataset=dblp --epochs=10
python -m gnnrec.hge.rgcn.train --dataset=ogbn-mag --num-hidden=48
python -m gnnrec.hge.rgcn.train --dataset=oag-venue --num-hidden=48 --epochs=30
```
（使用minibatch训练准确率就是只有20%多，不知道为什么）

### 预训练顶点嵌入
使用metapath2vec（随机游走+word2vec）预训练顶点嵌入，作为GNN模型的顶点输入特征
```shell
python -m gnnrec.hge.metapath2vec.random_walk model/word2vec/ogbn-mag_corpus.txt
python -m gnnrec.hge.metapath2vec.train_word2vec --size=128 --workers=8 model/word2vec/ogbn-mag_corpus.txt model/word2vec/ogbn-mag.model
```

### HGT
```shell
python -m gnnrec.hge.hgt.train_full --dataset=acm
python -m gnnrec.hge.hgt.train_full --dataset=dblp
python -m gnnrec.hge.hgt.train --dataset=ogbn-mag --node-embed-path=model/word2vec/ogbn-mag.model --epochs=40
python -m gnnrec.hge.hgt.train --dataset=oag-venue --node-embed-path=model/word2vec/oag-cs.model --epochs=40
```

### HGConv
```shell
python -m gnnrec.hge.hgconv.train_full --dataset=acm --epochs=5
python -m gnnrec.hge.hgconv.train_full --dataset=dblp --epochs=20
python -m gnnrec.hge.hgconv.train --dataset=ogbn-mag --node-embed-path=model/word2vec/ogbn-mag.model
python -m gnnrec.hge.hgconv.train --dataset=oag-venue --node-embed-path=model/word2vec/oag-cs.model
```

### R-HGNN
```shell
python -m gnnrec.hge.rhgnn.train_full --dataset=acm --num-layers=1 --epochs=15
python -m gnnrec.hge.rhgnn.train_full --dataset=dblp --epochs=20
python -m gnnrec.hge.rhgnn.train --dataset=ogbn-mag model/word2vec/ogbn-mag.model
python -m gnnrec.hge.rhgnn.train --dataset=oag-venue --epochs=50 model/word2vec/oag-cs.model
```

### C&S
```shell
python -m gnnrec.hge.cs.train --dataset=acm --epochs=5
python -m gnnrec.hge.cs.train --dataset=dblp --epochs=5
python -m gnnrec.hge.cs.train --dataset=ogbn-mag --prop-graph=data/graph/pos_graph_ogbn-mag_t5.bin
python -m gnnrec.hge.cs.train --dataset=oag-venue --prop-graph=data/graph/pos_graph_oag-venue_t5.bin
```

### HeCo
```shell
python -m gnnrec.hge.heco.train --dataset=ogbn-mag model/word2vec/ogbn-mag.model data/graph/pos_graph_ogbn-mag_t5.bin
python -m gnnrec.hge.heco.train --dataset=oag-venue model/word2vec/oag-cs.model data/graph/pos_graph_oag-venue_t5.bin
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

ACM
```shell
python -m gnnrec.hge.hgt.train_full --dataset=acm --save-path=model/hgt/hgt_acm.pt
python -m gnnrec.hge.rhco.build_pos_graph_full --dataset=acm --num-samples=5 --use-label model/hgt/hgt_acm.pt data/graph/pos_graph_acm_t5l.bin
python -m gnnrec.hge.rhco.train_full --dataset=acm data/graph/pos_graph_acm_t5l.bin
```

DBLP
```shell
python -m gnnrec.hge.hgt.train_full --dataset=dblp --save-path=model/hgt/hgt_dblp.pt
python -m gnnrec.hge.rhco.build_pos_graph_full --dataset=dblp --num-samples=5 --use-label model/hgt/hgt_dblp.pt data/graph/pos_graph_dblp_t5l.bin
python -m gnnrec.hge.rhco.train_full --dataset=dblp --use-data-pos data/graph/pos_graph_dblp_t5l.bin
```

ogbn-mag（第3步如果中断可使用--load-path参数继续训练）
```shell
python -m gnnrec.hge.hgt.train --dataset=ogbn-mag --node-embed-path=model/word2vec/ogbn-mag.model --epochs=40 --save-path=model/hgt/hgt_ogbn-mag.pt
python -m gnnrec.hge.rhco.build_pos_graph --dataset=ogbn-mag --num-samples=5 --use-label model/word2vec/ogbn-mag.model model/hgt/hgt_ogbn-mag.pt data/graph/pos_graph_ogbn-mag_t5l.bin
python -m gnnrec.hge.rhco.train --dataset=ogbn-mag --num-hidden=64 --contrast-weight=0.9 model/word2vec/ogbn-mag.model data/graph/pos_graph_ogbn-mag_t5l.bin model/rhco/rhco_ogbn-mag_d64_a0.9_t5l.pt
python -m gnnrec.hge.rhco.smooth --dataset=ogbn-mag model/word2vec/ogbn-mag.model data/graph/pos_graph_ogbn-mag_t5l.bin model/rhco/rhco_ogbn-mag_d64_a0.9_t5l.pt
```

oag-venue
```shell
python -m gnnrec.hge.hgt.train --dataset=oag-venue --node-embed-path=model/word2vec/oag-cs.model --epochs=40 --save-path=model/hgt/hgt_oag-venue.pt
python -m gnnrec.hge.rhco.build_pos_graph --dataset=oag-venue --num-samples=5 --use-label model/word2vec/oag-cs.model model/hgt/hgt_oag-venue.pt data/graph/pos_graph_oag-venue_t5l.bin
python -m gnnrec.hge.rhco.train --dataset=oag-venue --num-hidden=64 --contrast-weight=0.9 model/word2vec/oag-cs.model data/graph/pos_graph_oag-venue_t5l.bin model/rhco/rhco_oag-venue.pt
python -m gnnrec.hge.rhco.smooth --dataset=oag-venue model/word2vec/oag-cs.model data/graph/pos_graph_oag-venue_t5l.bin model/rhco/rhco_oag-venue.pt
```

消融实验
```shell
python -m gnnrec.hge.rhco.train --dataset=ogbn-mag --model=RHCO_sc model/word2vec/ogbn-mag.model data/graph/pos_graph_ogbn-mag_t5l.bin model/rhco/rhco_sc_ogbn-mag.pt
python -m gnnrec.hge.rhco.train --dataset=oag-venue --model=RHCO_sc model/word2vec/oag-cs.model data/graph/pos_graph_oag-venue_t5l.bin model/rhco/rhco_sc_oag-venue.pt
python -m gnnrec.hge.rhco.train --dataset=ogbn-mag --model=RHCO_pg model/word2vec/ogbn-mag.model data/graph/pos_graph_ogbn-mag_t5.bin model/rhco/rhco_pg_ogbn-mag.pt
python -m gnnrec.hge.rhco.train --dataset=oag-venue --model=RHCO_pg model/word2vec/oag-cs.model data/graph/pos_graph_oag-venue_t5.bin model/rhco/rhco_pg_oag-venue.pt
```

## 实验结果
[顶点分类](result/node_classification.csv)

[参数敏感性分析](result/param_analysis.csv)

[消融实验](result/ablation_study.csv)
