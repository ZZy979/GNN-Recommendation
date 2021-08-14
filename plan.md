# 工作计划
## 总体计划
* [x] 2020.12~2021.2 继续阅读国内外相关文献，细化技术方案
* [x] 2021.3~2021.4 准备数据集，实现异构图表示学习模型，并完成与现有方法的对比实验
* [ ] 2021.5~2021.6 实现基于知识图谱的推荐算法并尝试优化
* [ ] 2021.7~2021.9 实现可视化系统
* [ ] 2021.10~2021.12 整理实验结果，撰写毕业论文

## 论文阅读
### 异构图表示学习
#### 综述
* [x] 2020 [Heterogeneous Network Representation Learning: A Unified Framework with Survey and Benchmark](https://arxiv.org/pdf/2004.00216)
* [x] 2020 [A Survey on Heterogeneous Graph Embedding: Methods, Techniques, Applications and Sources](https://arxiv.org/pdf/2011.14867)
#### 图神经网络
* [x] 2014 KDD [DeepWalk](https://arxiv.org/pdf/1403.6652)
* [x] 2016 KDD [node2vec](https://arxiv.org/pdf/1607.00653)
* [x] 2017 KDD [metapath2vec](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf)
* [x] 2017 ICLR [GCN](https://arxiv.org/pdf/1609.02907)
* [x] 2018 ESWC [R-GCN](https://arxiv.org/pdf/1703.06103)
* [x] 2018 ICLR [GAT](https://arxiv.org/pdf/1710.10903)
* [x] 2019 KDD [HetGNN](https://dl.acm.org/doi/pdf/10.1145/3292500.3330961)
* [x] 2019 WWW [HAN](https://arxiv.org/pdf/1903.07293)
* [x] 2020 WWW [MAGNN](https://arxiv.org/pdf/2002.01680)
* [x] 2020 WWW [HGT](https://arxiv.org/pdf/2003.01332)
* [x] 2020 [HGConv](https://arxiv.org/pdf/2012.14722)
* [x] 2020 KDD [GPT-GNN](https://arxiv.org/pdf/2006.15437)
* [x] 2020 ICLR [GraphSAINT](https://openreview.net/pdf?id=BJe8pkHFwS)
* [x] 2020 [SIGN](https://arxiv.org/pdf/2004.11198)
* [x] 2020 [NARS](https://arxiv.org/pdf/2011.09679)
* [x] 2021 ICLR [SuperGAT](https://openreview.net/pdf?id=Wi5KUNlqWty)
* [x] 2021 [R-HGNN](https://arxiv.org/pdf/2105.11122)
#### 自监督/预训练
* [x] 2020 [Self-Supervised Graph Representation Learning via Global Context Prediction](https://arxiv.org/pdf/2003.01604)
* [ ] 2020 ICML [When Does Self-Supervision Help Graph Convolutional Networks?](http://proceedings.mlr.press/v119/you20a/you20a.pdf)
* [x] 2020 ICLR [Strategies for Pre-Training Graph Neural Networks](https://www.openreview.net/pdf?id=HJlWWJSFDH)
* [x] 2021 WWW [Self-Supervised Learning of Contextual Embeddings for Link Prediction in Heterogeneous Networks](https://arxiv.org/pdf/2007.11192)
* [x] 2021 KDD [HeCo](https://arxiv.org/pdf/2105.09111)
#### 其他
* [x] 2021 ICLR [C&S](https://arxiv.org/pdf/2010.13993)

### 基于知识图谱的推荐算法
#### 综述
* [x] 2020 IEEE [A Survey on Knowledge Graph-Based Recommender Systems](https://arxiv.org/pdf/2003.00911)
#### 基于嵌入的方法
* [x] 2016 KDD [CKE](https://www.kdd.org/kdd2016/papers/files/adf0066-zhangA.pdf)
* [x] 2018 [CFKG](https://arxiv.org/pdf/1803.06540)
* [ ] 2018 WSDM [SHINE](https://arxiv.org/pdf/1712.00732)
#### 基于路径的方法
* [x] 2013 IJCAI [Hete-MF](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.380.3668&rep=rep1&type=pdf)
* [x] 2014 ICDM [Hete-CF](https://arxiv.org/pdf/1412.7610)
* [x] 2013 RecSys [HeteRec](http://hanj.cs.illinois.edu/pdf/recsys13_xyu.pdf)
* [ ] 2015 CIKM [SemRec](https://papers-gamma.link/static/memory/pdfs/152-Shi_Semantic_Path_Based_Personalized_Recommendation_on_Weighted_HIN_2015.pdf)
* [ ] 2019 WWW [RuleRec](https://arxiv.org/pdf/1903.03714)
* [ ] 2018 KDD [MCRec](https://dl.acm.org/doi/pdf/10.1145/3219819.3219965)
* [ ] 2018 RecSys [RKGE](https://repository.tudelft.nl/islandora/object/uuid:9a3559e9-27b6-47cd-820d-d7ecc76cbc06/datastream/OBJ/download)
#### 嵌入和路径结合的方法
* [x] 2018 CIKM [RippleNet](https://arxiv.org/pdf/1803.03467)
* [ ] 2019 KDD [AKUPM](https://dl.acm.org/doi/abs/10.1145/3292500.3330705)
* [x] 2019 WWW [KGCN](https://arxiv.org/pdf/1904.12575)
* [x] 2019 KDD [KGAT](https://arxiv.org/pdf/1905.07854)
* [ ] 2019 [KNI](https://arxiv.org/pdf/1908.04032)

## 复现模型
### 异构图表示学习
* [x] [GCN](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/gcn)
* [x] [R-GCN](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/rgcn)
* [x] [GAT](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/gat)
* [x] [HetGNN](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/hetgnn)
* [x] [HAN](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/han)
* [x] [MAGNN](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/magnn)
* [x] [HGT](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/hgt)
* [x] [metapath2vec](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/metapath2vec)
* [x] [SIGN](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/sign)
* [x] [HGConv](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/hgconv)
* [x] [SuperGAT](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/supergat)
* [x] [R-HGNN](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/rhgnn)
* [x] [C&S](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/cs)
* [x] [HeCo](https://github.com/ZZy979/pytorch-tutorial/tree/master/gnn/heco)

### 基于知识图谱的推荐算法
* [ ] CKE
* [ ] RippleNet
* [ ] KGCN
* [ ] KGAT

## 具体计划
* 2020.12.21~12.25
    * [x] 阅读论文CKE
    * [x] 实现GCN
    * [x] 阅读论文R-GCN
* 2020.12.28~2021.1.1
    * [x] 实现R-GCN
    * [x] 阅读论文RippleNet
    * [x] 阅读论文Hete-MF
* 2021.1.4~1.8
    * [x] 实现GAT
    * [x] 实现HAN
    * [x] 阅读论文Hete-CF
    * [x] 阅读论文CFKG
* 2021.1.11~1.15
    * [x] 实现MAGNN
    * [x] 阅读论文KGCN
    * [x] 阅读论文HeteRec
* 2021.1.18~1.22
    * [x] 阅读论文KGAT
    * [x] 使用OGB数据集做实验
* 2021.2.22~2.26
    * [x] 实现ogbn-mag数据集baseline: MLP和Full-batch GCN
    * [x] 查找最新异构图表示学习论文
* 2021.3.1~3.5
    * [x] 实现ogbn-mag数据集 R-GCN模型
    * [x] 阅读论文HGConv
* 2021.3.8~3.12
    * [x] 实现ogbn-mag数据集 HGConv模型
    * [x] 尝试解决ogbn-mag数据集 HAN模型内存占用过大的问题
    * [x] 阅读论文NARS
* 2021.3.15~3.19
    * [x] 阅读论文SIGN
    * [x] 阅读论文GraphSAINT
    * [x] 阅读论文SuperGAT
* 2021.3.22~3.26
    * 继续看上周的论文（找实习面试好难啊😢）
    * 2021.4.1 人生中第一个offer🎉
* 2021.4.5~4.9
    * [x] 重新训练ogbn-mag数据集 HGConv模型
    * [x] 实现SuperGAT
* 2021.4.12~4.18
    * [x] 阅读论文GPT-GNN
    * [x] 实现metapath2vec
* 2021.4.19~4.25
    * [x] 使用子图采样的方法在ogbn-mag数据集上训练HAN模型
    * [x] 使用metapath2vec预训练ogbn-mag数据集的顶点特征，重新跑HGConv模型
    * [x] 阅读综述A Survey on Heterogeneous Graph Embedding
* 2021.4.26~5.9
    * [x] 实现HGT
    * [x] 实现HetGNN
    * [x] 实现ogbn-mag数据集 HGT模型
    * [x] 实现ogbn-mag数据集 HetGNN模型
    * [x] 尝试改进：HetGNN的内容聚集+HGConv
* 2021.5.10~5.16
    * [x] 阅读论文Strategies for Pre-Training Graph Neural Networks
    * [x] 阅读论文Self-Supervised Graph Representation Learning via Global Context Prediction
* 2021.5.17~5.23
    * [x] 继续尝试异构图表示学习模型的改进
    * [x] 阅读论文Self-Supervised Learning of Contextual Embeddings for Link Prediction in Heterogeneous Networks
    * [x] 整理OAG数据集
* 2021.5.24~5.30
    * 实习第一周
    * [x] 阅读论文R-HGNN
* 2021.5.31~6.6
    * [x] 实现R-HGNN
* 2021.6.7~6.13
    * [x] 利用OAG数据集构造计算机领域的子集
* 2021.6.14~6.20
    * [x] 阅读论文C&S
    * [x] 完成SciBERT模型的fine-tune，获取OAG-CS数据集的paper顶点输入特征
* 2021.7.5~7.18
    * [x] 实现C&S
    * [x] 阅读论文HeCo
* 2021.7.19~7.25
    * [x] 实现HeCo
* 2021.7.26~8.1
    * [x] 尝试改进HeCo：mini-batch训练、元路径编码器改为其他方式、Loss增加分类损失
    * 7.29 和吴嘉伟讨论HeCo的改进思路
        * 正样本选择策略：在下游任务上预训练一个两层HGT，第二层的注意力权重是一阶邻居对目标顶点的权重，
          第一层的注意力权重是二阶邻居对一阶邻居的权重，取类型与目标顶点相同的二阶邻居，并将两个权重相加，
          得到二阶邻居（同类型）对目标顶点的权重，取top-k作为目标顶点的正样本
        * 使用上面得到的正样本可以构造一个目标类型顶点的同构图，用于替换元路径编码器中基于元路径的同构图
    * [x] 确认HGT中对注意力权重做softmax的方式（同类型/跨类型）→同类型
* 2021.8.2~8.8
    * [x] 实现使用预训练的HGT计算的注意力权重选择HeCo的正样本的方法
    * [x] 将HeCo迁移到ogbn-mag数据集上，尝试效果 → 24.67%
        * [x] 元路径视图编码器替换为正样本图上的GCN编码器
        * [x] 适配mini-batch训练
* 2021.8.9~8.15
    * [x] 将HeCo训练方式改为半监督（loss增加分类损失），尝试效果 → 26.32%
    * [ ] 尝试其他构造正样本图的方法
    * [x] 尝试C&S Baseline在ogbn-mag数据集上的效果 → 不加Correct步骤能提升更多，正样本图>引用图
    * [x] 尝试增加C&S后处理步骤（重点是标签传播图的构造）
        * [x] R-HGNN+C&S → 正样本图上微提升，引用图上下降
        * [x] HeCo+C&S → 26.32% -> 27.7%
