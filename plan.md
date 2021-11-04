# 工作计划
## 总体计划
* [x] 2020.12~2021.3 继续阅读相关文献，考虑改进方法
* [x] 2021.3~2021.7 实现现有的异构图神经网络模型
* [x] 2021.7~2021.10 改进异构图神经网络模型，完成与现有方法的对比实验
* [ ] 2021.9~2021.10 构造学术网络数据集，实现基于图神经网络的推荐算法
* [ ] 2021.10~2021.11 整理实验结果，实现可视化系统，撰写毕业论文初稿
* [ ] 2021.11~2021.12 准备毕业答辩

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

### 基于图神经网络的推荐算法
#### 综述
* [x] 2020 IEEE [A Survey on Knowledge Graph-Based Recommender Systems](https://arxiv.org/pdf/2003.00911)
* [x] 2020 [Graph Neural Networks in Recommender Systems: A Survey](http://arxiv.org/pdf/2011.02260)
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

### 基于图神经网络的推荐算法
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
    * [x] 尝试C&S Baseline在ogbn-mag数据集上的效果 → 不加Correct步骤能提升更多，正样本图>引用图
    * [x] 尝试增加C&S后处理步骤（重点是标签传播图的构造）
        * [x] R-HGNN+C&S → 正样本图上微提升，引用图上下降
        * [x] HeCo+C&S → 26.32% -> 27.7%
* 2021.8.16~8.22
    * [x] 尝试HeCo的最终嵌入使用z_sc → 提升10%！
    * [x] 尝试将HeCo的网络结构编码器替换为R-HGNN
* 2021.8.23~8.29
    * [x] 写中期报告
* 2021.8.30~9.5
    * [x] 尝试将RHCO的网络结构编码器改为两层 → 提升4.4%
    * [x] 尝试其他构造正样本图的方法：训练集使用真实标签，验证集和测试集使用HGT预测 → 提升0.6%
* 2021.9.6~9.12
    * [x] 尝试将构造正样本图的方法改为使用预训练的R-HGNN模型计算的注意力权重、训练集使用真实标签 → 下降2.6%
    * [x] 使用metapath2vec预训练oag-cs数据集的顶点嵌入，备用
* 2021.9.13~9.19
    * [x] RHCO模型删除输入特征转换和dropout（网络结构编码器已包含）、增加学习率调度器（对比损失权重为0时应该达到与R-HGNN相似的性能） → +10%
    * [x] 设计推荐算法：使用SciBERT+对比学习实现召回：一篇论文的标题和关键词是一对正样本，使用（一个或两个）SciBERT分别将标题和关键词编码为向量，
      计算对比损失，以此方式进行微调；使用微调后的SciBERT模型将论文标题和输入关键词编码为向量，计算相似度即可召回与查询最相关的论文
* 2021.9.20~9.26
    * [x] 在oag-cs数据集上使用SciBERT+对比学习进行微调
    * [x] 实现输入关键词召回论文的功能
* 2021.9.27~10.10
    * [x] 实现推荐算法的精排部分
        * [x] 重新构造oag-cs数据集，使field顶点包含所有领域词
        * [x] 在oag-cs数据集上训练RHCO模型（预测任务：期刊分类），获取顶点表示向量
            * [x] 修改训练代码，使其能够适配不同的数据集
        * TODO 预测任务改为学者排名相关（例如学者和领域顶点的相似度），需要先获取ground truth：学者在某个领域的论文引用数之和，排序
    * [x] 初步实现可视化系统
        * [x] 创建Django项目（直接使用当前根目录即可）
        * [x] 创建数据库，将oag-cs数据导入数据库
        * [x] 实现论文召回的可视化
* 2021.10.11~10.17
    * 精排部分GNN模型训练思路：
        * （1）对于领域t召回论文，得到论文关联的学者集合，通过论文引用数之和构造学者排名；
        * （2）从排名中采样N个三元组(t, ap, an)，其中学者ap的排名在an之前，采样应包含简单样本（例如第1名和第10名）和困难样本（例如第1名和第3名）；
        * （3）计算三元组损失triplet_loss(t, ap, an) = d(t, ap) - d(t, an) + α
    * [x] 可视化系统：实现查看论文详情、学者详情等基本功能
    * [x] 开始写毕业论文
        * [x] 第一章 绪论
* 2021.10.18~10.24
    * [x] 异构图表示学习：增加ACM和DBLP数据集
    * [x] 写毕业论文
        * [x] 第二章 基础理论
        * [x] 第三章 基于对比学习的异构图表示学习模型
* 2021.10.25~10.31
    * [x] 完成毕业论文初稿
        * [x] 第四章 基于图神经网络的学术推荐算法
        * [x] 第五章 学术推荐系统设计与实现
    * [x] 异构图表示学习
        * [x] 正样本图改为类型的邻居各对应一个(PAP, PFP)，使用注意力组合
        * [x] 尝试：网络结构编码器由R-HGNN改为HGConv → ACM: -3.6%, DBLP: +4%, ogbn-mag: -1.86%
* 2021.11.1~11.7
    * [ ] 异构图表示学习
        * [ ] 完成参数敏感性分析和消融实验
    * [ ] 推荐算法精排部分
        * [x] 抓取AMiner AI 2000的人工智能学者榜单作为学者排名验证集
        * [x] 参考AI 2000的计算公式，使用某个领域的论文引用数加权求和构造学者排名ground truth训练集，采样三元组
        * [ ] 训练：使用三元组损失训练GNN模型
        * [ ] 预测：对于召回的论文构造子图，利用顶点嵌入计算查询词与学者的相似度，实现学者排名
