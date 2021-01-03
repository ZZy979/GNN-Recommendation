# 工作计划
## 总体计划
* [ ] 2020.12~2021.2 继续阅读国内外相关文献，细化技术方案
* [ ] 2021.3~2021.4 准备数据集，实现异构图表示学习模型，并完成与现有方法的对比实验
* [ ] 2021.5~2021.6 实现基于知识图谱的推荐算法并尝试优化
* [ ] 2021.7~2021.9 实现可视化系统
* [ ] 2021.10~2021.12 整理实验结果，撰写毕业论文

## 论文阅读
### 异构图表示学习
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
* [x] 2020 [Heterogeneous Network Representation Learning: A Unified Framework with Survey and Benchmark](https://arxiv.org/pdf/2004.00216)

### 基于知识图谱的推荐算法
* [x] 2020 IEEE [A Survey on Knowledge Graph-Based Recommender Systems](https://arxiv.org/pdf/2003.00911)
* [x] 2016 KDD [CKE](https://www.kdd.org/kdd2016/papers/files/adf0066-zhangA.pdf)
* [ ] 2018 [CFKG](https://arxiv.org/pdf/1803.06540)
* [ ] 2018 WSDM [SHINE](https://arxiv.org/pdf/1712.00732)
-----
* [x] 2013 IJCAI [Hete-MF](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.380.3668&rep=rep1&type=pdf)
* [ ] 2014 ICDM [Hete-CF](https://arxiv.org/pdf/1412.7610)
* [ ] 2013 RecSys [HeteRec](http://hanj.cs.illinois.edu/pdf/recsys13_xyu.pdf)
* [ ] 2015 CIKM [SemRec](https://papers-gamma.link/static/memory/pdfs/152-Shi_Semantic_Path_Based_Personalized_Recommendation_on_Weighted_HIN_2015.pdf)
* [ ] 2019 WWW [RuleRec](https://arxiv.org/pdf/1903.03714)
* [ ] 2018 KDD [MCRec](https://dl.acm.org/doi/pdf/10.1145/3219819.3219965)
* [ ] 2018 RecSys [RKGE](https://repository.tudelft.nl/islandora/object/uuid:9a3559e9-27b6-47cd-820d-d7ecc76cbc06/datastream/OBJ/download)
-----
* [x] 2018 CIKM [RippleNet](https://arxiv.org/pdf/1803.03467)
* [ ] 2019 KDD [AKUPM](https://dl.acm.org/doi/abs/10.1145/3292500.3330705)
* [ ] 2019 WWW [KGCN](https://arxiv.org/pdf/1904.12575)
* [ ] 2019 KDD [KGAT](https://arxiv.org/pdf/1905.07854)
* [ ] 2019 [KNI](https://arxiv.org/pdf/1908.04032)

## 复现模型
### 异构图表示学习
* [x] [GCN](https://github.com/ZZy979/pytorch-tutorial/tree/master/pytorch_tutorial/gnn/gcn)
* [x] [R-GCN]((https://github.com/ZZy979/pytorch-tutorial/tree/master/pytorch_tutorial/gnn/rgcn))
* [ ] GAT
* [ ] HetGNN
* [ ] HAN
* [ ] MAGNN
* [ ] HGT

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
    * [ ] 实现GAT
    * [ ] 实现HAN
    * [ ] 阅读论文Hete-CF
    * [ ] 阅读论文CFKG
