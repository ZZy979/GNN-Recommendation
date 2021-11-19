# 基于图神经网络的推荐算法
## 数据集
oag-cs - 使用OAG微软学术数据构造的计算机领域的学术网络（见 [readme](data/readme.md)）

## 预训练顶点嵌入
使用metapath2vec（随机游走+word2vec）预训练顶点嵌入，作为GNN模型的顶点输入特征
1. 随机游走
```shell
python -m gnnrec.kgrec.random_walk model/word2vec/oag_cs_corpus.txt
```

2. 训练词向量
```shell
python -m gnnrec.hge.metapath2vec.train_word2vec --size=128 --workers=8 model/word2vec/oag_cs_corpus.txt model/word2vec/oag_cs.model
```

## 召回
使用微调后的SciBERT模型（见 [readme](data/readme.md) 第2步）将查询词编码为向量，与预先计算好的论文标题向量计算余弦相似度，取top k
```shell
python -m gnnrec.kgrec.recall
```

召回结果示例：

graph neural network
```
0.9629	Aggregation Graph Neural Networks
0.9579	Neural Graph Learning: Training Neural Networks Using Graphs
0.9556	Heterogeneous Graph Neural Network
0.9552	Neural Graph Machines: Learning Neural Networks Using Graphs
0.9490	On the choice of graph neural network architectures
0.9474	Measuring and Improving the Use of Graph Information in Graph Neural Networks
0.9362	Challenging the generalization capabilities of Graph Neural Networks for network modeling
0.9295	Strategies for Pre-training Graph Neural Networks
0.9142	Supervised Neural Network Models for Processing Graphs
0.9112	Geometrically Principled Connections in Graph Neural Networks
```

recommendation algorithm based on knowledge graph
```
0.9172	Research on Video Recommendation Algorithm Based on Knowledge Reasoning of Knowledge Graph
0.8972	An Improved Recommendation Algorithm in Knowledge Network
0.8558	A personalized recommendation algorithm based on interest graph
0.8431	An Improved Recommendation Algorithm Based on Graph Model
0.8334	The Research of Recommendation Algorithm based on Complete Tripartite Graph Model
0.8220	Recommendation Algorithm based on Link Prediction and Domain Knowledge in Retail Transactions
0.8167	Recommendation Algorithm Based on Graph-Model Considering User Background Information
0.8034	A Tripartite Graph Recommendation Algorithm Based on Item Information and User Preference
0.7774	Improvement of TF-IDF Algorithm Based on Knowledge Graph
0.7770	Graph Searching Algorithms for Semantic-Social Recommendation
```

scholar disambiguation
```
0.9690	Scholar search-oriented author disambiguation
0.9040	Author name disambiguation in scientific collaboration and mobility cases
0.8901	Exploring author name disambiguation on PubMed-scale
0.8852	Author Name Disambiguation in Heterogeneous Academic Networks
0.8797	KDD Cup 2013: author disambiguation
0.8796	A survey of author name disambiguation techniques: 2010–2016
0.8721	Who is Who: Name Disambiguation in Large-Scale Scientific Literature
0.8660	Use of ResearchGate and Google CSE for author name disambiguation
0.8643	Automatic Methods for Disambiguating Author Names in Bibliographic Data Repositories
0.8641	A brief survey of automatic methods for author name disambiguation
```

## 精排
### 构造ground truth
（1）验证集

从AMiner发布的 [AI 2000人工智能全球最具影响力学者榜单](https://www.aminer.cn/ai2000) 抓取人工智能20个子领域的top 100学者
```shell
pip install scrapy>=2.3.0
cd gnnrec/kgrec/data/preprocess
scrapy runspider ai2000_crawler.py -a save_path=/home/zzy/GNN-Recommendation/data/rank/ai2000.json
```

与oag-cs数据集的学者匹配，并人工确认一些排名较高但未匹配上的学者，作为学者排名ground truth验证集
```shell
export DJANGO_SETTINGS_MODULE=academic_graph.settings.common
export SECRET_KEY=xxx
python -m gnnrec.kgrec.data.preprocess.build_author_rank build-val
```

（2）训练集

参考AI 2000的计算公式，根据某个领域的论文引用数加权求和构造学者排名，作为ground truth训练集

计算公式：
![计算公式](https://originalfileserver.aminer.cn/data/ranks/%E5%AD%A6%E8%80%85%E8%91%97%E4%BD%9C%E5%85%AC%E5%BC%8F.png)
即：假设一篇论文有n个作者，第k作者的权重为1/k，最后一个视为通讯作者，权重为1/2，归一化之后计算论文引用数的加权求和

```shell
python -m gnnrec.kgrec.data.preprocess.build_author_rank build-train
```

（3）评估ground truth训练集的质量
```shell
python -m gnnrec.kgrec.data.preprocess.build_author_rank eval
```

```
nDGC@100=0.2420 Precision@100=0.1859    Recall@100=0.2016
nDGC@50=0.2308  Precision@50=0.2494     Recall@50=0.1351
nDGC@20=0.2492  Precision@20=0.3118     Recall@20=0.0678
nDGC@10=0.2743  Precision@10=0.3471     Recall@10=0.0376
nDGC@5=0.3165   Precision@5=0.3765      Recall@5=0.0203
```

### 训练GNN模型
（1）论文召回
```shell
python -m gnnrec.kgrec.train_recall model/word2vec/oag-cs.model model/rhgnn_garec_recall.pt
```
训练完成后得到论文嵌入rank/paper_embed.pkl

TODO 使用这样得到的论文嵌入做第一步的召回结果又变得很差……分别尝试使用该嵌入和原始论文向量训练学者排名的效果

（2）学者排名
```shell
python -m gnnrec.kgrec.train_rank model/word2vec/oag-cs.model model/rhgnn_garec_rank.pt
```
训练完成后得到学者嵌入rank/author_embed.pkl

TODO 不管使用上一步得到的论文嵌入还是原始标题向量，学者排名效果都很差，nDGC@100只有0.02~0.03 `_(:з」∠)_`
训练学者排名前的论文召回改为直接使用论文召回的ground truth（领域和学者关联论文的交集），nDGC@100能到0.16~0.17，还是比较差
