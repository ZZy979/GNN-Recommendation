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
python -m gnnrec.kgrec.recall data/oag/cs/paper_feat.pkl model/scibert.pt data/oag/cs/mag_papers.txt
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
构造学者排名数据，作为ground truth
```shell
python -m gnnrec.kgrec.data.preprocess.build_author_rank build data/oag/cs/mag_fields.txt data/oag/cs/paper_feat.pkl model/scibert.pt data/rank/author_rank.json
```
