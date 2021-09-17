# 基于知识图谱的推荐算法
## 数据集
oag-cs - 使用OAG微软学术数据构造的计算机领域的学术网络（见 [readme](data/readme.md)）

## 预处理
### 预训练顶点嵌入
使用metapath2vec（随机游走+word2vec）预训练顶点嵌入，作为GNN模型的顶点输入特征
1. 随机游走 `python -m gnnrec.kgrec.preprocess.random_walk model/word2vec/oag_cs_corpus.txt`
2. 训练词向量 `python -m gnnrec.hge.metapath2vec.train_word2vec --size=128 --workers=8 model/word2vec/oag_cs_corpus.txt model/word2vec/oag_cs.model`

### 获取顶点表示向量
训练GNN模型，获取顶点的表示向量，GNN模型使用异构图表示学习模块改进的RHCO模型

`python -m gnnrec.kgrec.preprocess.get_embeds`
