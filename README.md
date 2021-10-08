# GNN-Recommendation
毕业设计：基于图神经网络的异构图表示学习和推荐算法研究

## 目录结构
```
GNN-Recommendation/
    gnnrec/             算法模块顶级包
        hge/            异构图表示学习模块
        kgrec/          基于知识图谱的推荐算法模块
    data/               数据集目录（已添加.gitignore）
    model/              模型保存目录（已添加.gitignore）
    academic_graph/     Django项目模块
    rank/               Django应用
    manage.py           Django管理脚本
```

## 异构图表示学习
### 数据集
[ogbn-mag](https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag) - OGB提供的微软学术数据集

### Baselines
* [R-GCN](https://arxiv.org/pdf/1703.06103)
* [HAN](https://arxiv.org/pdf/1903.07293)
* [HetGNN](https://dl.acm.org/doi/pdf/10.1145/3292500.3330961)
* [HGT](https://arxiv.org/pdf/2003.01332)
* [HGConv](https://arxiv.org/pdf/2012.14722)
* [R-HGNN](https://arxiv.org/pdf/2105.11122)
* [C&S](https://arxiv.org/pdf/2010.13993)
* [HeCo](https://arxiv.org/pdf/2105.09111)

## 基于知识图谱的推荐算法
### 数据集
oag-cs - 使用OAG微软学术数据构造的计算机领域的学术网络

## Django配置
MySQL数据库连接配置文件：根目录下.mylogin.cnf

```ini
[client]
host = x.x.x.x
port = 3306
user = username
password = password
database = database
default-character-set = utf8
```
