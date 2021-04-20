# ogbn-mag数据集
## 运行命令
* MLP: `python -m gnnrec.hge.mlp.run_ogbn_mag`
* GCN: `python -m gnnrec.hge.gcn.run_ogbn_mag`
* R-GCN: `python -m gnnrec.hge.rgcn.run_ogbn_mag`
* HAN: `python -m gnnrec.hge.han.run_ogbn_mag`
* HGConv
    * 随机游走： `python -m gnnrec.hge.metapath2vec.random_walk data/word2vec/ogbn_mag_corpus.txt`
    * 训练词向量（顶点嵌入）： `python -m gnnrec.hge.metapath2vec.train_word2vec --size=128 --workers=8 data/word2vec/ogbn_mag_corpus.txt data/word2vec/ogbn_mag.model`
    * `python -m gnnrec.hge.hgconv.run_ogbn_mag`

## 结果
| 模型 | Train Acc | Valid Acc | Test Acc |
| --- | --- | --- | --- |
| MLP | 0.2871 | 0.2603 | 0.2669 |
| GCN (PP) | 0.2802 | 0.2293 | 0.2184 |
| GCN (PAP) | 0.2973 | 0.2993 | 0.3086 |
| R-GCN | 0.3412 | 0.4184 | 0.3972 |
| HAN (PAP) | 0.2109 | 0.1486 | 0.1538 |
| HGConv | 0.5446 | 0.4772 | 0.4530 |

## TODO
* HAN模型目前仅使用了一条元路径PAP（转化后的同构图已经有6千万条边），尝试直接在异构图上做邻居采样
* HGConv模型目前使用的输入特征：学者是论文的平均、机构是学者的平均、领域是随机，而原代码中是使用metapath2vec预训练
