# ogbn-mag数据集
## 运行命令
### MLP
`python -m gnnrec.hge.mlp.run_ogbn_mag`

### GCN
`python -m gnnrec.hge.gcn.run_ogbn_mag`

### R-GCN (full batch)
`python -m gnnrec.hge.rgcn.run_ogbn_mag_full`

### HAN
`python -m gnnrec.hge.han.run_ogbn_mag`

### HetGNN
1. 预处理 `python -m gnnrec.hge.hetgnn.preprocess data/word2vec/ogbn_mag.model data/word2vec/ogbn_mag_corpus.txt data/hetgnn`
2. 训练模型（有监督） `python -m gnnrec.hge.hetgnn.run_ogbn_mag data/hetgnn`

### HGT
#### 邻居平均(average)
`python -m gnnrec.hge.hgt.run_ogbn_mag`

#### 预训练顶点嵌入(pretrained)
`python -m gnnrec.hge.hgt.run_ogbn_mag --node-feat=pretrained --node-embed-path=data/word2vec/ogbn_mag.model --epochs=40`

### HGConv
#### 邻居平均(average)
`python -m gnnrec.hge.hgconv.run_ogbn_mag`

#### 预训练顶点嵌入(pretrained)
`python -m gnnrec.hge.hgconv.run_ogbn_mag --node-feat=pretrained --node-embed-path=data/word2vec/ogbn_mag.model`

### MyGNN
`python -m gnnrec.hge.mygnn.run_ogbn_mag`

## 预训练顶点嵌入
1. 随机游走 `python -m gnnrec.hge.metapath2vec.random_walk data/word2vec/ogbn_mag_corpus.txt`
2. 训练词向量 `python -m gnnrec.hge.metapath2vec.train_word2vec --size=128 --workers=8 data/word2vec/ogbn_mag_corpus.txt data/word2vec/ogbn_mag.model`

## 结果
| 模型 | Train Acc | Valid Acc | Test Acc |
| --- | --- | --- | --- |
| MLP | 0.2871 | 0.2603 | 0.2669 |
| GCN (PP) | 0.2802 | 0.2293 | 0.2184 |
| GCN (PAP) | 0.2973 | 0.2993 | 0.3086 |
| R-GCN (full batch) | 0.3500 | 0.4043 | 0.3858 |
| HAN | 0.2154 | 0.2215 | 0.2364 |
| HetGNN | 0.4609 | 0.4093 | 0.4026 |
| HGT + average | 0.5960 | 0.4410 | 0.4139 |
| HGT + pretrained | 0.6506 | 0.4757 | 0.4482 |
| HGConv + average | 0.5186 | 0.4737 | 0.4556 |
| HGConv + pretrained | 0.5776 | 0.5009 | 0.4796 |
| MyGNN | 0.5919 | 0.4347 | 0.4006 |

## TODO
* R-GCN minibatch训练即使不使用邻居采样也无法达到与全图训练相同的准确率？
