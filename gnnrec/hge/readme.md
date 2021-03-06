# ogbn-mag数据集
## 运行命令
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

### R-HGNN
`python -m gnnrec.hge.rhgnn.run_ogbn_mag data/word2vec/ogbn_mag.model`

### MyGNN
`python -m gnnrec.hge.mygnn.run_ogbn_mag data/word2vec/ogbn_mag.model`

## 预训练顶点嵌入
1. 随机游走 `python -m gnnrec.hge.metapath2vec.random_walk data/word2vec/ogbn_mag_corpus.txt`
2. 训练词向量 `python -m gnnrec.hge.metapath2vec.train_word2vec --size=128 --workers=8 data/word2vec/ogbn_mag_corpus.txt data/word2vec/ogbn_mag.model`

## 结果
| 模型 | Train Acc | Valid Acc | Test Acc |
| --- | --- | --- | --- |
| R-GCN (full batch) | 0.3500 | 0.4043 | 0.3858 |
| HAN | 0.2154 | 0.2215 | 0.2364 |
| HetGNN | 0.4609 | 0.4093 | 0.4026 |
| HGT+average | 0.5956 | 0.4386 | 0.4160 |
| HGT+pretrained | 0.6507 | 0.4807 | 0.4491 |
| HGConv+average | 0.5032 | 0.4626 | 0.4507 |
| HGConv+pretrained | 0.5669 | 0.5039 | 0.4882 |
| R-HGNN | 0.5777 | 0.5321 | 0.5142 |
| HetGNN内容聚集+HGConv | 0.5919 | 0.4347 | 0.4006 |
| HGT注意力+HGConv | 0.5502 | 0.4469 | 0.4218 |

## TODO
* R-GCN minibatch训练即使不使用邻居采样也无法达到与全图训练相同的准确率？
