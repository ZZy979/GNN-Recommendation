# ogbn-mag数据集
## 运行命令
* MLP: `python -m gnnrec.hge.mlp.run_ogbn_mag`
* GCN: `python -m gnnrec.hge.gcn.run_ogbn_mag`
* R-GCN (full batch): `python -m gnnrec.hge.rgcn.run_ogbn_mag_full`
* HAN: `python -m gnnrec.hge.han.run_ogbn_mag`
* HGConv+average: `python -m gnnrec.hge.hgconv.run_ogbn_mag`
* HGConv+metapath2vec
    * 随机游走： `python -m gnnrec.hge.metapath2vec.random_walk <corpus-path>`
    * 训练词向量（顶点嵌入）： `python -m gnnrec.hge.metapath2vec.train_word2vec --size=128 --workers=8 <corpus-path> <model-path>`
    * `python -m gnnrec.hge.hgconv.run_ogbn_mag --node-feat=metapath2vec --word2vec-path=<model-path>`

## 结果
| 模型 | Train Acc | Valid Acc | Test Acc |
| --- | --- | --- | --- |
| MLP | 0.2871 | 0.2603 | 0.2669 |
| GCN (PP) | 0.2802 | 0.2293 | 0.2184 |
| GCN (PAP) | 0.2973 | 0.2993 | 0.3086 |
| R-GCN (full batch) | 0.3500 | 0.4043 | 0.3858 |
| HAN (PAP) | 0.2109 | 0.1486 | 0.1538 |
| HGConv+average | 0.5402 | 0.4528 | 0.4243 |
| HGConv+metapath2vec | 0.5851 | 0.4942 | 0.4659 |

## TODO
* HAN模型目前仅使用了一条元路径PAP（转化后的同构图已经有6千万条边），尝试直接在异构图上做邻居采样
* R-GCN minibatch训练即使不使用邻居采样也无法达到与全图训练相同的准确率？
