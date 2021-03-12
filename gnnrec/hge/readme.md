# ogbn-mag数据集
## 运行命令
* MLP: `python -m gnnrec.hge.run_ogbn_mag_mlp`
* GCN: `python -m gnnrec.hge.run_ogbn_mag_gcn`
* R-GCN: `python -m gnnrec.hge.run_ogbn_mag_rgcn`

## 结果
| 模型 | Train Acc | Valid Acc | Test Acc |
| --- | --- | --- | --- |
| MLP | 0.2871 | 0.2603 | 0.2669 |
| GCN (PP) | 0.2802 | 0.2293 | 0.2184 |
| GCN (PAP) | 0.2973 | 0.2993 | 0.3086 |
| R-GCN | 0.3412 | 0.4184 | 0.3972 |
| HAN (PAP) | 0.2109 | 0.1486 | 0.1538 |

## TODO
* HAN模型目前仅使用了一条元路径PAP（转化后的同构图已经有6千万条边），尝试直接在异构图上做邻居采样
* HGConv模型也存在内存占用过大的问题（可能是因为把torch.cat写成torch.stack，已修正，尚未重新验证）
