import random
from collections import defaultdict

import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset

if __name__ == '__main__':
    data = DglNodePropPredDataset('ogbn-mag', 'D:\\ogb')
    g, labels = data[0]
    labels = labels['paper'].squeeze(dim=1)

    label2node = defaultdict(list)
    for i, y in enumerate(labels.tolist()):
        label2node[y].append(i)

    u = []
    for i in range(g.num_nodes('paper')):
        u.extend(random.sample(label2node[labels[i].item()], 5))
    label_g = dgl.graph((u, torch.repeat_interleave(torch.arange(g.num_nodes('paper')), 5)))
    dgl.save_graphs('data/label_graph.bin', [label_g])
