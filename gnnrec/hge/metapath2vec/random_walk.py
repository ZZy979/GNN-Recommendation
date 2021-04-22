import argparse

import torch
from dgl.sampling import random_walk
from ogb.nodeproppred import DglNodePropPredDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from gnnrec.config import DATA_DIR
from gnnrec.hge.utils import add_reverse_edges


def main():
    parser = argparse.ArgumentParser(description='metapath2vec基于元路径的随机游走')
    parser.add_argument('--num-walks', type=int, default=5, help='每个顶点游走次数')
    parser.add_argument('--walk-length', type=int, default=32, help='元路径重复次数')
    parser.add_argument('output_file', help='输出文件名')
    args = parser.parse_args()

    data = DglNodePropPredDataset('ogbn-mag', DATA_DIR)
    g = add_reverse_edges(data[0][0])
    metapaths = {
        'author': [
            'writes', 'has_topic', 'has_topic_rev', 'writes_rev',
            'affiliated_with', 'affiliated_with_rev'
        ],  # APFPAIA
        'field_of_study': ['has_topic_rev', 'writes_rev', 'writes', 'has_topic'],  # FPAPF
        'institution': ['affiliated_with_rev', 'writes', 'writes_rev', 'affiliated_with']  # IAPAI
    }
    f = open(args.output_file, 'w')
    for ntype, metapath in metapaths.items():
        print(ntype)
        loader = DataLoader(torch.arange(g.num_nodes(ntype)), batch_size=200)
        for b in tqdm(loader):
            nodes = torch.repeat_interleave(b, args.num_walks)
            traces, types = random_walk(g, nodes, metapath=metapath * args.walk_length)
            f.writelines([trace2name(g, trace, types) + '\n' for trace in traces])
    f.close()


def trace2name(g, trace, types):
    return ' '.join(
        g.ntypes[t] + '_' + str(int(n)) for n, t in zip(trace, types)
        if int(n) >= 0 and g.ntypes[t] != 'paper'
    )


if __name__ == '__main__':
    main()
