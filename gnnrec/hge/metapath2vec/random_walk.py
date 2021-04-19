import argparse

from dgl.sampling import random_walk
from ogb.nodeproppred import DglNodePropPredDataset
from tqdm import trange

from gnnrec.config import DATA_DIR
from gnnrec.hge.utils import add_reverse_edges


def main():
    parser = argparse.ArgumentParser(description='metapath2vec基于元路径的随机游走')
    parser.add_argument('--num-walks', type=int, default=5, help='每个顶点游走次数')
    parser.add_argument('--walk-length', type=int, default=64, help='元路径重复次数')
    parser.add_argument('output_file', help='输出文件名')
    args = parser.parse_args()

    data = DglNodePropPredDataset('ogbn-mag', DATA_DIR)
    g = add_reverse_edges(data[0][0])
    # APFPPAIA
    metapath = [
        'writes', 'has_topic', 'has_topic_rev', 'cites', 'writes_rev',
        'affiliated_with', 'affiliated_with_rev'
    ]
    f = open(args.output_file, 'w')
    for aid in trange(g.num_nodes('author'), ncols=80):
        traces, types = random_walk(g, [aid] * args.num_walks, metapath=metapath * args.walk_length)
        f.writelines([trace2name(g, trace, types) + '\n' for trace in traces])
    f.close()


def trace2name(g, trace, types):
    return ' '.join(g.ntypes[t] + '_' + str(int(n)) for n, t in zip(trace, types) if int(n) >= 0)


if __name__ == '__main__':
    main()
