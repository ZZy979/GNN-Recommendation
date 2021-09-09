import argparse

import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from gnnrec.config import DATA_DIR
from gnnrec.hge.utils import add_reverse_edges


def random_walk(g, metapaths, num_walks, walk_length, output_file):
    """在异构图上按指定的元路径随机游走，将轨迹保存到指定文件

    :param g: DGLGraph 异构图
    :param metapaths: Dict[str, List[str]] 起点类型到元路径的映射，元路径表示为边类型列表，起点和终点类型应该相同
    :param num_walks: int 每个顶点的游走次数
    :param walk_length: int 元路径重复次数
    :param output_file: str 输出文件名
    :return:
    """
    with open(output_file, 'w') as f:
        for ntype, metapath in metapaths.items():
            print(ntype)
            loader = DataLoader(torch.arange(g.num_nodes(ntype)), batch_size=200)
            for b in tqdm(loader):
                nodes = torch.repeat_interleave(b, num_walks)
                traces, types = dgl.sampling.random_walk(g, nodes, metapath=metapath * walk_length)
                f.writelines([trace2name(g, trace, types) + '\n' for trace in traces])


def trace2name(g, trace, types):
    return ' '.join(g.ntypes[t] + '_' + str(int(n)) for n, t in zip(trace, types) if int(n) >= 0)


def main():
    parser = argparse.ArgumentParser(description='ogbn-mag数据集 metapath2vec基于元路径的随机游走')
    parser.add_argument('--num-walks', type=int, default=5, help='每个顶点游走次数')
    parser.add_argument('--walk-length', type=int, default=16, help='元路径重复次数')
    parser.add_argument('output_file', help='输出文件名')
    args = parser.parse_args()

    data = DglNodePropPredDataset('ogbn-mag', DATA_DIR)
    g = add_reverse_edges(data[0][0])
    metapaths = {
        'author': ['writes', 'has_topic', 'has_topic_rev', 'writes_rev'],  # APFPA
        'paper': ['writes_rev', 'writes', 'has_topic', 'has_topic_rev'],  # PAPFP
        'field_of_study': ['has_topic_rev', 'writes_rev', 'writes', 'has_topic'],  # FPAPF
        'institution': ['affiliated_with_rev', 'writes', 'writes_rev', 'affiliated_with']  # IAPAI
    }
    random_walk(g, metapaths, args.num_walks, args.walk_length, args.output_file)


if __name__ == '__main__':
    main()
