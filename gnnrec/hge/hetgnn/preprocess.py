import argparse
import os

from dgl.data.utils import save_graphs, save_info

from gnnrec.hge.hetgnn.utils import load_data


def preprocess(args):
    g, feats = load_data(args.pretrained_node_embed_path, args.neighbor_path, args.walks_per_node)
    save_graphs(os.path.join(args.save_path, 'ogbn_mag_neighbor_graph.bin'), [g])
    save_info(os.path.join(args.save_path, 'ogbn_mag_in_feats.pkl'), feats)
    print('邻居图和输入特征已保存到', args.save_path)


def main():
    parser = argparse.ArgumentParser(description='HetGNN预处理')
    parser.add_argument('--walks-per-node', type=int, default=5, help='每个顶点游走次数')
    parser.add_argument('pretrained_node_embed_path', help='预训练顶点嵌入的文件路径')
    parser.add_argument('neighbor_path', help='随机游走生成的顶点序列文件路径')
    parser.add_argument('save_path', help='保存预处理的邻居图和输入特征的路径')
    args = parser.parse_args()
    preprocess(args)


if __name__ == '__main__':
    main()
