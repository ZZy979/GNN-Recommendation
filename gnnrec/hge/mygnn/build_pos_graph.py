import argparse

import dgl
import torch
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from ogb.nodeproppred import DglNodePropPredDataset
from tqdm import tqdm

from gnnrec.config import DATA_DIR
from gnnrec.hge.hgt.model import HGT
from gnnrec.hge.utils import get_device, load_pretrained_node_embed, add_reverse_edges


def main():
    args = parse_args()
    print(args)
    device = get_device(args.device)

    data = DglNodePropPredDataset('ogbn-mag', DATA_DIR)
    g, _ = data[0]
    g = add_reverse_edges(g)
    load_pretrained_node_embed(g, args.node_embed_path)

    num_neighbors = [
        # 第1层：只保留PA, PP, PF边，其中PF边需要采样
        {'writes_rev': -1, 'cites': -1, 'has_topic': 5},
        # 第2层：只保留AP, PP, FP边
        {'writes': -1, 'cites': -1, 'has_topic_rev': -1},
    ]
    for i in range(len(num_neighbors)):
        d = dict.fromkeys(g.etypes, 0)
        d.update(num_neighbors[i])
        num_neighbors[i] = d
    sampler = MultiLayerNeighborSampler(num_neighbors)
    loader = NodeDataLoader(g, {'paper': g.nodes('paper')}, sampler, batch_size=args.batch_size)

    model = HGT(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes},
        args.num_hidden, data.num_classes, args.num_heads, g.ntypes, g.canonical_etypes,
        'paper', args.num_layers, args.dropout
    )
    model.load_state_dict(torch.load(args.hgt_model_path, map_location=torch.device('cpu')))
    model = model.to(device)
    pos = torch.zeros(g.num_nodes('paper'), args.num_samples, dtype=torch.long)
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in tqdm(loader):
            blocks = [b.to(device) for b in blocks]
            _ = model(blocks, blocks[0].srcdata['feat'])
            attn = calc_attn(model, blocks).t()  # (N_dst_paper, N_src_paper)
            _, nid = torch.topk(attn, args.num_samples)  # (N_dst_paper, T_pos)
            # nid是blocks[0]中的源paper顶点id，将其转换为原异构图中的paper顶点id
            pos[output_nodes['paper']] = input_nodes['paper'][nid]

    # 构造正样本图
    u = pos.view(1, -1).squeeze(dim=0)  # (N_paper*T_pos,)
    v = torch.repeat_interleave(g.nodes('paper'), args.num_samples)
    pos_g = dgl.graph((u, v), num_nodes=g.num_nodes('paper'))
    dgl.save_graphs(args.save_graph_path, [pos_g])
    print('正样本图已保存到', args.save_graph_path)


def calc_attn(model, blocks):
    # PAP, PPP, PFP
    relations = [('writes_rev', 'writes'), ('cites', 'cites'), ('has_topic', 'has_topic_rev')]
    attn = 0
    for e0, e1 in relations:
        s, _, d = blocks[0].to_canonical_etype(e0)  # s == 'paper', d是中间顶点类型
        a0 = torch.zeros(blocks[0].num_src_nodes(s), blocks[0].num_dst_nodes(d))
        a0[blocks[0].edges(etype=e0)] = model.layers[0].conv.mods[e0].attn.mean(dim=1)
        a1 = torch.zeros(blocks[1].num_src_nodes(d), blocks[1].num_dst_nodes(s))
        a1[blocks[1].edges(etype=e1)] = model.layers[1].conv.mods[e1].attn.mean(dim=1)
        attn += torch.matmul(a0, a1)  # (N_src_paper, N_dst_paper)
    return attn


def parse_args():
    parser = argparse.ArgumentParser(description='使用预训练的HGT计算的注意力权重构造paper顶点的正样本图')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--num-hidden', type=int, default=512, help='隐藏层维数')
    parser.add_argument('--num-heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--num-layers', type=int, default=2, help='层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--batch-size', type=int, default=256, help='批大小')
    parser.add_argument('--num-samples', type=int, default=5, help='每个顶点采样的正样本数量')
    parser.add_argument('node_embed_path', help='预训练顶点嵌入路径')
    parser.add_argument('hgt_model_path', help='预训练的HGT模型保存路径')
    parser.add_argument('save_graph_path', help='正样本图保存路径')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
