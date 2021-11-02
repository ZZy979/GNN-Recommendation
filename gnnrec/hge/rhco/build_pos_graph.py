import argparse
import random
from collections import defaultdict

import dgl
import torch
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from tqdm import tqdm

from gnnrec.hge.hgt.model import HGT
from gnnrec.hge.utils import set_random_seed, get_device, load_data, add_node_feat


def main():
    args = parse_args()
    print(args)
    set_random_seed(args.seed)
    device = get_device(args.device)

    data, g, _, labels, predict_ntype, train_idx, val_idx, test_idx, _ = load_data(args.dataset)
    g = g.to(device)
    labels = labels.tolist()
    train_idx = torch.cat([train_idx, val_idx])
    add_node_feat(g, 'pretrained', args.node_embed_path)

    label_neigh = sample_label_neighbors(labels, args.num_samples)  # (N, T_pos)
    # List[tensor(N, T_pos)] HGT计算出的注意力权重，M条元路径+一个总体
    attn_pos = calc_attn_pos(g, data.num_classes, predict_ntype, args.num_samples, device, args)

    # 元路径对应的正样本图
    v = torch.repeat_interleave(g.nodes(predict_ntype), args.num_samples).cpu()
    pos_graphs = []
    for p in attn_pos[:-1]:
        u = p.view(1, -1).squeeze(dim=0)  # (N*T_pos,)
        pos_graphs.append(dgl.graph((u, v)))

    # 整体正样本图
    pos = attn_pos[-1]
    if args.use_label:
        pos[train_idx] = label_neigh[train_idx]
        # pos[test_idx, 0] = label_neigh[test_idx, 0]
    u = pos.view(1, -1).squeeze(dim=0)
    pos_graphs.append(dgl.graph((u, v)))

    dgl.save_graphs(args.save_graph_path, pos_graphs)
    print('正样本图已保存到', args.save_graph_path)


def calc_attn_pos(g, num_classes, predict_ntype, num_samples, device, args):
    """使用预训练的HGT模型计算的注意力权重选择目标顶点的正样本。"""
    # 第1层只保留AB边，第2层只保留BA边，其中A是目标顶点类型，B是中间顶点类型
    num_neighbors = [{}, {}]
    # 形如ABA的元路径，其中A是目标顶点类型
    metapaths = []
    rev_etype = {
        e: next(re for rs, re, rd in g.canonical_etypes if rs == d and rd == s and re != e)
        for s, e, d in g.canonical_etypes
    }
    for s, e, d in g.canonical_etypes:
        if d == predict_ntype:
            re = rev_etype[e]
            num_neighbors[0][re] = num_neighbors[1][e] = 10
            metapaths.append((re, e))
    for i in range(len(num_neighbors)):
        d = dict.fromkeys(g.etypes, 0)
        d.update(num_neighbors[i])
        num_neighbors[i] = d
    sampler = MultiLayerNeighborSampler(num_neighbors)
    loader = NodeDataLoader(
        g, {predict_ntype: g.nodes(predict_ntype)}, sampler,
        device=device, batch_size=args.batch_size
    )

    model = HGT(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes},
        args.num_hidden, num_classes, args.num_heads, g.ntypes, g.canonical_etypes,
        predict_ntype, 2, args.dropout
    ).to(device)
    model.load_state_dict(torch.load(args.hgt_model_path, map_location=device))

    # 每条元路径ABA对应一个正样本图G_ABA，加一个总体正样本图G_pos
    pos = [
        torch.zeros(g.num_nodes(predict_ntype), num_samples, dtype=torch.long, device=device)
        for _ in range(len(metapaths) + 1)
    ]
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in tqdm(loader):
            _ = model(blocks, blocks[0].srcdata['feat'])
            # List[tensor(N_src, N_dst)]
            attn = [calc_attn(mp, model, blocks, device).t() for mp in metapaths]
            for i in range(len(attn)):
                _, nid = torch.topk(attn[i], num_samples)  # (N_dst, T_pos)
                # nid是blocks[0]中的源顶点id，将其转换为原异构图中的顶点id
                pos[i][output_nodes[predict_ntype]] = input_nodes[predict_ntype][nid]
            _, nid = torch.topk(sum(attn), num_samples)
            pos[-1][output_nodes[predict_ntype]] = input_nodes[predict_ntype][nid]
    return [p.cpu() for p in pos]


def calc_attn(metapath, model, blocks, device):
    """计算通过指定元路径与目标顶点连接的同类型顶点的注意力权重。"""
    re, e = metapath
    s, _, d = blocks[0].to_canonical_etype(re)  # s是目标顶点类型, d是中间顶点类型
    a0 = torch.zeros(blocks[0].num_src_nodes(s), blocks[0].num_dst_nodes(d), device=device)
    a0[blocks[0].edges(etype=re)] = model.layers[0].conv.mods[re].attn.mean(dim=1)
    a1 = torch.zeros(blocks[1].num_src_nodes(d), blocks[1].num_dst_nodes(s), device=device)
    a1[blocks[1].edges(etype=e)] = model.layers[1].conv.mods[e].attn.mean(dim=1)
    return torch.matmul(a0, a1)  # (N_src, N_dst)


def sample_label_neighbors(labels, num_samples):
    """为每个顶点采样相同标签的邻居。"""
    label2id = defaultdict(list)
    for i, y in enumerate(labels):
        label2id[y].append(i)
    return torch.tensor([random.sample(label2id[y], num_samples) for y in labels])


def parse_args():
    parser = argparse.ArgumentParser(description='使用预训练的HGT计算的注意力权重构造正样本图')
    parser.add_argument('--seed', type=int, default=0, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--dataset', choices=['ogbn-mag'], default='ogbn-mag', help='数据集')
    parser.add_argument('--num-hidden', type=int, default=512, help='隐藏层维数')
    parser.add_argument('--num-heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--batch-size', type=int, default=256, help='批大小')
    parser.add_argument('--num-samples', type=int, default=5, help='每个顶点采样的正样本数量')
    parser.add_argument('--use-label', action='store_true', help='训练集使用真实标签')
    parser.add_argument('node_embed_path', help='预训练顶点嵌入路径')
    parser.add_argument('hgt_model_path', help='预训练的HGT模型保存路径')
    parser.add_argument('save_graph_path', help='正样本图保存路径')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()
