from collections import Counter

import dgl
import dgl.function as fn
import torch
from gensim.models import Word2Vec
from tqdm import trange

from gnnrec.config import DATA_DIR
from gnnrec.hge.utils import load_ogbn_mag


def load_data(node_embed_path, neighbor_path, walks_per_node):
    g = load_ogbn_mag(DATA_DIR, True)[0]
    print('正在加载预训练的顶点嵌入...')
    load_pretrained_node_embed(g, node_embed_path)
    print('正在传播输入特征...')
    feats = propagate_feature(g)
    print('正在构造邻居图...')
    ng = construct_neighbor_graph(
        g, neighbor_path, walks_per_node, {'author': 10, 'paper': 10, 'field_of_study': 3}
    )
    return ng, feats


def load_pretrained_node_embed(g, node_embed_path):
    model = Word2Vec.load(node_embed_path)
    for ntype in ('author', 'paper', 'field_of_study'):
        g.nodes[ntype].data['net_embed'] = torch.from_numpy(
            model.wv[[f'{ntype}_{i}' for i in range(g.num_nodes(ntype))]]
        )


def propagate_feature(g):
    with g.local_scope():
        g.multi_update_all({
            'writes': (fn.copy_u('net_embed', 'm'), fn.mean('m', 'a_net_embed')),
            'has_topic_rev': (fn.copy_u('net_embed', 'm'), fn.mean('m', 'f_net_embed'))
        }, 'sum')
        paper_feats = torch.stack([
            g.nodes['paper'].data[k] for k in
            ('feat', 'net_embed', 'a_net_embed', 'f_net_embed')
        ], dim=1)  # (N_p, 4, d)

        ap = find_neighbors(g, 'writes_rev', 3).view(1, -1)  # (1, 3N_a)
        ap_feat = g.nodes['paper'].data['feat'][ap] \
            .view(g.num_nodes('author'), 3, -1)  # (N_a, 3, d)
        author_feats = torch.cat([
            g.nodes['author'].data['net_embed'].unsqueeze(dim=1), ap_feat
        ], dim=1)  # (N_a, 4, d)

        fp = find_neighbors(g, 'has_topic', 5).view(1, -1)  # (1, 5N_f)
        fp_feat = g.nodes['paper'].data['feat'][fp] \
            .view(g.num_nodes('field_of_study'), 5, -1)  # (N_f, 5, d)
        fos_feats = torch.cat([
            g.nodes['field_of_study'].data['net_embed'].unsqueeze(dim=1), fp_feat
        ], dim=1)  # (N_f, 6, d)

        return {'author': author_feats, 'paper': paper_feats, 'field_of_study': fos_feats}


def find_neighbors(g, etype, n):
    num_nodes = g.num_nodes(g.to_canonical_etype(etype)[2])
    u, v = g.in_edges(torch.arange(num_nodes), etype=etype)
    neighbors = [[] for _ in range(num_nodes)]
    for i in range(len(v)):
        neighbors[v[i].item()].append(u[i].item())
    for v in range(num_nodes):
        if len(neighbors[v]) < n:
            neighbors[v] += [neighbors[v][-1]] * (n - len(neighbors[v]))
        elif len(neighbors[v]) > n:
            neighbors[v] = neighbors[v][:n]
    return torch.tensor(neighbors)  # (N_dst, n)


def construct_neighbor_graph(g, neighbor_path, walks_per_node, neighbor_size):
    edges = {}
    with open(neighbor_path) as f:
        for dtype in ('author', 'paper', 'field_of_study'):
            src = {stype: [] for stype in neighbor_size}
            dst = {stype: [] for stype in neighbor_size}
            for v in trange(g.num_nodes(dtype)):
                counts = {stype: Counter() for stype in neighbor_size}
                for _ in range(walks_per_node):
                    line = f.readline()
                    center, neighbors = line.strip().split(' ', 1)
                    neighbors = neighbors.split(' ')
                    for n in neighbors:
                        stype, u = parse_node_name(n)
                        if stype in neighbor_size:
                            counts[stype][u] += 1
                for stype, c in counts.items():
                    assert len(c) >= neighbor_size[stype], \
                        f'{dtype} {v} has fewer than {neighbor_size[stype]} {stype} neighbors'
                    for u, _ in c.most_common(neighbor_size[stype]):
                        src[stype].append(u)
                        dst[stype].append(v)
            for stype in neighbor_size:
                edges[(stype, f'{stype}-{dtype}', dtype)] = (src[stype], dst[stype])
    return dgl.heterograph(edges, {ntype: g.num_nodes(ntype) for ntype in neighbor_size})


def parse_node_name(node):
    ntype, nid = node.rsplit('_', 1)
    return ntype, int(nid)


def construct_neg_graph(g, neg_sampler):
    return dgl.heterograph(
        neg_sampler(g, {etype: torch.arange(g.num_edges(etype)) for etype in g.etypes}),
        {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
    )
