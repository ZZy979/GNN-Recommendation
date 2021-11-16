import argparse
import json
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader, EdgeDataLoader
from dgl.dataloading.negative_sampler import Uniform
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from gnnrec.config import DATA_DIR
from gnnrec.hge.utils import set_random_seed, get_device, add_reverse_edges, add_node_feat
from gnnrec.kgrec.data import OAGCSDataset
from gnnrec.kgrec.model import RecallModel


def load_data():
    g = add_reverse_edges(OAGCSDataset()[0])

    # {field_id: [author_id]}
    with open(DATA_DIR / 'rank/author_rank_train.json') as f:
        author_rank = {int(k): v for k, v in json.load(f).items()}

    train_edges_src, train_edges_dst = [], []
    for f in author_rank:
        field_pid = set(g.in_edges(f, etype='has_field')[0].tolist())
        author_pid = set(g.out_edges(author_rank[f], etype='writes')[1].tolist())
        pid = list(field_pid & author_pid)
        train_edges_src.extend(pid)
        train_edges_dst.extend([f] * len(pid))
    train_eids = g.edge_ids(train_edges_src, train_edges_dst, etype='has_field')
    return g, train_eids


def calc_loss(pos_score, neg_score):
    label = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
    return F.binary_cross_entropy_with_logits(torch.cat([pos_score, neg_score]), label)


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    g, train_eids = load_data()
    out_dim = g.nodes['paper'].data['feat'].shape[1]
    add_node_feat(g, 'pretrained', args.node_embed_path)

    sampler = MultiLayerNeighborSampler([args.neighbor_size] * args.num_layers)
    loader = EdgeDataLoader(
        g, {'has_field': train_eids}, sampler, device=device,
        negative_sampler=Uniform(args.num_neg_samples), batch_size=args.batch_size
    )

    model = RecallModel(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes},
        args.num_hidden, out_dim, args.num_rel_hidden, args.num_rel_hidden,
        args.num_heads, g.ntypes, g.canonical_etypes, None, args.num_layers, args.dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(loader) * args.epochs, eta_min=args.lr / 100
    )
    warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for input_nodes, pos_g, neg_g, blocks in tqdm(loader):
            pos_score, neg_score = model(pos_g, neg_g, blocks, blocks[0].srcdata['feat'], 'has_field')
            loss = calc_loss(pos_score, neg_score)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()
        print('Epoch {:d} | Loss {:.4f}'.format(epoch, sum(losses) / len(losses)))
        torch.save(model.state_dict(), args.model_save_path)
        print('Precision {:.4f} | Recall {:.4f} | F1 {:.4f}'.format(*evaluate(model, loader)))
    torch.save(model.state_dict(), args.model_save_path)
    print('模型已保存到', args.model_save_path)

    embeds = infer(model.rhgnn, g, 'paper', out_dim, sampler, args.batch_size, device)
    paper_embed_save_path = DATA_DIR / 'rank/paper_embed.pkl'
    torch.save(embeds.cpu(), paper_embed_save_path)
    print('论文嵌入已保存到', paper_embed_save_path)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    pos_scores, neg_scores = [], []
    for input_nodes, pos_g, neg_g, blocks in loader:
        pos_score, neg_score = model(pos_g, neg_g, blocks, blocks[0].srcdata['feat'], 'has_field')
        pos_scores.append(pos_score.cpu().numpy())
        neg_scores.append(neg_score.cpu().numpy())
    pos_scores, neg_scores = np.concatenate(pos_scores), np.concatenate(neg_scores)
    y_pred = np.concatenate([pos_scores, neg_scores]) > 0.5
    y_true = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    return precision_recall_fscore_support(y_true, y_pred, average='binary')[:3]


@torch.no_grad()
def infer(model, g, ntype, out_dim, sampler, batch_size, device):
    model.eval()
    embeds = torch.zeros((g.num_nodes(ntype), out_dim), device=device)
    loader = NodeDataLoader(g, {ntype: g.nodes(ntype)}, sampler, device=device, batch_size=batch_size)
    for _, output_nodes, blocks in tqdm(loader):
        embeds[output_nodes[ntype]] = model(blocks, blocks[0].srcdata['feat'], True)[ntype]
    return embeds


def main():
    parser = argparse.ArgumentParser(description='GARec算法 训练论文召回GNN模型')
    parser.add_argument('--seed', type=int, default=0, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    # R-HGNN
    parser.add_argument('--num-hidden', type=int, default=64, help='隐藏层维数')
    parser.add_argument('--num-rel-hidden', type=int, default=8, help='关系表示的隐藏层维数')
    parser.add_argument('--num-heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--num-layers', type=int, default=2, help='层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--epochs', type=int, default=50, help='训练epoch数')
    parser.add_argument('--batch-size', type=int, default=1024, help='批大小')
    parser.add_argument('--neighbor-size', type=int, default=10, help='邻居采样数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--num-neg-samples', type=int, default=5, help='每条边采样负样本边的数量')
    parser.add_argument('node_embed_path', help='预训练顶点嵌入路径')
    parser.add_argument('model_save_path', help='模型保存路径')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
