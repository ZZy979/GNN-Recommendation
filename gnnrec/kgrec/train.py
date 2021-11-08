import argparse
import json
import math
import warnings

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from sklearn.metrics import ndcg_score
from tqdm import tqdm

from gnnrec.config import DATA_DIR
from gnnrec.hge.rhgnn.model import RHGNN
from gnnrec.hge.utils import set_random_seed, get_device, add_reverse_edges, add_node_feat
from gnnrec.kgrec.data import OAGCSDataset
from gnnrec.kgrec.utils import TripletNodeDataLoader


def load_data(device):
    g = add_reverse_edges(OAGCSDataset()[0]).to(device)
    field_feat = g.nodes['field'].data['feat']

    with open(DATA_DIR / 'rank/author_rank_triplets.txt') as f:
        triplets = torch.tensor([[int(x) for x in line.split()] for line in f], device=device)

    with open(DATA_DIR / 'rank/author_rank_train.json') as f:
        author_rank_train = json.load(f)
    train_fields = list(author_rank_train)
    true_relevance = np.zeros((len(train_fields), g.num_nodes('author')), dtype=np.int32)
    for i, f in enumerate(train_fields):
        for r, a in enumerate(author_rank_train[f]):
            true_relevance[i, a] = math.ceil((100 - r) / 10)
    train_fields = list(map(int, train_fields))

    return g, field_feat, triplets, true_relevance, train_fields


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    g, field_feat, triplets, true_relevance, train_fields = load_data(device)
    add_node_feat(g, 'pretrained', args.node_embed_path)

    sampler = MultiLayerNeighborSampler([args.neighbor_size] * args.num_layers)
    triplet_loader = TripletNodeDataLoader(g, triplets, sampler, device, batch_size=args.batch_size)
    node_loader = NodeDataLoader(g, {'author': g.nodes('author')}, sampler, device=device, batch_size=args.batch_size)

    model = RHGNN(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes},
        args.num_hidden, field_feat.shape[1], args.num_rel_hidden, args.num_rel_hidden,
        args.num_heads, g.ntypes, g.canonical_etypes, 'author', args.num_layers, args.dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(triplet_loader) * args.epochs, eta_min=args.lr / 100
    )
    warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch, output_nodes, blocks in tqdm(triplet_loader):
            batch_logits = model(blocks, blocks[0].srcdata['feat'])
            aid_map = {a: i for i, a in enumerate(output_nodes.tolist())}
            anchor = field_feat[batch[:, 0]]
            positive = batch_logits[[aid_map[a] for a in batch[:, 1].tolist()]]
            negative = batch_logits[[aid_map[a] for a in batch[:, 2].tolist()]]
            loss = F.triplet_margin_loss(anchor, positive, negative)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()
        print('Epoch {:d} | Loss {:.4f}'.format(epoch, sum(losses) / len(losses)))
        torch.save(model.state_dict(), args.model_save_path)
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            print('nDCG@{}={:.4f}'.format(args.k, evaluate(
                model, node_loader, g, field_feat.shape[1], 'author',
                field_feat[train_fields], true_relevance, args.k
            )))
    torch.save(model.state_dict(), args.model_save_path)
    print('模型已保存到', args.model_save_path)

    author_embeds = infer(model, node_loader, g, field_feat.shape[1], 'author')
    torch.save(author_embeds.cpu(), args.author_embed_save_path)
    print('学者嵌入已保存到', args.author_embed_save_path)


@torch.no_grad()
def evaluate(model, loader, g, out_dim, predict_ntype, field_feat, true_relevance, k):
    embeds = infer(model, loader, g, out_dim, predict_ntype)
    scores = torch.mm(field_feat, embeds.t()).detach().cpu().numpy()
    return ndcg_score(true_relevance, scores, k=k, ignore_ties=True)


@torch.no_grad()
def infer(model, loader, g, out_dim, predict_ntype):
    model.eval()
    embeds = torch.zeros((g.num_nodes(predict_ntype), out_dim), device=g.device)
    for _, output_nodes, blocks in tqdm(loader):
        embeds[output_nodes[predict_ntype]] = model(blocks, blocks[0].srcdata['feat'])
    return embeds


def main():
    parser = argparse.ArgumentParser(description='GARec算法 训练GNN模型')
    parser.add_argument('--seed', type=int, default=0, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    # R-HGNN
    parser.add_argument('--num-hidden', type=int, default=64, help='隐藏层维数')
    parser.add_argument('--num-rel-hidden', type=int, default=8, help='关系表示的隐藏层维数')
    parser.add_argument('--num-heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--num-layers', type=int, default=2, help='层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--epochs', type=int, default=200, help='训练epoch数')
    parser.add_argument('--batch-size', type=int, default=1024, help='批大小')
    parser.add_argument('--neighbor-size', type=int, default=10, help='邻居采样数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--eval-every', type=int, default=10, help='每多少个epoch评价一次')
    parser.add_argument('-k', type=int, default=20, help='评价指标只考虑top k的学者')
    parser.add_argument('node_embed_path', help='预训练顶点嵌入路径')
    parser.add_argument('model_save_path', help='模型保存路径')
    parser.add_argument('author_embed_save_path', help='学者嵌入保存路径')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
