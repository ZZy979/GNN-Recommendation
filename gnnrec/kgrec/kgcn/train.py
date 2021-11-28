import argparse

import dgl
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerNeighborSampler

from gnnrec.hge.utils import set_random_seed, get_device
from gnnrec.kgrec.kgcn.data import RatingKnowledgeGraphDataset
from gnnrec.kgrec.kgcn.dataloader import KGCNEdgeDataLoader
from gnnrec.kgrec.kgcn.model import KGCN
from gnnrec.kgrec.utils import load_rank_data, recall_paper, calc_metrics, METRICS_STR


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    g, author_rank, field_ids, true_relevance = load_rank_data(device)
    field_paper = recall_paper(g.cpu(), field_ids, args.num_recall)
    data = RatingKnowledgeGraphDataset()
    user_item_graph = data.user_item_graph
    knowledge_graph = dgl.sampling.sample_neighbors(
        data.knowledge_graph, data.knowledge_graph.nodes(), args.neighbor_size, replace=True
    )

    sampler = MultiLayerNeighborSampler([args.neighbor_size] * args.num_hops)
    train_loader = KGCNEdgeDataLoader(
        user_item_graph, torch.arange(user_item_graph.num_edges()), sampler, knowledge_graph,
        device=device, batch_size=args.batch_size
    )

    model = KGCN(args.num_hidden, args.neighbor_size, 'sum', args.num_hops, *data.get_num()).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for _, pair_graph, blocks in train_loader:
            scores = model(pair_graph, blocks)
            loss = F.binary_cross_entropy(scores, pair_graph.edata['label'])
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch {:d} | Loss {:.4f}'.format(epoch, sum(losses) / len(losses)))
        print(METRICS_STR.format(*evaluate(
            model, g, knowledge_graph, sampler, field_ids, author_rank, true_relevance, field_paper
        )))


@torch.no_grad()
def evaluate(model, g, kg, sampler, field_ids, author_rank, true_relevance, field_paper):
    model.eval()
    predict_rank = {}
    for i, f in enumerate(field_ids):
        aid = g.in_edges(field_paper[f], etype='writes')[0].unique()
        pair_graph = dgl.heterograph({
            ('user', 'rate', 'item'): (torch.zeros(aid.shape, dtype=torch.long), torch.arange(aid.shape[0]))
        }, device=g.device)
        pair_graph.ndata[dgl.NID] = {'user': torch.tensor([i], device=g.device), 'item': aid}
        blocks = sampler.sample_blocks(kg, aid)
        aid = aid.cpu().numpy()
        scores = model(pair_graph, blocks).cpu()
        predict_rank[f] = (aid, scores)
    return calc_metrics(field_ids, author_rank, true_relevance, predict_rank)


def main():
    parser = argparse.ArgumentParser('训练KGCN模型')
    parser.add_argument('--seed', type=int, default=0, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--num-hidden', type=int, default=16, help='隐藏层维数')
    parser.add_argument('--num-hops', type=int, default=1, help='层数')
    parser.add_argument('--epochs', type=int, default=50, help='训练epoch数')
    parser.add_argument('--batch-size', type=int, default=256, help='批大小')
    parser.add_argument('--neighbor-size', type=int, default=8, help='邻居采样数')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--num-recall', type=int, default=200, help='每个领域召回论文的数量')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
