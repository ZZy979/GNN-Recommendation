import argparse
import json
import math
import random
import warnings

import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from dgl.utils import to_dgl_context
from sklearn.metrics import ndcg_score
from tqdm import tqdm

from gnnrec.config import DATA_DIR
from gnnrec.hge.rhgnn.model import RHGNN
from gnnrec.hge.utils import set_random_seed, get_device, add_reverse_edges, add_node_feat
from gnnrec.kgrec.data import OAGCSDataset
from gnnrec.kgrec.utils import TripletNodeCollator


def load_data(device):
    g = add_reverse_edges(OAGCSDataset()[0]).to(device)

    # {field_id: [author_id]}
    with open(DATA_DIR / 'rank/author_rank_train.json') as f:
        author_rank = {int(k): v for k, v in json.load(f).items()}
    field_ids = list(author_rank)

    # (N_field, N_author) 领域-学者真实相关性得分
    true_relevance = np.zeros((len(field_ids), g.num_nodes('author')), dtype=np.int32)
    for i, f in enumerate(field_ids):
        for r, a in enumerate(author_rank[f]):
            true_relevance[i, a] = math.ceil((100 - r) / 10)

    return g, author_rank, field_ids, true_relevance


def recall_paper(g, field_ids, num_recall):
    """预先计算论文召回

    :param g: DGLGraph 异构图
    :param field_ids: List[int] 目标领域id
    :param num_recall: 每个领域召回的论文数
    :return: Dict[int, List[int]] {field_id: [paper_id]}
    """
    similarity = torch.zeros((len(field_ids), g.num_nodes('paper')), device=g.device)
    sg = dgl.out_subgraph(g['has_field_rev'], {'field': field_ids}, relabel_nodes=True)
    sg.apply_edges(fn.u_dot_v('feat', 'feat', 's'))
    f, p = sg.edges()
    similarity[f, sg.nodes['paper'].data[dgl.NID][p]] = sg.edata['s'].squeeze(dim=1)
    _, pid = similarity.topk(num_recall, dim=1)
    return {f: pid[i].tolist() for i, f in enumerate(field_ids)}


def sample_triplets(field_id, author_ids, args):
    """根据领域学者排名采样三元组(t, ap, an)，表示对于领域t，学者ap的排名在an之前

    :param field_id: int 领域id
    :param author_ids: List[int] 学者排名
    :param args: 命令行参数
    :return: tensor(N, 3) 采样的三元组
    """
    n = len(author_ids)
    easy_margin, hard_margin = int(n * args.easy_margin), int(n * args.hard_margin)
    num_triplets = min(args.max_triplets, 2 * n - easy_margin - hard_margin)
    num_hard = int(num_triplets * args.hard_ratio)
    num_easy = num_triplets - num_hard
    easy_triplets = [
        (field_id, author_ids[i], author_ids[i + easy_margin])
        for i in random.sample(range(n - easy_margin), num_easy)
    ]
    hard_triplets = [
        (field_id, author_ids[i], author_ids[i + hard_margin])
        for i in random.sample(range(n - hard_margin), num_hard)
    ]
    return torch.tensor(easy_triplets + hard_triplets)


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    g, author_rank, field_ids, true_relevance = load_data(device)
    g.nodes['paper'].data['feat'] = torch.load(DATA_DIR / 'rank/paper_embed.pkl', map_location=device)
    out_dim = g.nodes['field'].data['feat'].shape[1]
    add_node_feat(g, 'pretrained', args.node_embed_path)
    field_paper = recall_paper(g, field_ids, args.num_recall)  # {field_id: [paper_id]}

    sampler = MultiLayerNeighborSampler([args.neighbor_size] * args.num_layers)
    sampler.set_output_context(to_dgl_context(device))
    triplet_collator = TripletNodeCollator(g, sampler)

    model = RHGNN(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes},
        args.num_hidden, out_dim, args.num_rel_hidden, args.num_rel_hidden,
        args.num_heads, g.ntypes, g.canonical_etypes, 'author', args.num_layers, args.dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(field_ids) * args.epochs, eta_min=args.lr / 100
    )
    warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for f in tqdm(field_ids):
            triplets = sample_triplets(f, author_rank[f], args).to(device)
            aid, blocks = triplet_collator.collate(triplets)
            author_embeds = model(blocks, blocks[0].srcdata['feat'])
            aid_map = {a: i for i, a in enumerate(aid.tolist())}
            anchor = g.nodes['field'].data['feat'][triplets[:, 0]]
            positive = author_embeds[[aid_map[a] for a in triplets[:, 1].tolist()]]
            negative = author_embeds[[aid_map[a] for a in triplets[:, 2].tolist()]]
            loss = F.triplet_margin_loss(anchor, positive, negative)

            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()
        print('Epoch {:d} | Loss {:.4f}'.format(epoch, sum(losses) / len(losses)))
        torch.save(model.state_dict(), args.model_save_path)
        print('nDCG@{}={:.4f}'.format(args.k, evaluate(
            model, g, out_dim, sampler, args.batch_size, device, field_ids,
            field_paper, true_relevance, args.k
        )))
    torch.save(model.state_dict(), args.model_save_path)
    print('模型已保存到', args.model_save_path)

    embeds = infer(model, g, 'author', out_dim, sampler, args.batch_size, device)
    author_embed_save_path = DATA_DIR / 'rank/author_embed.pkl'
    torch.save(embeds.cpu(), author_embed_save_path)
    print('学者嵌入已保存到', author_embed_save_path)


@torch.no_grad()
def evaluate(model, g, out_dim, sampler, batch_size, device, field_ids, field_paper, true_relevance, k):
    model.eval()
    author_embeds = infer(model, g, 'author', out_dim, sampler, batch_size, device)  # (N_author, d)
    author_scores = torch.zeros(len(field_ids), author_embeds.shape[0])  # (N_field, N_author)
    for i, f in enumerate(field_ids):
        aid = g.in_edges(field_paper[f], etype='writes')[0].numpy()
        author_scores[i, aid] = torch.matmul(author_embeds[aid], g.nodes['field'].data['feat'][f])
    return ndcg_score(true_relevance, author_scores, k=k, ignore_ties=True)


@torch.no_grad()
def infer(model, g, ntype, out_dim, sampler, batch_size, device):
    model.eval()
    embeds = torch.zeros((g.num_nodes(ntype), out_dim), device=device)
    loader = NodeDataLoader(g, {ntype: g.nodes(ntype)}, sampler, device=device, batch_size=batch_size)
    for _, output_nodes, blocks in tqdm(loader):
        embeds[output_nodes[ntype]] = model(blocks, blocks[0].srcdata['feat'])
    return embeds


def main():
    parser = argparse.ArgumentParser(description='GARec算法 训练学者排名GNN模型')
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
    # 采样三元组
    parser.add_argument('--max-triplets', type=int, default=100, help='每个领域采样三元组最大数量')
    parser.add_argument('--easy-margin', type=float, default=0.2, help='简单样本间隔（百分比）')
    parser.add_argument('--hard-margin', type=float, default=0.05, help='困难样本间隔（百分比）')
    parser.add_argument('--hard-ratio', type=float, default=0.5, help='困难样本比例')
    # 评价
    parser.add_argument('--num-recall', type=int, default=1000, help='评价时每个领域召回论文的数量')
    parser.add_argument('-k', type=int, default=100, help='评价指标只考虑top k的学者')
    parser.add_argument('node_embed_path', help='预训练顶点嵌入路径')
    parser.add_argument('model_save_path', help='模型保存路径')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
