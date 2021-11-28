import argparse
import random
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from dgl.utils import to_dgl_context
from tqdm import tqdm

from gnnrec.config import DATA_DIR
from gnnrec.hge.rhgnn.model import RHGNN
from gnnrec.hge.utils import set_random_seed, get_device, add_node_feat
from gnnrec.kgrec.utils import load_rank_data, recall_paper, TripletNodeCollator, calc_metrics, \
    METRICS_STR


def sample_triplets(field_id, true_author_ids, false_author_ids, num_triplets):
    """根据领域学者排名采样三元组(t, ap, an)，表示对于领域t，学者ap的排名在an之前

    :param field_id: int 领域id
    :param true_author_ids: List[int] top-n学者id，真实排名
    :param false_author_ids: List[int] 不属于top-n的学者id
    :param num_triplets: int 采样的三元组数量
    :return: tensor(N, 3) 采样的三元组
    """
    n = len(true_author_ids)
    easy_margin, hard_margin = int(n * 0.2), int(n * 0.05)
    easy_triplets = [
        (field_id, true_author_ids[i], true_author_ids[i + easy_margin])
        for i in range(n - easy_margin)
    ]  # 简单样本
    hard_triplets = [
        (field_id, true_author_ids[i], true_author_ids[i + hard_margin])
        for i in range(n - hard_margin)
    ]  # 困难样本
    m = num_triplets - len(easy_triplets) - len(hard_triplets)
    true_false_triplets = [
        (field_id, t, f)
        for t, f in zip(random.choices(true_author_ids, k=m), random.choices(false_author_ids, k=m))
    ]  # 真-假样本
    return torch.tensor(easy_triplets + hard_triplets + true_false_triplets)


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    g, author_rank, field_ids, true_relevance = load_rank_data(device)
    out_dim = g.nodes['field'].data['feat'].shape[1]
    add_node_feat(g, 'pretrained', args.node_embed_path, use_raw_id=True)
    field_paper = recall_paper(g, field_ids, args.num_recall)  # {field_id: [paper_id]}

    sampler = MultiLayerNeighborSampler([args.neighbor_size] * args.num_layers)
    sampler.set_output_context(to_dgl_context(device))
    triplet_collator = TripletNodeCollator(g, sampler)

    model = RHGNN(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes},
        args.num_hidden, out_dim, args.num_rel_hidden, args.num_rel_hidden,
        args.num_heads, g.ntypes, g.canonical_etypes, 'author', args.num_layers, args.dropout
    ).to(device)
    if args.load_path:
        model.load_state_dict(torch.load(args.load_path, map_location=device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(field_ids) * args.epochs, eta_min=args.lr / 100
    )
    warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for f in tqdm(field_ids):
            false_author_ids = list(set(g.in_edges(field_paper[f], etype='writes')[0].tolist()) - set(author_rank[f]))
            triplets = sample_triplets(f, author_rank[f], false_author_ids, args.num_triplets).to(device)
            aid, blocks = triplet_collator.collate(triplets)
            author_embeds = model(blocks, blocks[0].srcdata['feat'])
            author_embeds = author_embeds / author_embeds.norm(dim=1, keepdim=True)
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
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            print(METRICS_STR.format(*evaluate(
                model, g, out_dim, sampler, args.batch_size, device, field_ids, field_paper,
                author_rank, true_relevance
            )))
    torch.save(model.state_dict(), args.model_save_path)
    print('模型已保存到', args.model_save_path)

    embeds = infer(model, g, 'author', out_dim, sampler, args.batch_size, device)
    author_embed_save_path = DATA_DIR / 'rank/author_embed.pkl'
    torch.save(embeds.cpu(), author_embed_save_path)
    print('学者嵌入已保存到', author_embed_save_path)


@torch.no_grad()
def evaluate(model, g, out_dim, sampler, batch_size, device, field_ids, field_paper, author_rank, true_relevance):
    model.eval()
    predict_rank = {}
    field_feat = g.nodes['field'].data['feat']
    author_embeds = infer(model, g, 'author', out_dim, sampler, batch_size, device)  # (N_author, d)
    for i, f in enumerate(field_ids):
        aid = g.in_edges(field_paper[f], etype='writes')[0].cpu().unique()
        similarity = torch.matmul(author_embeds[aid], field_feat[f]).cpu()
        predict_rank[f] = (aid, similarity)
    return calc_metrics(field_ids, author_rank, true_relevance, predict_rank)


@torch.no_grad()
def infer(model, g, ntype, out_dim, sampler, batch_size, device):
    model.eval()
    embeds = torch.zeros((g.num_nodes(ntype), out_dim), device=device)
    loader = NodeDataLoader(g, {ntype: g.nodes(ntype)}, sampler, device=device, batch_size=batch_size)
    for _, output_nodes, blocks in tqdm(loader):
        embeds[output_nodes[ntype]] = model(blocks, blocks[0].srcdata['feat'])
    embeds = embeds / embeds.norm(dim=1, keepdim=True)
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
    parser.add_argument('--epochs', type=int, default=100, help='训练epoch数')
    parser.add_argument('--batch-size', type=int, default=1024, help='批大小')
    parser.add_argument('--neighbor-size', type=int, default=10, help='邻居采样数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--load-path', help='模型加载路径，用于继续训练')
    # 采样三元组
    parser.add_argument('--num-triplets', type=int, default=1000, help='每个领域采样三元组数量')
    # 评价
    parser.add_argument('--eval-every', type=int, default=10, help='每多少个epoch评价一次')
    parser.add_argument('--num-recall', type=int, default=200, help='每个领域召回论文的数量')
    parser.add_argument('node_embed_path', help='预训练顶点嵌入路径')
    parser.add_argument('model_save_path', help='模型保存路径')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
