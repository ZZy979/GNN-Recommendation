import argparse

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import NodeDataLoader
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from gnnrec.hge.heco.model import HeCo
from gnnrec.hge.heco.sampler import PositiveSampler
from gnnrec.hge.utils import set_random_seed, get_device, load_data, add_node_feat, accuracy, \
    calc_metrics, METRICS_STR


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    data, g, _, labels, predict_ntype, train_idx, val_idx, test_idx, evaluator = \
        load_data(args.dataset, device)
    add_node_feat(g, 'pretrained', args.node_embed_path)
    relations = [r for r in g.canonical_etypes if r[2] == predict_ntype]

    pos_g = dgl.load_graphs(args.pos_graph_path)[0][0].to(device)
    pos_g.ndata['feat'] = g.nodes[predict_ntype].data['feat']
    pos = pos_g.in_edges(pos_g.nodes())[0].view(pos_g.num_nodes(), -1)  # (N, T_pos) 每个目标顶点的正样本id

    id_loader = DataLoader(train_idx, batch_size=args.batch_size)
    loader = NodeDataLoader(
        g, {predict_ntype: train_idx}, PositiveSampler([None], pos),
        device=device, batch_size=args.batch_size
    )
    pos_loader = NodeDataLoader(
        pos_g, train_idx, PositiveSampler([None], pos), device=device, batch_size=args.batch_size
    )

    model = HeCo(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes},
        args.num_hidden, args.feat_drop, args.attn_drop, relations, args.tau, args.lambda_
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for (batch, (_, _, blocks), (_, _, pos_blocks)) in tqdm(zip(id_loader, loader, pos_loader)):
            block = blocks[0]
            pos_block = pos_blocks[0]
            batch_pos = torch.zeros(pos_block.num_dst_nodes(), batch.shape[0], dtype=torch.int, device=device)
            batch_pos[pos_block.in_edges(torch.arange(batch.shape[0], device=device))] = 1
            loss, _ = model(block, block.srcdata['feat'], pos_block, pos_block.srcdata['feat'], batch_pos.t())
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        print('Epoch {:d} | Loss {:.4f}'.format(epoch, sum(losses) / len(losses)))
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            print(METRICS_STR.format(*evaluate(
                model, pos_g, pos_g.ndata['feat'], device, labels, data.num_classes,
                train_idx, val_idx, test_idx
            )))


def evaluate(model, pos_g, feat, device, labels, num_classes, train_idx, val_idx, test_idx):
    model.eval()
    embeds = model.get_embeds(pos_g, feat)

    clf = nn.Linear(embeds.shape[1], num_classes).to(device)
    optimizer = optim.Adam(clf.parameters(), lr=0.05)
    best_acc, best_logits = 0, None
    for epoch in trange(200):
        clf.train()
        logits = clf(embeds)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            clf.eval()
            logits = clf(embeds)
            predict = logits.argmax(dim=1)
            if accuracy(predict[val_idx], labels[val_idx]) > best_acc:
                best_logits = logits
    return calc_metrics(best_logits, labels, train_idx, val_idx, test_idx)


def main():
    parser = argparse.ArgumentParser(description='训练HeCo模型')
    parser.add_argument('--seed', type=int, default=0, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--dataset', choices=['ogbn-mag'], default='ogbn-mag', help='数据集')
    parser.add_argument('--num-hidden', type=int, default=64, help='隐藏层维数')
    parser.add_argument('--feat-drop', type=float, default=0.3, help='特征dropout')
    parser.add_argument('--attn-drop', type=float, default=0.5, help='注意力dropout')
    parser.add_argument('--tau', type=float, default=0.8, help='温度参数')
    parser.add_argument('--lambda', type=float, default=0.5, dest='lambda_', help='对比损失的平衡系数')
    parser.add_argument('--epochs', type=int, default=200, help='训练epoch数')
    parser.add_argument('--batch-size', type=int, default=1024, help='批大小')
    parser.add_argument('--lr', type=float, default=0.0008, help='学习率')
    parser.add_argument('--eval-every', type=int, default=10, help='每多少个epoch计算一次准确率')
    parser.add_argument('node_embed_path', help='预训练顶点嵌入路径')
    parser.add_argument('pos_graph_path', help='正样本图路径')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
