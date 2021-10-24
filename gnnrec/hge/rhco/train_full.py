import argparse

import dgl
import torch
import torch.nn.functional as F
import torch.optim as optim

from gnnrec.hge.rhco.model import RHCOFull
from gnnrec.hge.rhco.smooth import smooth
from gnnrec.hge.utils import set_random_seed, get_device, load_data, add_node_feat, calc_metrics, \
    METRICS_STR


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    data, g, features, labels, predict_ntype, train_idx, val_idx, test_idx, _ = \
        load_data(args.dataset, device)
    add_node_feat(g, 'one-hot')

    pos_v, pos_u = data.pos
    pos_g = dgl.graph((pos_u, pos_v), device=device)
    pos_g.ndata['feat'] = features
    pos = torch.zeros((g.num_nodes(predict_ntype), g.num_nodes(predict_ntype)), dtype=torch.int, device=device)
    pos[data.pos] = 1

    model = RHCOFull(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes},
        args.num_hidden, data.num_classes, args.num_rel_hidden, args.num_heads,
        g.ntypes, g.canonical_etypes, predict_ntype, args.num_layers, args.dropout,
        args.tau, args.lambda_
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr / 100
    )
    alpha = args.contrast_weight
    for epoch in range(args.epochs):
        model.train()
        contrast_loss, logits = model(g, g.ndata['feat'], pos_g, pos_g.ndata['feat'], pos)
        clf_loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss = alpha * contrast_loss + (1 - alpha) * clf_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()
        print(('Epoch {:d} | Loss {:.4f} | ' + METRICS_STR).format(
            epoch, loss.item(), *evaluate(model, g, pos_g, pos, labels, train_idx, val_idx, test_idx)
        ))

    model.eval()
    _, base_pred = model(g, g.ndata['feat'], pos_g, pos_g.ndata['feat'], pos)
    mask = torch.cat([train_idx, val_idx])
    logits = smooth(base_pred, pos_g, labels, mask, args)
    _, _, test_acc, _, _, test_f1 = calc_metrics(logits, labels, train_idx, val_idx, test_idx)
    print('After smoothing: Test Acc {:.4f} | Test Macro-F1 {:.4f}'.format(test_acc, test_f1))


@torch.no_grad()
def evaluate(model, g, pos_g, pos, labels, train_idx, val_idx, test_idx):
    model.eval()
    _, logits = model(g, g.ndata['feat'], pos_g, pos_g.ndata['feat'], pos)
    return calc_metrics(logits, labels, train_idx, val_idx, test_idx)


def main():
    parser = argparse.ArgumentParser(description='训练RHCO模型(full-batch)')
    parser.add_argument('--seed', type=int, default=0, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--dataset', choices=['acm', 'dblp'], default='acm', help='数据集')
    parser.add_argument('--num-hidden', type=int, default=64, help='隐藏层维数')
    parser.add_argument('--num-rel-hidden', type=int, default=8, help='关系表示的隐藏层维数')
    parser.add_argument('--num-heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--num-layers', type=int, default=2, help='层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--tau', type=float, default=0.8, help='温度参数')
    parser.add_argument('--lambda', type=float, default=0.5, dest='lambda_', help='对比损失的平衡系数')
    parser.add_argument('--epochs', type=int, default=5, help='训练epoch数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--contrast-weight', type=float, default=0.5, help='对比损失权重')
    parser.add_argument('--num-smooth-layers', type=int, default=50, help='Smooth步骤传播层数')
    parser.add_argument('--smooth-alpha', type=float, default=0.5, help='Smooth步骤α值')
    parser.add_argument(
        '--smooth-norm', choices=['left', 'right', 'both'], default='right',
        help='Smooth步骤归一化方式'
    )
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
