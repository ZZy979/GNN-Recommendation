import argparse
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim

from gnnrec.hge.hgconv.model import HGConvFull
from gnnrec.hge.utils import set_random_seed, get_device, load_data, add_node_feat, evaluate_full


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    data, g, _, labels, predict_ntype, train_idx, val_idx, test_idx, _ = \
        load_data(args.dataset, device)
    add_node_feat(g, 'one-hot')

    model = HGConvFull(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes},
        args.num_hidden, data.num_classes, args.num_heads, g.ntypes, g.canonical_etypes,
        predict_ntype, args.num_layers, args.dropout, args.residual
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
    for epoch in range(args.epochs):
        model.train()
        logits = model(g, g.ndata['feat'])
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        print('Epoch {:d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f} | Test Acc {:.4f}'.format(
            epoch, loss.item(), *evaluate_full(model, g, labels, train_idx, val_idx, test_idx)
        ))


def main():
    parser = argparse.ArgumentParser(description='训练HGConv模型(full-batch)')
    parser.add_argument('--seed', type=int, default=8, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--dataset', choices=['acm', 'dblp'], default='acm', help='数据集')
    parser.add_argument('--num-hidden', type=int, default=32, help='隐藏层维数')
    parser.add_argument('--num-heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--num-layers', type=int, default=2, help='层数')
    parser.add_argument('--no-residual', action='store_false', help='不使用残差连接', dest='residual')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--epochs', type=int, default=10, help='训练epoch数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='权重衰减')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
