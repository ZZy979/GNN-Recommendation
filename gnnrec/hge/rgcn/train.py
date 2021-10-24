import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim

from gnnrec.hge.rgcn.model import RGCN
from gnnrec.hge.utils import set_random_seed, get_device, load_data, calc_metrics, METRICS_STR


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    data, g, features, labels, predict_ntype, train_idx, val_idx, test_idx, evaluator = \
        load_data(args.dataset, device)

    model = RGCN(
        features.shape[1], args.num_hidden, data.num_classes, [predict_ntype],
        {ntype: g.num_nodes(ntype) for ntype in g.ntypes}, g.etypes,
        predict_ntype, args.num_layers, args.dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    features = {predict_ntype: features}
    for epoch in range(args.epochs):
        model.train()
        logits = model(g, features)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(('Epoch {:d} | Loss {:.4f} | ' + METRICS_STR).format(
            epoch, loss.item(), *evaluate(model, g, features, labels, train_idx, val_idx, test_idx)
        ))


@torch.no_grad()
def evaluate(model, g, features, labels, train_idx, val_idx, test_idx):
    model.eval()
    logits = model(g, features)
    return calc_metrics(logits, labels, train_idx, val_idx, test_idx)


def main():
    parser = argparse.ArgumentParser(description='训练R-GCN模型')
    parser.add_argument('--seed', type=int, default=8, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--dataset', choices=['acm', 'dblp', 'ogbn-mag'], default='ogbn-mag', help='数据集')
    parser.add_argument('--num-hidden', type=int, default=32, help='隐藏层维数')
    parser.add_argument('--num-layers', type=int, default=2, help='模型层数')
    parser.add_argument('--dropout', type=float, default=0.8, help='Dropout概率')
    parser.add_argument('--epochs', type=int, default=50, help='训练epoch数')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
