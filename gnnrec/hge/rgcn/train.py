import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim

from gnnrec.hge.rgcn.model import RGCN
from gnnrec.hge.utils import set_random_seed, get_device, load_data, accuracy


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    g, features, labels, num_classes, predict_ntype, train_idx, val_idx, test_idx, evaluator = \
        load_data(args.dataset, device)

    model = RGCN(
        features.shape[1], args.num_hidden, num_classes, [predict_ntype],
        {ntype: g.num_nodes(ntype) for ntype in g.ntypes}, g.etypes,
        predict_ntype, args.num_layers, args.dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        logits = model(g, {predict_ntype: features})
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch {:d} | Train Acc {:.4f} | Val Acc {:.4f} | Test Acc {:.4f}'.format(
            epoch, *evaluate(model, g, features, labels, predict_ntype, train_idx, val_idx, test_idx, evaluator)
        ))


@torch.no_grad()
def evaluate(model, g, features, labels, predict_ntype, train_idx, val_idx, test_idx, evaluator):
    model.eval()
    logits = model(g, {predict_ntype: features})
    train_acc = accuracy(logits[train_idx], labels[train_idx], evaluator)
    val_acc = accuracy(logits[val_idx], labels[val_idx], evaluator)
    test_acc = accuracy(logits[test_idx], labels[test_idx], evaluator)
    return train_acc, val_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='训练R-GCN模型')
    parser.add_argument('--seed', type=int, default=8, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--dataset', choices=['ogbn-mag'], default='ogbn-mag', help='数据集')
    parser.add_argument('--num-hidden', type=int, default=32, help='隐藏层维数')
    parser.add_argument('--num-layers', type=int, default=2, help='模型层数')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout概率')
    parser.add_argument('--epochs', type=int, default=50, help='训练epoch数')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
