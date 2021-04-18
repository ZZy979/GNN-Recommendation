import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from ogb.nodeproppred import Evaluator

from gnnrec.config import DATA_DIR
from gnnrec.hge.models.mlp import MLP
from gnnrec.hge.utils import set_random_seed, get_device, load_ogbn_mag, accuracy


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)

    data, g, features, labels, train_idx, val_idx, test_idx = load_ogbn_mag(DATA_DIR, device=device)
    evaluator = Evaluator(data.name)

    model = MLP(features.shape[1], args.num_hidden, data.num_classes, args.num_layers, args.dropout)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        model.train()
        logits = model(features[train_idx])
        loss = F.cross_entropy(logits, labels[train_idx].squeeze(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits, labels[train_idx], evaluator)
        val_acc = evaluate(model, features[val_idx], labels[val_idx], evaluator)
        print('Epoch {:d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f}'.format(
            epoch, loss.item(), train_acc, val_acc
        ))
    test_acc = evaluate(model, features[test_idx], labels[test_idx], evaluator)
    print('Test Acc {:.4f}'.format(test_acc))


def evaluate(model, features, labels, evaluator):
    model.eval()
    with torch.no_grad():
        logits = model(features)
    return accuracy(logits, labels, evaluator)


def main():
    parser = argparse.ArgumentParser(description='ogbn-mag数据集 MLP模型')
    parser.add_argument('--seed', type=int, default=8, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--num-hidden', type=int, default=256, help='隐藏层维数')
    parser.add_argument('--num-layers', type=int, default=3, help='模型层数')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout概率')
    parser.add_argument('--epochs', type=int, default=500, help='训练epoch数')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='权重衰减')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
