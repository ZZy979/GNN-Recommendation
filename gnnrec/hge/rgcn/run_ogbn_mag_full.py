import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim

from gnnrec.config import DATA_DIR
from gnnrec.hge.rgcn.model import RGCNFull
from gnnrec.hge.utils import set_random_seed, get_device, load_ogbn_mag, accuracy


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)

    g, features, labels, num_classes, train_idx, val_idx, test_idx, evaluator = \
        load_ogbn_mag(DATA_DIR, True, device)

    model = RGCNFull(
        {ntype: g.num_nodes(ntype) for ntype in g.ntypes},
        {'paper': features.shape[1]}, args.num_hidden, num_classes, g.etypes, 'paper',
        args.num_hidden_layers, dropout=args.dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model_input = {'paper': features}
    for epoch in range(args.epochs):
        model.train()
        logits = model(g, model_input)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx].squeeze(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits[train_idx], labels[train_idx], evaluator)
        val_acc = evaluate(model, g, model_input, labels, val_idx, evaluator)
        print('Epoch {:d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f}'.format(
            epoch, loss.item(), train_acc, val_acc
        ))
    test_acc = evaluate(model, g, model_input, labels, test_idx, evaluator)
    print('Test Acc {:.4f}'.format(test_acc))


@torch.no_grad()
def evaluate(model, g, features, labels, mask, evaluator):
    model.eval()
    logits = model(g, features)
    return accuracy(logits[mask], labels[mask], evaluator)


def main():
    parser = argparse.ArgumentParser(description='ogbn-mag数据集 R-GCN模型')
    parser.add_argument('--seed', type=int, default=8, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--num-hidden', type=int, default=32, help='隐藏层维数')
    parser.add_argument('--num-hidden-layers', type=int, default=1, help='隐藏层数')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout概率')
    parser.add_argument('--epochs', type=int, default=100, help='训练epoch数')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='权重衰减')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
