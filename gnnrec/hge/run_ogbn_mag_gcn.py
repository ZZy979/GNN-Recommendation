import argparse

import dgl
import torch
import torch.nn.functional as F
import torch.optim as optim
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from gnnrec.config import DATA_DIR
from gnnrec.hge.models.gcn import GCN
from gnnrec.hge.utils import set_random_seed, add_reverse_edges


def train(args):
    set_random_seed(args.seed)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    data = DglNodePropPredDataset('ogbn-mag', DATA_DIR)
    g, labels = data[0]
    g = dgl.metapath_reachable_graph(add_reverse_edges(g), ['writes_rev', 'writes']).to(device)
    features = g.ndata['feat']
    labels = labels['paper'].to(device)
    split_idx = data.get_idx_split()
    train_idx = split_idx['train']['paper'].to(device)
    val_idx = split_idx['valid']['paper'].to(device)
    test_idx = split_idx['test']['paper'].to(device)

    model = GCN(features.shape[1], args.num_hidden, data.num_classes, args.dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    evaluator = Evaluator(data.name)
    for epoch in range(args.epochs):
        model.train()
        logits = model(g, features)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx].squeeze(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits[train_idx], labels[train_idx], evaluator)
        val_acc = evaluate(model, g, features, labels, val_idx, evaluator)
        print('Epoch {:d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f}'.format(
            epoch, loss.item(), train_acc, val_acc
        ))

    test_acc = evaluate(model, g, features, labels, test_idx, evaluator)
    print('Test Acc {:.4f}'.format(test_acc))


def accuracy(logits, labels, evaluator):
    predict = logits.argmax(dim=-1, keepdim=True)
    return evaluator.eval({'y_true': labels, 'y_pred': predict})['acc']


def evaluate(model, g, features, labels, mask, evaluator):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
    return accuracy(logits[mask], labels[mask], evaluator)


def main():
    parser = argparse.ArgumentParser(description='ogbn-mag数据集 GCN模型')
    parser.add_argument('--seed', type=int, default=8, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--num-hidden', type=int, default=256, help='隐藏层维数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--epochs', type=int, default=200, help='训练epoch数')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='权重衰减')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
