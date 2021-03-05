import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from gnnrec.config import DATA_DIR
from gnnrec.hge.models.rgcn import RGCN
from gnnrec.hge.utils import set_random_seed, add_reverse_edges


def train(args):
    set_random_seed(args.seed)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    data = DglNodePropPredDataset('ogbn-mag', DATA_DIR)
    g, labels = data[0]
    g = add_reverse_edges(g).to(device)
    features = g.nodes['paper'].data['feat']
    labels = labels['paper'].to(device)
    split_idx = data.get_idx_split()
    train_idx = split_idx['train']['paper'].to(device)
    val_idx = split_idx['valid']['paper'].to(device)
    test_idx = split_idx['test']['paper'].to(device)

    model = RGCN(
        {ntype: g.num_nodes(ntype) for ntype in g.ntypes},
        {'paper': features.shape[1]}, args.num_hidden, data.num_classes, g.etypes,
        args.num_hidden_layers, dropout=args.dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    evaluator = Evaluator(data.name)
    model_input = {'paper': features}
    for epoch in range(args.epochs):
        model.train()
        logits = model(g, model_input)['paper']
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


def accuracy(logits, labels, evaluator):
    predict = logits.argmax(dim=-1, keepdim=True)
    return evaluator.eval({'y_true': labels, 'y_pred': predict})['acc']


def evaluate(model, g, features, labels, mask, evaluator):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)['paper']
    return accuracy(logits[mask], labels[mask], evaluator)


def main():
    parser = argparse.ArgumentParser(description='ogbn-mag数据集 R-GCN模型')
    parser.add_argument('--seed', type=int, default=8, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--num-hidden', type=int, default=64, help='隐藏层维数')
    parser.add_argument('--num-hidden-layers', type=int, default=1, help='隐藏层数')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout概率')
    parser.add_argument('--epochs', type=int, default=50, help='训练epoch数')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='权重衰减')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
