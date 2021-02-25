import argparse

import dgl
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.utils import set_new_frames, extract_node_subframes
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from gnnrec.config import DATA_DIR
from gnnrec.hge.models.rgcn import RGCN


def train(args):
    data = DglNodePropPredDataset('ogbn-mag', DATA_DIR)
    predict_ntype = 'paper'
    g, labels = data[0]
    g = add_reverse_edges(g)
    features = g.nodes[predict_ntype].data['feat']
    labels = labels[predict_ntype]
    split_idx = data.get_idx_split()
    train_idx = split_idx['train'][predict_ntype]
    val_idx = split_idx['valid'][predict_ntype]
    test_idx = split_idx['test'][predict_ntype]

    model = RGCN(
        {ntype: g.num_nodes(ntype) for ntype in g.ntypes},
        {'paper': features.shape[1]}, args.num_hidden, data.num_classes, g.etypes,
        args.num_hidden_layers, args.num_bases, args.self_loop, args.dropout
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    evaluator = Evaluator(data.name)
    for epoch in range(args.epochs):
        model.train()
        logits = model(g, {'paper': features})[predict_ntype]
        loss = F.cross_entropy(logits[train_idx], labels[train_idx].squeeze(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits[train_idx], labels[train_idx], evaluator)
        val_acc = evaluate(model, g, {'paper': features}, labels, val_idx, evaluator)
        print('Epoch {:d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f}'.format(
            epoch, loss.item(), train_acc, val_acc
        ))

    test_acc = evaluate(model, g, {'paper': features}, labels, test_idx, evaluator)
    print('Test Acc {:.4f}'.format(test_acc))


def add_reverse_edges(g):
    data = {}
    for stype, etype, dtype in g.canonical_etypes:
        u, v = g.edges(etype=(stype, etype, dtype))
        data[(stype, etype, dtype)] = u, v
        if stype != dtype:
            data[(dtype, etype + '_rev', stype)] = v, u
    new_g = dgl.heterograph(data)
    set_new_frames(new_g, node_frames=extract_node_subframes(g, None))
    return new_g


def accuracy(logits, labels, evaluator):
    predict = logits.argmax(dim=-1, keepdim=True)
    return evaluator.eval({'y_true': labels, 'y_pred': predict})['acc']


def evaluate(model, g, features, labels, mask, evaluator):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)['paper']
    return accuracy(logits[mask], labels[mask], evaluator)


def main():
    parser = argparse.ArgumentParser(description='ogbn-mag')
    parser.add_argument('--num-hidden', type=int, default=64, help='隐藏层维数')
    parser.add_argument('--num-hidden-layers', type=int, default=1, help='隐藏层数')
    parser.add_argument('--num-bases', type=int, default=0, help='基的个数')
    parser.add_argument('--self-loop', action='store_true', help='是否包括自环消息')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--epochs', type=int, default=50, help='训练epoch数')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='权重衰减')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
