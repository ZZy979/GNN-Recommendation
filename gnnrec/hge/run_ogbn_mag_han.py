import argparse

import dgl
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerNeighborSampler, NodeCollator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from torch.utils.data import DataLoader

from gnnrec.config import DATA_DIR
from gnnrec.hge.models.han import HAN
from gnnrec.hge.utils import set_random_seed, add_reverse_edges


def train(args):
    set_random_seed(args.seed)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    data = DglNodePropPredDataset('ogbn-mag', DATA_DIR)
    g, labels = data[0]
    g = add_reverse_edges(g)
    features = g.nodes['paper'].data['feat'].to(device)
    labels = labels['paper'].to(device)
    split_idx = data.get_idx_split()
    train_idx = split_idx['train']['paper']
    val_idx = split_idx['valid']['paper']
    test_idx = split_idx['test']['paper']

    # PAP, PFP
    # metapaths = [['writes_rev', 'writes'], ['has_topic', 'has_topic_rev']]
    metapaths = [['writes_rev', 'writes']]
    mgs = [dgl.metapath_reachable_graph(g, metapath) for metapath in metapaths]
    sampler = MultiLayerNeighborSampler([args.neighbor_size])
    collators = [NodeCollator(mg, None, sampler) for mg in mgs]
    train_dataloader = DataLoader(train_idx, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_idx, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_idx, batch_size=args.batch_size)

    model = HAN(
        len(mgs), features.shape[1], args.num_hidden, data.num_classes, args.num_heads, args.dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    evaluator = Evaluator(data.name)
    for epoch in range(args.epochs):
        model.train()
        logits, train_labels, losses = [], [], []
        for batch in train_dataloader:
            gs = [collator.collate(batch)[2][0].to(device) for collator in collators]
            batch_labels = labels[batch]
            batch_logits = model(gs, gs[0].srcdata['feat'])
            loss = F.cross_entropy(batch_logits, batch_labels.squeeze(dim=1))

            logits.append(batch_logits)
            train_labels.append(batch_labels)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = accuracy(torch.cat(logits, dim=0), torch.cat(train_labels, dim=0), evaluator)
        val_acc = evaluate(collators, val_dataloader, device, model, labels, evaluator)
        print('Epoch {:d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f}'.format(
            epoch, torch.tensor(losses).mean().item(), train_acc, val_acc
        ))
    test_acc = evaluate(collators, test_dataloader, device, model, labels, evaluator)
    print('Test Acc {:.4f}'.format(test_acc))


def accuracy(logits, labels, evaluator):
    predict = logits.argmax(dim=-1, keepdim=True).cpu()
    return evaluator.eval({'y_true': labels.cpu(), 'y_pred': predict})['acc']


def evaluate(collators, dataloader, device, model, labels, evaluator):
    model.eval()
    logits, eval_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            gs = [collator.collate(batch)[2][0].to(device) for collator in collators]
            batch_labels = labels[batch]
            batch_logits = model(gs, gs[0].srcdata['feat'])

            logits.append(batch_logits)
            eval_labels.append(batch_labels)
    return accuracy(torch.cat(logits, dim=0), torch.cat(eval_labels, dim=0), evaluator)


def main():
    parser = argparse.ArgumentParser(description='ogbn-mag数据集 HAN模型')
    parser.add_argument('--seed', type=int, default=8, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--num-hidden', type=int, default=8, help='隐藏层维数')
    parser.add_argument('--num-heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout概率')
    parser.add_argument('--epochs', type=int, default=50, help='训练epoch数')
    parser.add_argument('--batch-size', type=int, default=1024, help='批大小')
    parser.add_argument('--neighbor-size', type=int, default=10, help='邻居采样数')
    parser.add_argument('--lr', type=float, default=0.005, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='权重衰减')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
