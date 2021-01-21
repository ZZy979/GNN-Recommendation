import argparse

import dgl
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerNeighborSampler, NodeCollator
from dgl.utils import set_new_frames, extract_node_subframes
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from torch.utils.data import DataLoader
from tqdm import tqdm

from gnnrec.config import DATA_DIR
from gnnrec.hge.models.han import HAN


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
    out_shape = (g.num_nodes(predict_ntype), data.num_classes)

    # PAP, PFP
    metapaths = [['writes_rev', 'writes'], ['has_topic_rev', 'has_topic']]
    mgs = [dgl.metapath_reachable_graph(g, metapath) for metapath in metapaths]
    sampler = MultiLayerNeighborSampler([args.neighbor_size])
    collators = [NodeCollator(mg, None, sampler) for mg in mgs]
    train_dataloader = DataLoader(train_idx, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_idx, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_idx, batch_size=args.batch_size)

    model = HAN(
        len(mgs), features.shape[1], args.num_hidden, data.num_classes, args.num_heads, args.dropout
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    evaluator = Evaluator(data.name)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        train_logits = torch.zeros(out_shape)
        for i, batch in tqdm(enumerate(train_dataloader)):
            gs, hs = [], []
            for collator in collators:
                input_nodes, output_nodes, blocks = collator.collate(batch)
                gs.append(blocks[0])
                hs.append(features[input_nodes])
            train_logits[batch] = logits = model(gs, hs)
            loss = F.cross_entropy(logits, labels[batch].squeeze(dim=1))
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = accuracy(train_logits[train_idx], labels[train_idx], evaluator)
        val_acc = evaluate(out_shape, collators, val_dataloader, model, mgs, features, labels, evaluator)
        print('Epoch {:d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f}'.format(
            epoch, torch.tensor(losses).mean().item(), train_acc, val_acc
        ))

    test_acc = evaluate(out_shape, collators, test_dataloader, model, mgs, features, labels, evaluator)
    print('Test Acc {:.4f}'.format(test_acc))


def add_reverse_edges(g):
    data = {}
    for stype, etype, dtype in g.canonical_etypes:
        u, v = g.edges(etype=(stype, etype, dtype))
        data[(stype, etype, dtype)] = u, v
        data[(dtype, etype + '_rev', stype)] = v, u
    new_g = dgl.heterograph(data)
    set_new_frames(new_g, node_frames=extract_node_subframes(g, None))
    return new_g


def accuracy(logits, labels, evaluator):
    predict = logits.argmax(dim=-1, keepdim=True)
    return evaluator.eval({'y_true': labels, 'y_pred': predict})['acc']


def evaluate(out_shape, collators, dataloader, model, gs, features, labels, evaluator):
    logits = torch.zeros(out_shape)
    idx = dataloader.dataset
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            gs, hs = [], []
            for collator in collators:
                input_nodes, output_nodes, blocks = collator.collate(batch)
                gs.append(blocks[0])
                hs.append(features[input_nodes])
            logits[batch] = model(gs, hs)
    return accuracy(logits[idx], labels[idx], evaluator)


def main():
    parser = argparse.ArgumentParser(description='OGB顶点分类(ogbn-mag)')
    parser.add_argument('--num-hidden', type=int, default=8, help='隐藏层维数')
    parser.add_argument('--num-heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout概率')
    parser.add_argument('--epochs', type=int, default=50, help='训练epoch数')
    parser.add_argument('--batch-size', type=int, default=32, help='批大小')
    parser.add_argument('--neighbor-size', type=int, default=100, help='邻居采样数')
    parser.add_argument('--lr', type=float, default=0.005, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='权重衰减')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
