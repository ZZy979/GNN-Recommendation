import argparse
import warnings

import dgl.function as fn
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from tqdm import tqdm

from gnnrec.config import DATA_DIR
from gnnrec.hge.models.hgconv import HGConv
from gnnrec.hge.utils import set_random_seed, add_reverse_edges


def train(args):
    set_random_seed(args.seed)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    data = DglNodePropPredDataset('ogbn-mag', DATA_DIR)
    g, labels = data[0]
    g = add_reverse_edges(g)
    add_node_feat(g)
    labels = labels['paper'].to(device)
    split_idx = data.get_idx_split()
    train_idx = split_idx['train']['paper'].to(device)
    val_idx = split_idx['valid']['paper'].to(device)
    test_idx = split_idx['test']['paper'].to(device)

    sampler = MultiLayerNeighborSampler([args.neighbor_size] * args.num_layers)
    train_loader = NodeDataLoader(g, {'paper': train_idx}, sampler, batch_size=args.batch_size)
    val_loader = NodeDataLoader(g, {'paper': val_idx}, sampler, batch_size=args.batch_size)
    test_loader = NodeDataLoader(g, {'paper': test_idx}, sampler, batch_size=args.batch_size)

    model = HGConv(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes},
        args.num_hidden, data.num_classes, args.num_heads, g, 'paper',
        args.num_layers, args.dropout, args.residual
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    evaluator = Evaluator(data.name)
    warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
    for epoch in range(args.epochs):
        model.train()
        logits, train_labels, losses = [], [], []
        for input_nodes, output_nodes, blocks in tqdm(train_loader):
            blocks = [b.to(device) for b in blocks]
            features = {ntype: blocks[0].srcnodes[ntype].data['feat'] for ntype in blocks[0].ntypes}
            batch_labels = labels[output_nodes['paper']]
            batch_logits = model(blocks, features)
            loss = F.cross_entropy(batch_logits, batch_labels.squeeze(dim=1))

            logits.append(batch_logits)
            train_labels.append(batch_labels)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_acc = accuracy(torch.cat(logits, dim=0), torch.cat(train_labels, dim=0), evaluator)
        val_acc = evaluate(val_loader, device, model, labels, evaluator)
        print('Epoch {:d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f}'.format(
            epoch, torch.tensor(losses).mean().item(), train_acc, val_acc
        ))
    test_acc = evaluate(test_loader, device, model, labels, evaluator)
    print('Test Acc {:.4f}'.format(test_acc))


def add_node_feat(g):
    g.multi_update_all({'writes_rev': (fn.copy_u('feat', 'm'), fn.mean('m', 'feat'))}, 'sum')
    g.multi_update_all({'affiliated_with': (fn.copy_u('feat', 'm'), fn.mean('m', 'feat'))}, 'sum')
    g.nodes['field_of_study'].data['feat'] = torch.eye(g.num_nodes('field_of_study'))


def accuracy(logits, labels, evaluator):
    predict = logits.argmax(dim=-1, keepdim=True).cpu()
    return evaluator.eval({'y_true': labels.cpu(), 'y_pred': predict})['acc']


def evaluate(loader, device, model, labels, evaluator):
    model.eval()
    logits, eval_labels = [], []
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in loader:
            blocks = [b.to(device) for b in blocks]
            features = {ntype: blocks[0].srcnodes[ntype].data['feat'] for ntype in blocks[0].ntypes}
            batch_labels = labels[output_nodes['paper']]
            batch_logits = model(blocks, features)

            logits.append(batch_logits)
            eval_labels.append(batch_labels)
    return accuracy(torch.cat(logits, dim=0), torch.cat(eval_labels, dim=0), evaluator)


def main():
    parser = argparse.ArgumentParser(description='ogbn-mag数据集 HGConv模型')
    parser.add_argument('--seed', type=int, default=8, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--num-hidden', type=int, default=32, help='隐藏层维数')
    parser.add_argument('--num-heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--num-layers', type=int, default=2, help='层数')
    parser.add_argument('--no-residual', action='store_false', help='不使用残差连接', dest='residual')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--epochs', type=int, default=200, help='训练epoch数')
    parser.add_argument('--batch-size', type=int, default=2560, help='批大小')
    parser.add_argument('--neighbor-size', type=int, default=10, help='邻居采样数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='权重衰减')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
