import argparse

import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from ogb.nodeproppred import Evaluator
from tqdm import tqdm

from gnnrec.config import DATA_DIR
from gnnrec.hge.rgcn.model import RGCN
from gnnrec.hge.utils import set_random_seed, get_device, load_ogbn_mag, accuracy


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)

    data, g, features, labels, train_idx, val_idx, test_idx = load_ogbn_mag(DATA_DIR, True, device)
    g = g.cpu()
    evaluator = Evaluator(data.name)

    sampler = MultiLayerNeighborSampler([args.neighbor_size] * (args.num_hidden_layers + 1))
    train_loader = NodeDataLoader(g, {'paper': train_idx}, sampler, batch_size=args.batch_size)
    val_loader = NodeDataLoader(g, {'paper': val_idx}, sampler, batch_size=args.batch_size)
    test_loader = NodeDataLoader(g, {'paper': test_idx}, sampler, batch_size=args.batch_size)

    model = RGCN(
        {ntype: g.num_nodes(ntype) for ntype in g.ntypes},
        {'paper': features.shape[1]}, args.num_hidden, data.num_classes,
        get_rel_names(g, args.num_hidden_layers + 1),
        'paper', args.num_hidden_layers, dropout=args.dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        model.train()
        logits, train_labels, losses = [], [], []
        for input_nodes, output_nodes, blocks in tqdm(train_loader):
            blocks = [b.to(device) for b in blocks]
            features = {'paper': blocks[0].srcnodes['paper'].data['feat']}
            batch_labels = labels[output_nodes['paper']]
            batch_logits = model(blocks, features)
            loss = F.cross_entropy(batch_logits, batch_labels.squeeze(dim=1))

            # 仅用于计算准确率记得使用.detach().cpu()，否则CUDA会内存溢出！
            logits.append(batch_logits.detach().cpu())
            train_labels.append(batch_labels.detach().cpu())
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


def get_rel_names(g, num_layers):
    # 使用minibatch训练时block可能不包含原图中的所有边类型，因此R-GCN模型中缺少的边类型对应的参数没有使用，这将导致内存泄漏
    # 因此需要预先计算出每一层的block包含哪些边类型
    # https://discuss.dgl.ai/t/cuda-out-of-memory-error-training-after-a-few-epochs/666/4
    ntypes = ['paper']
    etypes = []
    for _ in range(num_layers):
        etypes.append(list(e for _, e, d in g.canonical_etypes if d in ntypes))
        ntypes = list(set(s for s, _, d in g.canonical_etypes if d in ntypes))
    etypes.reverse()
    return etypes


def evaluate(loader, device, model, labels, evaluator):
    model.eval()
    logits, eval_labels = [], []
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in loader:
            blocks = [b.to(device) for b in blocks]
            features = {'paper': blocks[0].srcnodes['paper'].data['feat']}
            batch_labels = labels[output_nodes['paper']]
            batch_logits = model(blocks, features)

            logits.append(batch_logits.detach().cpu())
            eval_labels.append(batch_labels.detach().cpu())
    return accuracy(torch.cat(logits, dim=0), torch.cat(eval_labels, dim=0), evaluator)


def main():
    parser = argparse.ArgumentParser(description='ogbn-mag数据集 R-GCN模型')
    parser.add_argument('--seed', type=int, default=8, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--num-hidden', type=int, default=64, help='隐藏层维数')
    parser.add_argument('--num-hidden-layers', type=int, default=1, help='隐藏层数')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout概率')
    parser.add_argument('--epochs', type=int, default=50, help='训练epoch数')
    parser.add_argument('--batch-size', type=int, default=2560, help='批大小')
    parser.add_argument('--neighbor-size', type=int, default=10, help='邻居采样数')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='权重衰减')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
