import argparse
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.data.utils import load_graphs, load_info
from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader
from tqdm import tqdm

from gnnrec.config import DATA_DIR
from gnnrec.hge.hetgnn.model import HetGNN
from gnnrec.hge.utils import set_random_seed, get_device, load_ogbn_mag, accuracy


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)

    g = load_graphs(os.path.join(args.data_path, 'ogbn_mag_neighbor_graph.bin'))[0][0]
    feats = load_info(os.path.join(args.data_path, 'ogbn_mag_in_feats.pkl'))
    g.ndata['feat'] = feats
    _, _, labels, num_classes, train_idx, val_idx, test_idx, evaluator = \
        load_ogbn_mag(DATA_DIR, device=device)

    sampler = MultiLayerFullNeighborSampler(1)
    train_loader = NodeDataLoader(g, {'paper': train_idx}, sampler, batch_size=args.batch_size)
    val_loader = NodeDataLoader(g, {'paper': val_idx}, sampler, batch_size=args.batch_size)
    test_loader = NodeDataLoader(g, {'paper': test_idx}, sampler, batch_size=args.batch_size)

    model = HetGNN(
        feats['author'].shape[-1], args.num_hidden, num_classes, g.ntypes, 'paper', args.dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        model.train()
        logits, train_labels, losses = [], [], []
        for input_nodes, output_nodes, blocks in tqdm(train_loader):
            block = blocks[0].to(device)
            batch_labels = labels[output_nodes['paper']]
            batch_logits = model(block, block.srcdata['feat'])
            loss = F.cross_entropy(batch_logits, batch_labels.squeeze(dim=1))

            logits.append(batch_logits.detach().cpu())
            train_labels.append(batch_labels.detach().cpu())
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

        train_acc = accuracy(torch.cat(logits, dim=0), torch.cat(train_labels, dim=0), evaluator)
        val_acc = evaluate(val_loader, device, model, labels, evaluator)
        print('Epoch {:d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f}'.format(
            epoch, torch.tensor(losses).mean().item(), train_acc, val_acc
        ))
    test_acc = evaluate(test_loader, device, model, labels, evaluator)
    print('Test Acc {:.4f}'.format(test_acc))


@torch.no_grad()
def evaluate(loader, device, model, labels, evaluator):
    model.eval()
    logits, eval_labels = [], []
    for input_nodes, output_nodes, blocks in loader:
        block = blocks[0].to(device)
        batch_labels = labels[output_nodes['paper']]
        batch_logits = model(block, block.srcdata['feat'])

        logits.append(batch_logits.detach().cpu())
        eval_labels.append(batch_labels.detach().cpu())
    return accuracy(torch.cat(logits, dim=0), torch.cat(eval_labels, dim=0), evaluator)


def main():
    parser = argparse.ArgumentParser(description='HetGNN无监督训练')
    parser.add_argument('--seed', type=int, default=10, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--num-hidden', type=int, default=128, help='隐藏层维数')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout概率')
    parser.add_argument('--epochs', type=int, default=50, help='训练epoch数')
    parser.add_argument('--batch-size', type=int, default=4096, help='批大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='权重衰减')
    parser.add_argument('data_path', help='预处理数据所在目录')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
