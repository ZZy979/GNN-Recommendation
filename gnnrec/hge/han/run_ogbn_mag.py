import argparse

import dgl
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from gnnrec.config import DATA_DIR
from gnnrec.hge.han.dataloader import MetapathNodeCollator
from gnnrec.hge.han.model import HAN
from gnnrec.hge.utils import set_random_seed, get_device, load_ogbn_mag, accuracy


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)

    g, features, labels, num_classes, train_idx, val_idx, test_idx, evaluator = \
        load_ogbn_mag(DATA_DIR, True)
    features = features.to(device)
    labels = labels.to(device)

    # PAP, PFP
    metapaths = [['writes_rev', 'writes'], ['has_topic', 'has_topic_rev']]
    etype_subgraphs = [dgl.edge_type_subgraph(g, etypes) for etypes in metapaths]
    collators = [
        MetapathNodeCollator(eg, None, metapath, args.neighbor_size)
        for eg, metapath in zip(etype_subgraphs, metapaths)
    ]
    train_dataloader = DataLoader(train_idx, batch_size=args.batch_size)
    val_dataloader = DataLoader(val_idx, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_idx, batch_size=args.batch_size)

    model = HAN(
        len(metapaths), features.shape[1], args.num_hidden, num_classes,
        args.num_heads, args.dropout
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        model.train()
        logits, train_labels, losses = [], [], []
        for batch in tqdm(train_dataloader):
            mgs = [collator.collate(batch).to(device) for collator in collators]
            in_feats = [mg.srcdata['feat'] for mg in mgs]
            batch_labels = labels[batch]
            batch_logits = model(mgs, in_feats)
            loss = F.cross_entropy(batch_logits, batch_labels.squeeze(dim=1))

            logits.append(batch_logits.detach().cpu())
            train_labels.append(batch_labels.detach().cpu())
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

        train_acc = accuracy(torch.cat(logits, dim=0), torch.cat(train_labels, dim=0), evaluator)
        val_acc = evaluate(collators, val_dataloader, device, model, labels, evaluator)
        print('Epoch {:d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f}'.format(
            epoch, torch.tensor(losses).mean().item(), train_acc, val_acc
        ))
    test_acc = evaluate(collators, test_dataloader, device, model, labels, evaluator)
    print('Test Acc {:.4f}'.format(test_acc))


@torch.no_grad()
def evaluate(collators, dataloader, device, model, labels, evaluator):
    model.eval()
    logits, eval_labels = [], []
    for batch in dataloader:
        mgs = [collator.collate(batch).to(device) for collator in collators]
        in_feats = [mg.srcdata['feat'] for mg in mgs]
        batch_labels = labels[batch]
        batch_logits = model(mgs, in_feats)

        logits.append(batch_logits.detach().cpu())
        eval_labels.append(batch_labels.detach().cpu())
    return accuracy(torch.cat(logits, dim=0), torch.cat(eval_labels, dim=0), evaluator)


def main():
    parser = argparse.ArgumentParser(description='ogbn-mag数据集 HAN模型')
    parser.add_argument('--seed', type=int, default=8, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--num-hidden', type=int, default=8, help='隐藏层维数')
    parser.add_argument('--num-heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout概率')
    parser.add_argument('--epochs', type=int, default=50, help='训练epoch数')
    parser.add_argument('--batch-size', type=int, default=2048, help='批大小')
    parser.add_argument('--neighbor-size', type=int, default=10, help='邻居采样数')
    parser.add_argument('--lr', type=float, default=0.005, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='权重衰减')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
