import argparse
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim

from gnnrec.hge.hgt.model import HGTFull
from gnnrec.hge.utils import set_random_seed, get_device, load_data, add_node_feat, evaluate_full


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    data, g, _, labels, predict_ntype, train_idx, val_idx, test_idx, _ = \
        load_data(args.dataset, device)
    add_node_feat(g, 'one-hot')

    model = HGTFull(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes},
        args.num_hidden, data.num_classes, args.num_heads, g.ntypes, g.canonical_etypes,
        predict_ntype, args.num_layers, args.dropout
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), eps=1e-6)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, args.max_lr, epochs=args.epochs, steps_per_epoch=1,
        pct_start=0.05, anneal_strategy='linear', final_div_factor=10.0
    )
    warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
    for epoch in range(args.epochs):
        model.train()
        logits = model(g, g.ndata['feat'])
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()
        print('Epoch {:d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f} | Test Acc {:.4f}'.format(
            epoch, loss.item(), *evaluate_full(model, g, labels, train_idx, val_idx, test_idx)
        ))


def main():
    parser = argparse.ArgumentParser(description='训练HGT模型(full-batch)')
    parser.add_argument('--seed', type=int, default=1, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--dataset', choices=['acm', 'dblp'], default='acm', help='数据集')
    parser.add_argument('--num-hidden', type=int, default=512, help='隐藏层维数')
    parser.add_argument('--num-heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--num-layers', type=int, default=2, help='层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--epochs', type=int, default=10, help='训练epoch数')
    parser.add_argument('--max-lr', type=float, default=5e-4, help='学习率上界')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
