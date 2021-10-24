import argparse

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gnnrec.hge.cs.model import CorrectAndSmooth
from gnnrec.hge.utils import set_random_seed, get_device, load_data, calc_metrics, METRICS_STR


def train_base_model(base_model, feats, labels, train_idx, val_idx, test_idx, args):
    print('Training base model...')
    optimizer = optim.Adam(base_model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        base_model.train()
        logits = base_model(feats)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(('Epoch {:d} | Loss {:.4f} | ' + METRICS_STR).format(
            epoch, loss.item(), *evaluate(base_model, feats, labels, train_idx, val_idx, test_idx)
        ))


@torch.no_grad()
def evaluate(model, feats, labels, train_idx, val_idx, test_idx):
    model.eval()
    logits = model(feats)
    return calc_metrics(logits, labels, train_idx, val_idx, test_idx)


def correct_and_smooth(base_model, g, feats, labels, train_idx, val_idx, test_idx, args):
    print('Training C&S...')
    base_model.eval()
    base_pred = base_model(feats).softmax(dim=1)  # 注意要softmax

    cs = CorrectAndSmooth(
        args.num_correct_layers, args.correct_alpha, args.correct_norm,
        args.num_smooth_layers, args.smooth_alpha, args.smooth_norm, args.scale
    )
    mask = torch.cat([train_idx, val_idx])
    logits = cs(g, F.one_hot(labels).float(), base_pred, mask)
    _, _, test_acc, _, _, test_f1 = calc_metrics(logits, labels, train_idx, val_idx, test_idx)
    print('Test Acc {:.4f} | Test Macro-F1 {:.4f}'.format(test_acc, test_f1))


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    data, g, feat, labels, predict_ntype, train_idx, val_idx, test_idx, evaluator = \
        load_data(args.dataset, device)
    feat = (feat - feat.mean(dim=0)) / feat.std(dim=0)
    # 标签传播图
    if args.dataset == 'ogbn-mag':
        pg = dgl.load_graphs(args.prop_graph)[0][0].to(device)
    else:
        pos_v, pos_u = data.pos
        pg = dgl.graph((pos_u, pos_v), device=device)

    base_model = nn.Linear(feat.shape[1], data.num_classes).to(device)
    train_base_model(base_model, feat, labels, train_idx, val_idx, test_idx, args)
    correct_and_smooth(base_model, pg, feat, labels, train_idx, val_idx, test_idx, args)


def main():
    parser = argparse.ArgumentParser(description='训练C&S模型')
    parser.add_argument('--seed', type=int, default=0, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--dataset', choices=['acm', 'dblp', 'ogbn-mag'], default='ogbn-mag', help='数据集')
    # 基础模型
    parser.add_argument('--epochs', type=int, default=300, help='基础模型训练epoch数')
    parser.add_argument('--lr', type=float, default=0.01, help='基础模型学习率')
    # C&S
    parser.add_argument('--prop-graph', help='标签传播图所在路径')
    parser.add_argument('--num-correct-layers', type=int, default=50, help='Correct步骤传播层数')
    parser.add_argument('--correct-alpha', type=float, default=0.5, help='Correct步骤α值')
    parser.add_argument(
        '--correct-norm', choices=['left', 'right', 'both'], default='both',
        help='Correct步骤归一化方式'
    )
    parser.add_argument('--num-smooth-layers', type=int, default=50, help='Smooth步骤传播层数')
    parser.add_argument('--smooth-alpha', type=float, default=0.5, help='Smooth步骤α值')
    parser.add_argument(
        '--smooth-norm', choices=['left', 'right', 'both'], default='both',
        help='Smooth步骤归一化方式'
    )
    parser.add_argument('--scale', type=float, default=20, help='放缩系数')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
