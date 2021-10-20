import argparse

import dgl
import torch
import torch.nn.functional as F

from gnnrec.hge.cs.model import LabelPropagation
from gnnrec.hge.rhco.model import RHCO
from gnnrec.hge.utils import get_device, load_data, add_node_feat, accuracy


def smooth(base_pred, g, labels, mask, args):
    cs = LabelPropagation(args.num_smooth_layers, args.smooth_alpha, args.smooth_norm)
    labels = F.one_hot(labels).float()
    base_pred[mask] = labels[mask]
    return cs(g, base_pred)


def main():
    args = parse_args()
    print(args)
    device = get_device(args.device)
    data, g, _, labels, predict_ntype, train_idx, val_idx, test_idx, evaluator = \
        load_data(args.dataset, device)
    add_node_feat(g, 'pretrained', args.node_embed_path, True)
    pos_g = dgl.load_graphs(args.pos_graph_path)[0][0].to(device)

    model = RHCO(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes},
        args.num_hidden, data.num_classes, args.num_rel_hidden, args.num_heads,
        g.ntypes, g.canonical_etypes, predict_ntype, args.num_layers, args.dropout,
        args.tau, args.lambda_
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    base_pred = model.get_embeds(g, predict_ntype, args.neighbor_size, args.batch_size, device)
    mask = torch.cat([train_idx, val_idx])
    logits = smooth(base_pred, pos_g, labels, mask, args)
    test_acc = accuracy(logits[test_idx], labels[test_idx], evaluator)
    print('After smoothing: Test Acc {:.4f}'.format(test_acc))


def parse_args():
    parser = argparse.ArgumentParser(description='RHCO+C&S（仅Smooth步骤）')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--dataset', choices=['ogbn-mag'], default='ogbn-mag', help='数据集')
    # RHCO
    parser.add_argument('--num-hidden', type=int, default=64, help='隐藏层维数')
    parser.add_argument('--num-rel-hidden', type=int, default=8, help='关系表示的隐藏层维数')
    parser.add_argument('--num-heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--num-layers', type=int, default=2, help='层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--tau', type=float, default=0.8, help='温度参数')
    parser.add_argument('--lambda', type=float, default=0.5, dest='lambda_', help='对比损失的平衡系数')
    parser.add_argument('--batch-size', type=int, default=1024, help='批大小')
    parser.add_argument('--neighbor-size', type=int, default=10, help='邻居采样数')
    parser.add_argument('node_embed_path', help='预训练顶点嵌入路径')
    parser.add_argument('pos_graph_path', help='正样本图保存路径')
    parser.add_argument('model_path', help='预训练的模型保存路径')
    # C&S
    parser.add_argument('--num-smooth-layers', type=int, default=50, help='Smooth步骤传播层数')
    parser.add_argument('--smooth-alpha', type=float, default=0.5, help='Smooth步骤α值')
    parser.add_argument(
        '--smooth-norm', choices=['left', 'right', 'both'], default='right',
        help='Smooth步骤归一化方式'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
