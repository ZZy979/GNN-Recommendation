import argparse

import dgl
import torch
import torch.nn.functional as F

from gnnrec.config import DATA_DIR
from gnnrec.hge.cs.model import LabelPropagation
from gnnrec.hge.mygnn.model import HeCo
from gnnrec.hge.utils import get_device, load_ogbn_mag, load_pretrained_node_embed, accuracy


def smooth(base_pred, g, labels, evaluator, mask, args):
    cs = LabelPropagation(args.num_smooth_layers, args.smooth_alpha, args.smooth_norm)
    labels = F.one_hot(labels.squeeze(dim=1)).float()
    base_pred[mask] = labels[mask]
    return cs(g, base_pred)


def main():
    args = parse_args()
    print(args)
    device = get_device(args.device)

    g, feat, labels, num_classes, train_idx, val_idx, test_idx, evaluator = \
        load_ogbn_mag(DATA_DIR, True, device, False)
    g = g.cpu()
    load_pretrained_node_embed(g, args.node_embed_path)
    pos_g = dgl.load_graphs(args.pos_graph_path)[0][0].to(device)
    pos = pos_g.edges()[0].view(pos_g.num_nodes(), -1)  # (N_p, T_pos) 每个paper顶点的正样本id
    relations = [
        ('author', 'writes', 'paper'),
        ('paper', 'cites', 'paper'),
        ('field_of_study', 'has_topic_rev', 'paper')
    ]

    model = HeCo(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes},
        args.num_hidden, num_classes, args.feat_drop, args.attn_drop,
        relations, args.tau, args.lambda_
    )
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    model.eval()

    base_pred = model.get_embeds(g, g.ndata['feat'], pos, args.batch_size, device)
    mask = torch.cat([train_idx, val_idx])
    logits = smooth(base_pred, pos_g, labels, evaluator, mask, args)
    test_acc = accuracy(logits[test_idx], labels[test_idx], evaluator)
    print('After smoothing: Test Acc {:.4f}'.format(test_acc))


def parse_args():
    parser = argparse.ArgumentParser(description='HeCo+C&S（仅Smooth步骤） ogbn-mag数据集')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    # HeCo
    parser.add_argument('--num-hidden', type=int, default=64, help='隐藏层维数')
    parser.add_argument('--feat-drop', type=float, default=0.3, help='特征dropout')
    parser.add_argument('--attn-drop', type=float, default=0.5, help='注意力dropout')
    parser.add_argument('--tau', type=float, default=0.8, help='温度参数')
    parser.add_argument('--lambda', type=float, default=0.5, dest='lambda_', help='对比损失的平衡系数')
    parser.add_argument('--batch-size', type=int, default=4096, help='批大小')
    parser.add_argument('node_embed_path', help='预训练顶点嵌入路径')
    parser.add_argument('pos_graph_path', help='正样本图保存路径')
    parser.add_argument('model_path', help='预训练的HeCo模型保存路径')
    # C&S
    parser.add_argument('--num-smooth-layers', type=int, default=50, help='Smooth步骤传播层数')
    parser.add_argument('--smooth-alpha', type=float, default=0.5, help='Smooth步骤α值')
    parser.add_argument(
        '--smooth-norm', choices=['left', 'right', 'both'], default='both',
        help='Smooth步骤归一化方式'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
