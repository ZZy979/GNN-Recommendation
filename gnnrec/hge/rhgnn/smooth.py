import argparse

import dgl
import torch
import torch.nn.functional as F
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from tqdm import tqdm

from gnnrec.config import DATA_DIR
from gnnrec.hge.cs.model import LabelPropagation
from gnnrec.hge.rhgnn.model import RHGNN
from gnnrec.hge.rhgnn.run_ogbn_mag import load_pretrained_node_embed
from gnnrec.hge.utils import get_device, load_ogbn_mag, accuracy


def get_embeds(g, num_classes, args, device):
    sampler = MultiLayerNeighborSampler(
        list(range(args.neighbor_size, args.neighbor_size + args.num_layers))
    )
    loader = NodeDataLoader(g, {'paper': g.nodes('paper')}, sampler, batch_size=args.batch_size)

    model = RHGNN(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes},
        args.num_hidden, num_classes, args.num_rel_hidden, args.num_rel_hidden, args.num_heads,
        g.ntypes, g.canonical_etypes, 'paper', args.num_layers, args.dropout, residual=args.residual
    )
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    model.eval()
    embeds = torch.zeros(g.num_nodes('paper'), num_classes, device=device)
    with torch.no_grad():
        for input_nodes, output_nodes, blocks in tqdm(loader):
            blocks = [b.to(device) for b in blocks]
            logits = model(blocks, blocks[0].srcdata['feat'])
            embeds[output_nodes['paper']] = logits
    return embeds


def smooth(base_pred, g, labels, evaluator, mask, args):
    cs = LabelPropagation(args.num_smooth_layers, args.smooth_alpha, args.smooth_norm)
    labels = F.one_hot(labels.squeeze(dim=1)).float()
    base_pred[mask] = labels[mask]
    return cs(g, base_pred)


def main():
    args = parse_args()
    print(args)
    device = get_device(args.device)

    g, _, labels, num_classes, train_idx, val_idx, test_idx, evaluator = \
        load_ogbn_mag(DATA_DIR, True, device)
    g = g.cpu()
    load_pretrained_node_embed(g, args.node_embed_path)

    base_pred = get_embeds(g, num_classes, args, device)
    test_acc = accuracy(base_pred[test_idx], labels[test_idx], evaluator)
    print('Base prediction: Test Acc {:.4f}'.format(test_acc))

    pg = dgl.load_graphs(args.paper_graph)[0][0].to(device)
    mask = torch.cat([train_idx, val_idx])
    logits = smooth(base_pred, pg, labels, evaluator, mask, args)
    test_acc = accuracy(logits[test_idx], labels[test_idx], evaluator)
    print('After smoothing: Test Acc {:.4f}'.format(test_acc))


def parse_args():
    parser = argparse.ArgumentParser(description='R-HGNN+C&S（仅Smooth步骤） ogbn-mag数据集')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    # R-HGNN
    parser.add_argument('--num-hidden', type=int, default=64, help='隐藏层维数')
    parser.add_argument('--num-rel-hidden', type=int, default=8, help='关系表示的隐藏层维数')
    parser.add_argument('--num-heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--num-layers', type=int, default=2, help='层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--no-residual', action='store_false', help='不使用残差连接', dest='residual')
    parser.add_argument('--batch-size', type=int, default=4096, help='批大小')
    parser.add_argument('--neighbor-size', type=int, default=10, help='邻居采样数')
    parser.add_argument('node_embed_path', help='预训练顶点嵌入路径')
    parser.add_argument('model_path', help='预训练的R-HGNN模型保存路径')
    # C&S
    parser.add_argument('--num-smooth-layers', type=int, default=50, help='Smooth步骤传播层数')
    parser.add_argument('--smooth-alpha', type=float, default=0.5, help='Smooth步骤α值')
    parser.add_argument(
        '--smooth-norm', choices=['left', 'right', 'both'], default='both',
        help='Smooth步骤归一化方式'
    )
    parser.add_argument('paper_graph', help='用于C&S的paper顶点同构图所在路径')
    return parser.parse_args()


if __name__ == '__main__':
    main()
