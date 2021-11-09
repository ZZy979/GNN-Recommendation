import argparse

import dgl
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import NodeDataLoader
from torch.utils.data import DataLoader
from tqdm import tqdm

from gnnrec.hge.heco.sampler import PositiveSampler
from gnnrec.hge.rhco.model import RHCO
from gnnrec.hge.utils import set_random_seed, get_device, load_data, add_node_feat, calc_metrics, \
    METRICS_STR


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    data, g, _, labels, predict_ntype, train_idx, val_idx, test_idx, evaluator = \
        load_data(args.dataset, device)
    add_node_feat(g, 'pretrained', args.node_embed_path, True)
    features = g.nodes[predict_ntype].data['feat']

    (*mgs, pos_g), _ = dgl.load_graphs(args.pos_graph_path)
    mgs = [mg.to(device) for mg in mgs]
    pos_g = pos_g.to(device)
    pos = pos_g.in_edges(pos_g.nodes())[0].view(pos_g.num_nodes(), -1)  # (N, T_pos) 每个目标顶点的正样本id
    # 不能用pos_g.edges()，必须按终点id排序

    id_loader = DataLoader(train_idx, batch_size=args.batch_size)
    loader = NodeDataLoader(
        g, {predict_ntype: train_idx}, PositiveSampler([args.neighbor_size] * args.num_layers, pos),
        device=device, batch_size=args.batch_size
    )
    sampler = PositiveSampler([None], pos)
    mg_loaders = [
        NodeDataLoader(mg, train_idx, sampler, device=device, batch_size=args.batch_size)
        for mg in mgs
    ]
    pos_loader = NodeDataLoader(pos_g, train_idx, sampler, device=device, batch_size=args.batch_size)

    model = RHCO(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes},
        args.num_hidden, data.num_classes, args.num_rel_hidden, args.num_heads,
        g.ntypes, g.canonical_etypes, predict_ntype, args.num_layers, args.dropout,
        len(mgs), args.tau, args.lambda_
    ).to(device)
    if args.load_path:
        model.load_state_dict(torch.load(args.load_path, map_location=device))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(loader) * args.epochs, eta_min=args.lr / 100
    )
    alpha = args.contrast_weight
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for (batch, (_, _, blocks), *mg_blocks, (_, _, pos_blocks)) in tqdm(zip(id_loader, loader, *mg_loaders, pos_loader)):
            mg_feats = [features[i] for i, _, _ in mg_blocks]
            mg_blocks = [b[0] for _, _, b in mg_blocks]
            pos_block = pos_blocks[0]
            # pos_block.num_dst_nodes() = batch_size + 正样本数
            batch_pos = torch.zeros(pos_block.num_dst_nodes(), batch.shape[0], dtype=torch.int, device=device)
            batch_pos[pos_block.in_edges(torch.arange(batch.shape[0], device=device))] = 1
            contrast_loss, logits = model(blocks, blocks[0].srcdata['feat'], mg_blocks, mg_feats, batch_pos.t())
            clf_loss = F.cross_entropy(logits, labels[batch])
            loss = alpha * contrast_loss + (1 - alpha) * clf_loss
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()
        print('Epoch {:d} | Loss {:.4f}'.format(epoch, sum(losses) / len(losses)))
        torch.save(model.state_dict(), args.save_path)
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            print(METRICS_STR.format(*evaluate(
                model, g, args.neighbor_size, args.batch_size, device,
                labels, train_idx, val_idx, test_idx, evaluator
            )))
    torch.save(model.state_dict(), args.save_path)
    print('模型已保存到', args.save_path)


@torch.no_grad()
def evaluate(model, g, neighbor_size, batch_size, device, labels, train_idx, val_idx, test_idx, evaluator):
    model.eval()
    embeds = model.get_embeds(g, neighbor_size, batch_size, device)
    return calc_metrics(embeds, labels, train_idx, val_idx, test_idx, evaluator)


def main():
    parser = argparse.ArgumentParser(description='训练RHCO模型')
    parser.add_argument('--seed', type=int, default=0, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--dataset', choices=['ogbn-mag', 'oag-venue'], default='ogbn-mag', help='数据集')
    parser.add_argument('--num-hidden', type=int, default=64, help='隐藏层维数')
    parser.add_argument('--num-rel-hidden', type=int, default=8, help='关系表示的隐藏层维数')
    parser.add_argument('--num-heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--num-layers', type=int, default=2, help='层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--tau', type=float, default=0.8, help='温度参数')
    parser.add_argument('--lambda', type=float, default=0.5, dest='lambda_', help='对比损失的平衡系数')
    parser.add_argument('--epochs', type=int, default=150, help='训练epoch数')
    parser.add_argument('--batch-size', type=int, default=512, help='批大小')
    parser.add_argument('--neighbor-size', type=int, default=10, help='邻居采样数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--contrast-weight', type=float, default=0.5, help='对比损失权重')
    parser.add_argument('--eval-every', type=int, default=10, help='每多少个epoch计算一次准确率')
    parser.add_argument('--load-path', help='模型加载路径，用于继续训练')
    parser.add_argument('node_embed_path', help='预训练顶点嵌入路径')
    parser.add_argument('pos_graph_path', help='正样本图路径')
    parser.add_argument('save_path', help='模型保存路径')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
