import argparse
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from tqdm import tqdm

from gnnrec.hge.hgt.model import HGT
from gnnrec.hge.utils import set_random_seed, get_device, load_data, add_node_feat, evaluate, \
    METRICS_STR


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    data, g, _, labels, predict_ntype, train_idx, val_idx, test_idx, evaluator = \
        load_data(args.dataset, device)
    add_node_feat(g, args.node_feat, args.node_embed_path)

    sampler = MultiLayerNeighborSampler([args.neighbor_size] * args.num_layers)
    train_loader = NodeDataLoader(g, {predict_ntype: train_idx}, sampler, device=device, batch_size=args.batch_size)
    loader = NodeDataLoader(g, {predict_ntype: g.nodes(predict_ntype)}, sampler, device=device, batch_size=args.batch_size)

    model = HGT(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes},
        args.num_hidden, data.num_classes, args.num_heads, g.ntypes, g.canonical_etypes,
        predict_ntype, args.num_layers, args.dropout
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), eps=1e-6)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, args.max_lr, epochs=args.epochs, steps_per_epoch=len(train_loader),
        pct_start=0.05, anneal_strategy='linear', final_div_factor=10.0
    )
    warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for input_nodes, output_nodes, blocks in tqdm(train_loader):
            batch_logits = model(blocks, blocks[0].srcdata['feat'])
            batch_labels = labels[output_nodes[predict_ntype]]
            loss = F.cross_entropy(batch_logits, batch_labels)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()
        print('Epoch {:d} | Loss {:.4f}'.format(epoch, sum(losses) / len(losses)))
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            print(METRICS_STR.format(*evaluate(
                model, loader, g, labels, data.num_classes, predict_ntype,
                train_idx, val_idx, test_idx, evaluator
            )))
    if args.save_path:
        torch.save(model.cpu().state_dict(), args.save_path)
        print('模型已保存到', args.save_path)


def main():
    parser = argparse.ArgumentParser(description='训练HGT模型')
    parser.add_argument('--seed', type=int, default=1, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--dataset', choices=['ogbn-mag', 'oag-cs-venue'], default='ogbn-mag', help='数据集')
    parser.add_argument(
        '--node-feat', choices=['average', 'pretrained'], default='pretrained',
        help='如何获取无特征顶点的输入特征'
    )
    parser.add_argument('--node-embed-path', help='预训练顶点嵌入路径')
    parser.add_argument('--num-hidden', type=int, default=512, help='隐藏层维数')
    parser.add_argument('--num-heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--num-layers', type=int, default=2, help='层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--epochs', type=int, default=100, help='训练epoch数')
    parser.add_argument('--batch-size', type=int, default=2048, help='批大小')
    parser.add_argument('--neighbor-size', type=int, default=10, help='邻居采样数')
    parser.add_argument('--max-lr', type=float, default=5e-4, help='学习率上界')
    parser.add_argument('--eval-every', type=int, default=10, help='每多少个epoch计算一次准确率')
    parser.add_argument('--save-path', help='模型保存路径')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
