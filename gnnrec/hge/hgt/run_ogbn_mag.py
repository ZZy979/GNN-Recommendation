import argparse
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from tqdm import tqdm

from gnnrec.config import DATA_DIR
from gnnrec.hge.hgt.model import HGT
from gnnrec.hge.utils import set_random_seed, get_device, load_ogbn_mag, accuracy, \
    average_node_feat, load_pretrained_node_embed


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)

    g, _, labels, num_classes, train_idx, val_idx, test_idx, evaluator = \
        load_ogbn_mag(DATA_DIR, True, device)
    g = g.cpu()
    add_node_feat(g, args.node_feat, args.node_embed_path)

    sampler = MultiLayerNeighborSampler([args.neighbor_size] * args.num_layers)
    train_loader = NodeDataLoader(g, {'paper': train_idx}, sampler, batch_size=args.batch_size)
    val_loader = NodeDataLoader(g, {'paper': val_idx}, sampler, batch_size=args.batch_size)
    test_loader = NodeDataLoader(g, {'paper': test_idx}, sampler, batch_size=args.batch_size)

    model = HGT(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes},
        args.num_hidden, num_classes, g.ntypes, g.etypes, args.num_heads, args.num_layers,
        'paper', args.dropout
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), eps=1e-6)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, args.max_lr, epochs=args.epochs, steps_per_epoch=len(train_loader),
        pct_start=0.05, anneal_strategy='linear', final_div_factor=10.0
    )
    warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
    for epoch in range(args.epochs):
        model.train()
        logits, train_labels, losses = [], [], []
        for input_nodes, output_nodes, blocks in tqdm(train_loader):
            blocks = [b.to(device) for b in blocks]
            batch_labels = labels[output_nodes['paper']]
            batch_logits = model(blocks, blocks[0].srcdata['feat'])
            loss = F.cross_entropy(batch_logits, batch_labels.squeeze(dim=1))

            logits.append(batch_logits.detach().cpu())
            train_labels.append(batch_labels.detach().cpu())
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()

        train_acc = accuracy(torch.cat(logits, dim=0), torch.cat(train_labels, dim=0), evaluator)
        val_acc = evaluate(val_loader, device, model, labels, evaluator)
        print('Epoch {:d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f}'.format(
            epoch, torch.tensor(losses).mean().item(), train_acc, val_acc
        ))
    test_acc = evaluate(test_loader, device, model, labels, evaluator)
    print('Test Acc {:.4f}'.format(test_acc))


def add_node_feat(g, method, node_embed_path):
    if method == 'average':
        average_node_feat(g)
    elif method == 'pretrained':
        load_pretrained_node_embed(g, node_embed_path)


@torch.no_grad()
def evaluate(loader, device, model, labels, evaluator):
    model.eval()
    logits, eval_labels = [], []
    for input_nodes, output_nodes, blocks in loader:
        blocks = [b.to(device) for b in blocks]
        batch_labels = labels[output_nodes['paper']]
        batch_logits = model(blocks, blocks[0].srcdata['feat'])

        logits.append(batch_logits.detach().cpu())
        eval_labels.append(batch_labels.detach().cpu())
    return accuracy(torch.cat(logits, dim=0), torch.cat(eval_labels, dim=0), evaluator)


def main():
    parser = argparse.ArgumentParser(description='ogbn-mag数据集 HGT模型')
    parser.add_argument('--seed', type=int, default=1, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument(
        '--node-feat', choices=['average', 'pretrained'], default='average',
        help='如何获取无特征顶点的输入特征'
    )
    parser.add_argument('--node-embed-path', help='预训练顶点嵌入路径')
    parser.add_argument('--num-hidden', type=int, default=512, help='隐藏层维数')
    parser.add_argument('--num-heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--num-layers', type=int, default=2, help='层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--epochs', type=int, default=100, help='训练epoch数')
    parser.add_argument('--batch-size', type=int, default=4096, help='批大小')
    parser.add_argument('--neighbor-size', type=int, default=10, help='邻居采样数')
    parser.add_argument('--max-lr', type=float, default=5e-4, help='学习率上界')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
