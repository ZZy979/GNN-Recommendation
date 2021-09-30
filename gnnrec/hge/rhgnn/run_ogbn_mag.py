import argparse
import warnings

import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader
from gensim.models import Word2Vec
from tqdm import tqdm

from gnnrec.config import DATA_DIR
from gnnrec.hge.rhgnn.model import RHGNN
from gnnrec.hge.utils import set_random_seed, get_device, load_ogbn_mag, accuracy


def load_pretrained_node_embed(g, path):
    model = Word2Vec.load(path)
    paper_embed = torch.from_numpy(model.wv[[f'paper_{i}' for i in range(g.num_nodes('paper'))]]).to(g.device)
    g.nodes['paper'].data['feat'] = torch.cat([g.nodes['paper'].data['feat'], paper_embed], dim=1)
    for ntype in ('author', 'field_of_study', 'institution'):
        g.nodes[ntype].data['feat'] = torch.from_numpy(
            model.wv[[f'{ntype}_{i}' for i in range(g.num_nodes(ntype))]]
        ).to(g.device)


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)
    g, _, labels, num_classes, train_idx, val_idx, test_idx, evaluator = \
        load_ogbn_mag(DATA_DIR, True, device)
    load_pretrained_node_embed(g, args.node_embed_path)

    sampler = MultiLayerNeighborSampler(list(range(args.neighbor_size, args.neighbor_size + args.num_layers)))
    train_loader = NodeDataLoader(g, {'paper': train_idx}, sampler, device=device, batch_size=args.batch_size)
    val_loader = NodeDataLoader(g, {'paper': val_idx}, sampler, device=device, batch_size=args.batch_size)
    test_loader = NodeDataLoader(g, {'paper': test_idx}, sampler, device=device, batch_size=args.batch_size)

    model = RHGNN(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes},
        args.num_hidden, num_classes, args.num_rel_hidden, args.num_rel_hidden, args.num_heads,
        g.ntypes, g.canonical_etypes, 'paper', args.num_layers, args.dropout, residual=args.residual
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * args.epochs, eta_min=args.lr / 100
    )
    warnings.filterwarnings('ignore', 'Setting attributes on ParameterDict is not supported')
    for epoch in range(args.epochs):
        model.train()
        logits, train_labels, losses = [], [], []
        for input_nodes, output_nodes, blocks in tqdm(train_loader):
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
        test_acc = evaluate(test_loader, device, model, labels, evaluator)
        print('Epoch {:d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f} | Test Acc {:.4f}'.format(
            epoch, torch.tensor(losses).mean().item(), train_acc, val_acc, test_acc
        ))
    test_acc = evaluate(test_loader, device, model, labels, evaluator)
    print('Test Acc {:.4f}'.format(test_acc))
    if args.save_path:
        torch.save(model.cpu().state_dict(), args.save_path)
        print('模型已保存到', args.save_path)


@torch.no_grad()
def evaluate(loader, device, model, labels, evaluator):
    model.eval()
    logits, eval_labels = [], []
    for input_nodes, output_nodes, blocks in loader:
        batch_labels = labels[output_nodes['paper']]
        batch_logits = model(blocks, blocks[0].srcdata['feat'])

        logits.append(batch_logits.detach().cpu())
        eval_labels.append(batch_labels.detach().cpu())
    return accuracy(torch.cat(logits, dim=0), torch.cat(eval_labels, dim=0), evaluator)


def main():
    parser = argparse.ArgumentParser(description='R-HGNN模型 ogbn-mag数据集')
    parser.add_argument('--seed', type=int, default=0, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--num-hidden', type=int, default=64, help='隐藏层维数')
    parser.add_argument('--num-rel-hidden', type=int, default=8, help='关系表示的隐藏层维数')
    parser.add_argument('--num-heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--num-layers', type=int, default=2, help='层数')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout概率')
    parser.add_argument('--no-residual', action='store_false', help='不使用残差连接', dest='residual')
    parser.add_argument('--epochs', type=int, default=200, help='训练epoch数')
    parser.add_argument('--batch-size', type=int, default=1024, help='批大小')
    parser.add_argument('--neighbor-size', type=int, default=10, help='邻居采样数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='权重衰减')
    parser.add_argument('--save-path', help='模型保存路径')
    parser.add_argument('node_embed_path', help='预训练顶点嵌入路径')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
