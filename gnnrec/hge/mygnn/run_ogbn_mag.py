import argparse

import dgl
import torch
import torch.nn.functional as F
import torch.optim as optim
from dgl.dataloading import MultiLayerNeighborSampler, MultiLayerFullNeighborSampler
from torch.utils.data import DataLoader
from tqdm import tqdm

from gnnrec.config import DATA_DIR
from gnnrec.hge.mygnn.collator import PositiveSampleCollator
from gnnrec.hge.mygnn.model import HeCo
from gnnrec.hge.utils import set_random_seed, get_device, load_ogbn_mag, \
    load_pretrained_node_embed, accuracy


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)

    g, _, labels, num_classes, train_idx, val_idx, test_idx, evaluator = \
        load_ogbn_mag(DATA_DIR, True, device, False)
    g = g.cpu()
    load_pretrained_node_embed(g, args.node_embed_path)
    relations = [
        ('author', 'writes', 'paper'),
        ('paper', 'cites', 'paper'),
        ('field_of_study', 'has_topic_rev', 'paper')
    ]

    pos_g = dgl.load_graphs(args.pos_graph_path)[0][0]
    pos_g.ndata['feat'] = g.nodes['paper'].data['feat']
    pos = pos_g.edges()[0].view(pos_g.num_nodes(), -1)  # (N_p, T_pos) 每个paper顶点的正样本id

    collator = PositiveSampleCollator(g, MultiLayerNeighborSampler([None]), pos, 'paper')
    pos_collator = PositiveSampleCollator(pos_g, MultiLayerFullNeighborSampler(1), pos)
    train_loader = DataLoader(train_idx.cpu(), batch_size=args.batch_size)

    model = HeCo(
        {ntype: g.nodes[ntype].data['feat'].shape[1] for ntype in g.ntypes},
        args.num_hidden, num_classes, args.feat_drop, args.attn_drop,
        relations, args.tau, args.lambda_
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch in tqdm(train_loader):
            block = collator.collate(batch).to(device)
            pos_block = pos_collator.collate(batch).to(device)
            batch_pos = torch.zeros(pos_block.num_dst_nodes(), batch.shape[0], dtype=torch.int, device=device)
            batch_pos[pos_block.in_edges(torch.arange(batch.shape[0], device=device))] = 1
            contrast_loss, logits = model(
                block, block.srcdata['feat'], pos_block, pos_block.srcdata['feat'], batch_pos.t()
            )
            loss = contrast_loss + F.cross_entropy(logits, labels[batch].squeeze(dim=1))
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
        print('Epoch {:d} | Train Loss {:.4f}'.format(epoch, sum(losses) / len(losses)))
        if epoch % args.eval_every == 0 or epoch == args.epochs - 1:
            print('Train Acc {:.4f} | Val Acc {:.4f} | Test Acc {:.4f}'.format(*evaluate(
                model, g, pos, args.batch_size, device,
                labels, train_idx, val_idx, test_idx, evaluator
            )))
    if args.save_path:
        torch.save(model.cpu().state_dict(), args.save_path)
        print('模型已保存到', args.save_path)


def evaluate(model, g, pos, batch_size, device, labels, train_idx, val_idx, test_idx, evaluator):
    model.eval()
    embeds = model.get_embeds(g, g.ndata['feat'], pos, batch_size, device)
    train_acc = accuracy(embeds[train_idx], labels[train_idx], evaluator)
    val_acc = accuracy(embeds[val_idx], labels[val_idx], evaluator)
    test_acc = accuracy(embeds[test_idx], labels[test_idx], evaluator)
    return train_acc, val_acc, test_acc


def main():
    parser = argparse.ArgumentParser(description='HeCo模型 ogbn-mag数据集')
    parser.add_argument('--seed', type=int, default=0, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--num-hidden', type=int, default=64, help='隐藏层维数')
    parser.add_argument('--feat-drop', type=float, default=0.3, help='特征dropout')
    parser.add_argument('--attn-drop', type=float, default=0.5, help='注意力dropout')
    parser.add_argument('--tau', type=float, default=0.8, help='温度参数')
    parser.add_argument('--lambda', type=float, default=0.5, dest='lambda_', help='对比损失的平衡系数')
    parser.add_argument('--epochs', type=int, default=200, help='训练epoch数')
    parser.add_argument('--batch-size', type=int, default=4096, help='批大小')
    parser.add_argument('--lr', type=float, default=0.0008, help='学习率')
    parser.add_argument('--eval-every', type=int, default=10, help='每多少个epoch计算一次准确率')
    parser.add_argument('--save-path', help='模型保存路径')
    parser.add_argument('node_embed_path', help='预训练顶点嵌入路径')
    parser.add_argument('pos_graph_path', help='正样本图保存路径')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
