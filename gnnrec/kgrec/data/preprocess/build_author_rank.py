import argparse
import json

import dgl
import dgl.function as fn
from tqdm import tqdm

import gnnrec.kgrec.recall as recall
from gnnrec.hge.utils import set_random_seed
from gnnrec.kgrec.data import OAGCSDataset


def build_author_rank(args):
    """对每个领域构造引用数最多的学者排名"""
    data = OAGCSDataset()
    g = data[0]
    apg = dgl.reverse(g['author', 'writes', 'paper'], copy_ndata=False)
    apg.nodes['paper'].data['c'] = g.in_degrees(g.nodes('paper'), etype='cites').float().log1p()

    # 1.筛选论文数>=num_papers的领域
    field_in_degree, fid = g.in_degrees(g.nodes('field'), etype='has_field').sort(descending=True)
    fid = fid[field_in_degree >= args.num_papers].tolist()

    # 2.对每个领域召回论文，构造学者-论文子图，通过论文引用数之和对学者排名
    with open(args.raw_field_file, encoding='utf8') as f:
        fields = [json.loads(line)['name'] for line in f]
    ctx = recall.get_context(args.paper_embeds_file, args.scibert_model_file)
    author_rank = {}
    for i in tqdm(fid):
        _, pid = recall.recall(ctx, fields[i], args.num_recall)
        sg = dgl.out_subgraph(apg, {'paper': pid}, relabel_nodes=True)
        sg.update_all(fn.copy_u('c', 'm'), fn.sum('m', 'c'))
        author_citation = sg.nodes['author'].data['c']
        _, aid = author_citation.topk(args.num_authors)
        aid = sg.nodes['author'].data[dgl.NID][aid]
        author_rank[fields[i]] = aid.tolist()

    with open(args.author_rank_save_file, 'w') as f:
        json.dump(author_rank, f)
    print('结果已保存到', args.author_rank_save_file)


def sample_triplets(args):
    set_random_seed(args.seed)
    # 三元组：(t, ap, an)，表示对于领域t，学者ap的排名在an之前
    triplets = []


def main():
    parser = argparse.ArgumentParser(description='基于oag-cs数据集构造学者排名数据集')
    subparsers = parser.add_subparsers()
    
    build_parser = subparsers.add_parser('build', help='构造学者排名')
    build_parser.add_argument('--num-papers', type=int, default=2000, help='筛选领域的论文数阈值')
    build_parser.add_argument('--num-recall', type=int, default=1000, help='每个领域召回论文的数量')
    build_parser.add_argument('--num-authors', type=int, default=100, help='每个领域取top k的学者数量')
    build_parser.add_argument('raw_field_file', help='原始领域数据文件')
    build_parser.add_argument('paper_embeds_file', help='预训练的论文标题向量文件')
    build_parser.add_argument('scibert_model_file', help='微调后的SciBERT模型文件')
    build_parser.add_argument('author_rank_save_file', help='学者排名结果保存文件')
    build_parser.set_defaults(func=build_author_rank)

    sample_parser = subparsers.add_parser('sample', help='采样三元组')
    sample_parser.add_argument('--seed', type=int, default=0, help='随机数种子')
    sample_parser.add_argument('author_rank_file', help='学者排名结果保存文件')
    sample_parser.add_argument('triplets_save_file', help='三元组保存文件')
    sample_parser.set_defaults(func=sample_triplets)

    args = parser.parse_args()
    print(args)
    args.func(args)


if __name__ == '__main__':
    main()
