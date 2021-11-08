import argparse
import json
import math
import random

import dgl
import dgl.function as fn
import django
import numpy as np
import torch
from dgl.ops import edge_softmax
from sklearn.metrics import ndcg_score
from tqdm import tqdm

from gnnrec.config import DATA_DIR
from gnnrec.hge.utils import set_random_seed, add_reverse_edges
from gnnrec.kgrec.data import OAGCSDataset
from gnnrec.kgrec.utils import iter_json, precision_at_k, recall_at_k


def build_ground_truth_valid(args):
    """从AI 2000抓取的学者排名数据匹配学者id，作为学者排名ground truth验证集。"""
    field_map = {
        'AAAI/IJCAI': 'artificial intelligence',
        'Machine Learning': 'machine learning',
        'Computer Vision': 'computer vision',
        'Natural Language Processing': 'natural language processing',
        'Robotics': 'robotics',
        'Knowledge Engineering': 'knowledge engineering',
        'Speech Recognition': 'speech recognition',
        'Data Mining': 'data mining',
        'Information Retrieval and Recommendation': 'information retrieval',
        'Database': 'database',
        'Human-Computer Interaction': 'human computer interaction',
        'Computer Graphics': 'computer graphics',
        'Multimedia': 'multimedia',
        'Visualization': 'visualization',
        'Security and Privacy': 'security privacy',
        'Computer Networking': 'computer network',
        'Computer Systems': 'operating system',
        'Theory': 'theory',
        'Chip Technology': 'chip',
        'Internet of Things': 'internet of things',
    }
    with open(DATA_DIR / 'rank/ai2000.json', encoding='utf8') as f:
        ai2000_author_rank = json.load(f)

    django.setup()
    from rank.models import Author

    author_rank = {}
    for field, scholars in ai2000_author_rank.items():
        aid = []
        for s in scholars:
            qs = Author.objects.filter(name=s['name'], institution__name=s['org']).order_by('-n_citation')
            if qs.exists():
                aid.append(qs[0].id)
            else:
                qs = Author.objects.filter(name=s['name']).order_by('-n_citation')
                aid.append(qs[0].id if qs.exists() else -1)
        author_rank[field_map[field]] = aid
    if not args.use_field_name:
        field2id = {f['name']: i for i, f in enumerate(iter_json(DATA_DIR / 'oag/cs/mag_fields.txt'))}
        author_rank = {field2id[f]: aid for f, aid in author_rank.items()}

    with open(DATA_DIR / 'rank/author_rank_val.json', 'w') as f:
        json.dump(author_rank, f)
        print('结果已保存到', f.name)


def build_ground_truth_train(args):
    """根据某个领域的论文引用数加权求和构造学者排名，作为ground truth训练集。"""
    data = OAGCSDataset()
    g = data[0]
    g.nodes['paper'].data['citation'] = g.nodes['paper'].data['citation'].float().log1p()
    g.edges['writes'].data['order'] = g.edges['writes'].data['order'].float()
    apg = g['author', 'writes', 'paper']

    # 1.筛选论文数>=num_papers的领域
    field_in_degree, fid = g.in_degrees(g.nodes('field'), etype='has_field').sort(descending=True)
    fid = fid[field_in_degree >= args.num_papers].tolist()

    # 2.对每个领域召回论文，构造学者-论文子图，通过论文引用数之和对学者排名
    author_rank = {}
    for i in tqdm(fid):
        pid, _ = g.in_edges(i, etype='has_field')
        sg = add_reverse_edges(dgl.in_subgraph(apg, {'paper': pid}, relabel_nodes=True))

        # 第k作者的权重为1/k，最后一个视为通讯作者，权重为1/2
        sg.edges['writes'].data['w'] = 1.0 / sg.edges['writes'].data['order']
        sg.update_all(fn.copy_e('w', 'w'), fn.min('w', 'mw'), etype='writes')
        sg.apply_edges(fn.copy_u('mw', 'mw'), etype='writes_rev')
        w, mw = sg.edges['writes'].data.pop('w'), sg.edges['writes_rev'].data.pop('mw')
        w[w == mw] = 0.5

        # 每篇论文所有作者的权重归一化，每个学者所有论文的引用数加权求和
        p = edge_softmax(sg['author', 'writes', 'paper'], torch.log(w).unsqueeze(dim=1))
        sg.edges['writes_rev'].data['p'] = p.squeeze(dim=1)
        sg.update_all(fn.u_mul_e('citation', 'p', 'c'), fn.sum('c', 'c'), etype='writes_rev')
        author_citation = sg.nodes['author'].data['c']

        _, aid = author_citation.topk(args.num_authors)
        aid = sg.nodes['author'].data[dgl.NID][aid]
        author_rank[i] = aid.tolist()
    if args.use_field_name:
        fields = [f['name'] for f in iter_json(DATA_DIR / 'oag/cs/mag_fields.txt')]
        author_rank = {fields[i]: aid for i, aid in author_rank.items()}

    with open(DATA_DIR / 'rank/author_rank_train.json', 'w') as f:
        json.dump(author_rank, f)
        print('结果已保存到', f.name)


def evaluate_ground_truth(args):
    """评估ground truth训练集的质量。"""
    with open(DATA_DIR / 'rank/author_rank_val.json') as f:
        author_rank_val = json.load(f)
    with open(DATA_DIR / 'rank/author_rank_train.json') as f:
        author_rank_train = json.load(f)
    fields = list(set(author_rank_val) & set(author_rank_train))
    author_rank_val = {k: v for k, v in author_rank_val.items() if k in fields}
    author_rank_train = {k: v for k, v in author_rank_train.items() if k in fields}

    num_authors = OAGCSDataset()[0].num_nodes('author')
    true_relevance = np.zeros((len(fields), num_authors), dtype=np.int32)
    scores = np.zeros_like(true_relevance)
    for i, f in enumerate(fields):
        for r, a in enumerate(author_rank_val[f]):
            if a != -1:
                true_relevance[i, a] = math.ceil((100 - r) / 10)
        for r, a in enumerate(author_rank_train[f]):
            scores[i, a] = len(author_rank_train[f]) - r

    for k in (100, 50, 20, 10, 5):
        print('nDGC@{0}={1:.4f}\tPrecision@{0}={2:.4f}\tRecall@{0}={3:.4f}'.format(
            k, ndcg_score(true_relevance, scores, k=k, ignore_ties=True),
            sum(precision_at_k(author_rank_val[f], author_rank_train[f], k) for f in fields) / len(fields),
            sum(recall_at_k(author_rank_val[f], author_rank_train[f], k) for f in fields) / len(fields)
        ))


def sample_triplets(args):
    set_random_seed(args.seed)
    with open(DATA_DIR / 'rank/author_rank_train.json') as f:
        author_rank = json.load(f)

    # 三元组：(t, ap, an)，表示对于领域t，学者ap的排名在an之前
    triplets = []
    for fid, aid in author_rank.items():
        fid = int(fid)
        n = len(aid)
        easy_margin, hard_margin = int(n * args.easy_margin), int(n * args.hard_margin)
        num_triplets = min(args.max_num, 2 * n - easy_margin - hard_margin)
        num_hard = int(num_triplets * args.hard_ratio)
        num_easy = num_triplets - num_hard
        triplets.extend(
            (fid, aid[i], aid[i + easy_margin])
            for i in random.sample(range(n - easy_margin), num_easy)
        )
        triplets.extend(
            (fid, aid[i], aid[i + hard_margin])
            for i in random.sample(range(n - hard_margin), num_hard)
        )

    with open(DATA_DIR / 'rank/author_rank_triplets.txt', 'w') as f:
        for t, ap, an in triplets:
            f.write(f'{t} {ap} {an}\n')
        print('结果已保存到', f.name)


def main():
    parser = argparse.ArgumentParser(description='基于oag-cs数据集构造学者排名数据集')
    subparsers = parser.add_subparsers()

    build_val_parser = subparsers.add_parser('build-val', help='构造学者排名验证集')
    build_val_parser.add_argument('--use-field-name', action='store_true', help='使用领域名称（用于调试）')
    build_val_parser.set_defaults(func=build_ground_truth_valid)

    build_train_parser = subparsers.add_parser('build-train', help='构造学者排名训练集')
    build_train_parser.add_argument('--num-papers', type=int, default=5000, help='筛选领域的论文数阈值')
    build_train_parser.add_argument('--num-authors', type=int, default=100, help='每个领域取top k的学者数量')
    build_train_parser.add_argument('--use-field-name', action='store_true', help='使用领域名称（用于调试）')
    build_train_parser.set_defaults(func=build_ground_truth_train)

    evaluate_parser = subparsers.add_parser('eval', help='评估ground truth训练集的质量')
    evaluate_parser.set_defaults(func=evaluate_ground_truth)

    sample_parser = subparsers.add_parser('sample', help='采样三元组')
    sample_parser.add_argument('--seed', type=int, default=0, help='随机数种子')
    sample_parser.add_argument('--max-num', type=int, default=100, help='每个领域采样三元组最大数量')
    sample_parser.add_argument('--easy-margin', type=float, default=0.2, help='简单样本间隔（百分比）')
    sample_parser.add_argument('--hard-margin', type=float, default=0.05, help='困难样本间隔（百分比）')
    sample_parser.add_argument('--hard-ratio', type=float, default=0.5, help='困难样本比例')
    sample_parser.set_defaults(func=sample_triplets)

    args = parser.parse_args()
    print(args)
    args.func(args)


if __name__ == '__main__':
    main()
