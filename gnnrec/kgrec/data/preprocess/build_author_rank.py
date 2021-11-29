import argparse
import json
import math

import dgl
import numpy as np
from sklearn.metrics import ndcg_score
from tqdm import tqdm

from gnnrec.config import DATA_DIR
from gnnrec.hge.utils import add_reverse_edges
from gnnrec.kgrec.data import OAGCSDataset, OAGCoreDataset
from gnnrec.kgrec.utils import iter_json, load_author_rank, precision_at_k, recall_at_k, \
    calc_author_citation


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

    import django
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
    data = OAGCSDataset() if args.use_original_id else OAGCoreDataset()
    g = data[0]
    g.nodes['paper'].data['citation'] = g.nodes['paper'].data['citation'].float()
    if args.log_citation:
        g.nodes['paper'].data['citation'] = g.nodes['paper'].data['citation'].log1p()
    g.edges['writes'].data['order'] = g.edges['writes'].data['order'].float()
    apg = g['author', 'writes', 'paper']

    # 1.筛选论文数>=num_papers的领域
    field_in_degree, fid = g.in_degrees(g.nodes('field'), etype='has_field').sort(descending=True)
    fid = fid[field_in_degree >= args.num_papers].tolist()

    # 2.对每个领域关联的论文，构造学者-论文子图，通过论文引用数之和构造学者排名
    author_rank = {}
    for i in tqdm(fid):
        pid, _ = g.in_edges(i, etype='has_field')
        sg = add_reverse_edges(dgl.in_subgraph(apg, {'paper': pid}, relabel_nodes=True))
        author_citation = calc_author_citation(sg)
        _, idx = author_citation.topk(args.num_authors)
        aid = sg.nodes['author'].data[dgl.NID][idx]
        author_rank[i] = aid.tolist()

    suffix = '_original' if args.use_original_id else ''
    with open(DATA_DIR / f'rank/author_rank_train{suffix}.json', 'w') as f:
        json.dump(author_rank, f)
        print('结果已保存到', f.name)


def evaluate_ground_truth(args):
    """评估ground truth训练集的质量。"""
    author_rank_val = load_author_rank('val')
    author_rank_train = load_author_rank('train_original')
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
        author_rank_val[f] = [a for a in author_rank_val[f] if a != -1]
        for r, a in enumerate(author_rank_train[f]):
            scores[i, a] = len(author_rank_train[f]) - r

    for k in (100, 50, 20, 10, 5):
        print('nDCG@{0}={1:.4f}\tPrecision@{0}={2:.4f}\tRecall@{0}={3:.4f}'.format(
            k, ndcg_score(true_relevance, scores, k=k, ignore_ties=True),
            sum(precision_at_k(author_rank_val[f], author_rank_train[f], k) for f in fields) / len(fields),
            sum(recall_at_k(author_rank_val[f], author_rank_train[f], k) for f in fields) / len(fields)
        ))


def main():
    parser = argparse.ArgumentParser(description='基于oag-cs数据集构造学者排名数据集')
    subparsers = parser.add_subparsers()

    build_val_parser = subparsers.add_parser('build-val', help='构造学者排名验证集')
    build_val_parser.add_argument('--use-field-name', action='store_true', help='使用领域名称（用于调试）')
    build_val_parser.set_defaults(func=build_ground_truth_valid)

    build_train_parser = subparsers.add_parser('build-train', help='构造学者排名训练集')
    build_train_parser.add_argument('--num-papers', type=int, default=5000, help='筛选领域的论文数阈值')
    build_train_parser.add_argument('--num-authors', type=int, default=100, help='每个领域取top k的学者数量')
    build_train_parser.add_argument('--log-citation', action='store_true', help='论文引用数取对数')
    build_train_parser.add_argument('--use-original-id', action='store_true', help='使用原始id')
    build_train_parser.set_defaults(func=build_ground_truth_train)

    evaluate_parser = subparsers.add_parser('eval', help='评估ground truth训练集的质量')
    evaluate_parser.set_defaults(func=evaluate_ground_truth)

    args = parser.parse_args()
    print(args)
    args.func(args)


if __name__ == '__main__':
    main()
