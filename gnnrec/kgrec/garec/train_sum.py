import argparse

import dgl
import dgl.function as fn
import torch

from gnnrec.kgrec.utils import load_rank_data, recall_paper, calc_metrics, METRICS_STR


def train(args):
    g, author_rank, field_ids, true_relevance = load_rank_data()
    field_paper = recall_paper(g, field_ids, args.num_recall)
    print(METRICS_STR.format(*evaluate(
        g, field_ids, author_rank, true_relevance, field_paper
    )))


@torch.no_grad()
def evaluate(g, field_ids, author_rank, true_relevance, field_paper):
    predict_rank = {}
    field_feat = g.nodes['field'].data['feat']
    apg = g['paper', 'writes_rev', 'author']
    for i, f in enumerate(field_ids):
        pid = field_paper[f]
        paper_score = torch.matmul(g.nodes['paper'].data['feat'][pid], field_feat[f])
        sg = dgl.out_subgraph(apg, {'paper': pid}, relabel_nodes=True)
        sg.nodes['paper'].data['score'] = paper_score
        sg.update_all(fn.copy_u('score', 's'), fn.sum('s', 's'))
        predict_rank[f] = (sg.nodes['author'].data[dgl.NID], sg.nodes['author'].data['s'])
    return calc_metrics(field_ids, author_rank, true_relevance, predict_rank)


def main():
    parser = argparse.ArgumentParser(description='通过论文得分求和计算学者排名')
    parser.add_argument('--num-recall', type=int, default=200, help='每个领域召回论文的数量')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
