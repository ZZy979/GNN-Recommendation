import json
import math

import dgl
import numpy as np
import torch
from dgl.dataloading import Collator

__all__ = [
    'iter_json', 'TripletNodeCollator', 'calc_author_citation',
    'load_author_rank', 'calc_true_relevance', 'load_rank_data', 'recall_paper'
]


def iter_json(filename):
    """遍历每行一个JSON格式的文件。"""
    with open(filename, encoding='utf8') as f:
        for line in f:
            yield json.loads(line)


class TripletNodeCollator(Collator):

    def __init__(self, g, block_sampler):
        """根据三元组中的学者id构造子图的NodeCollator

        :param g: DGLGraph 异构图
        :param block_sampler: BlockSampler 邻居采样器
        """
        self.g = g
        self.block_sampler = block_sampler

    def collate(self, items):
        """根据三元组中的学者id构造子图

        :param items: tensor(B, 3) 一个批次的三元组
        :return: (tensor(N_dst), List[DGLBlock]) 学者顶点id和多层block
        """
        seed_nodes = items[:, 1:].flatten().unique()
        blocks = self.block_sampler.sample_blocks(self.g, {'author': seed_nodes})
        output_nodes = blocks[-1].dstnodes['author'].data[dgl.NID]
        return output_nodes, blocks

    @property
    def dataset(self):
        return None


def calc_author_citation(g):
    """使用论文引用数加权求和计算学者引用数

    :param g: DGLGraph 学者-论文二分图
    :return: tensor(N_author) 学者引用数
    """
    import dgl.function as fn
    from dgl.ops import edge_softmax
    with g.local_scope():
        # 第k作者的权重为1/k，最后一个视为通讯作者，权重为1/2
        g.edges['writes'].data['w'] = 1.0 / g.edges['writes'].data['order']
        g.update_all(fn.copy_e('w', 'w'), fn.min('w', 'mw'), etype='writes')
        g.apply_edges(fn.copy_u('mw', 'mw'), etype='writes_rev')
        w, mw = g.edges['writes'].data.pop('w'), g.edges['writes_rev'].data.pop('mw')
        w[w == mw] = 0.5

        # 每篇论文所有作者的权重归一化，每个学者所有论文的引用数加权求和
        p = edge_softmax(g['author', 'writes', 'paper'], torch.log(w).unsqueeze(dim=1))
        g.edges['writes_rev'].data['p'] = p.squeeze(dim=1)
        g.update_all(fn.u_mul_e('citation', 'p', 'c'), fn.sum('c', 'c'), etype='writes_rev')
        return g.nodes['author'].data['c']


def load_author_rank(train=True):
    """加载领域学者排名数据集

    :param train: bool, optional True - 训练集，False - 验证集(AI 2000)
    :return: Dict[int, List[int]] {field_id: [author_id]}
    """
    from ...config import DATA_DIR
    split = 'train' if train else 'val'
    with open(DATA_DIR / f'rank/author_rank_{split}.json') as f:
        return {int(k): v for k, v in json.load(f).items()}


def calc_true_relevance(author_rank, field_ids, num_authors):
    """计算领域-学者真实相关性得分

    :param author_rank: Dict[int, List[int]] {field_id: [author_id]}
    :param field_ids: List[int] 领域id列表
    :param num_authors: int 学者数量
    :return: ndarray(N_field, N_author)
    """
    true_relevance = np.zeros((len(field_ids), num_authors), dtype=np.int32)
    for i, f in enumerate(field_ids):
        for r, a in enumerate(author_rank[f]):
            true_relevance[i, a] = math.ceil((len(author_rank[f]) - r) / 10)
    return true_relevance


def load_rank_data(device='cpu'):
    """加载学者排名数据

    :param device: torch.device
    :return: DGLGraph, Dict[int, List[int]], List[int], ndarray(N_field, N_author)
      异构图，真实学者排名，领域列表，领域-学者真实相关性得分
    """
    from ...hge.utils import add_reverse_edges
    from ..data import OAGCoreDataset
    g = add_reverse_edges(OAGCoreDataset()[0]).to(device)
    author_rank = load_author_rank()
    field_ids = list(author_rank)
    true_relevance = calc_true_relevance(author_rank, field_ids, g.num_nodes('author'))
    return g, author_rank, field_ids, true_relevance


def recall_paper(g, field_ids, num_recall):
    """预先计算论文召回

    :param g: DGLGraph 异构图
    :param field_ids: List[int] 目标领域id
    :param num_recall: 每个领域召回的论文数
    :return: Dict[int, List[int]] {field_id: [paper_id]}
    """
    field_paper = {}
    for f in field_ids:
        pid = g.in_edges(f, etype='has_field')[0]
        field_paper[f] = pid[g.nodes['paper'].data['citation'][pid].topk(num_recall)[1]].tolist()
    return field_paper
