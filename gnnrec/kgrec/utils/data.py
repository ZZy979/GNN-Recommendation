import json

import dgl
import torch
from dgl.dataloading import Collator
from dgl.utils import to_dgl_context
from torch.utils.data import DataLoader


def iter_json(filename):
    """遍历每行一个JSON格式的文件。"""
    with open(filename, encoding='utf8') as f:
        for line in f:
            yield json.loads(line)


class TripletNodeCollator(Collator):

    def __init__(self, g, triplets, block_sampler, ntype):
        """用于OAGCSAuthorRankDataset数据集的NodeCollator

        :param g: DGLGraph 异构图
        :param triplets: tensor(N, 3) (t, ap, an)三元组
        :param block_sampler: BlockSampler 邻居采样器
        :param ntype: str 目标顶点类型
        """
        self.g = g
        self.triplets = triplets
        self.block_sampler = block_sampler
        self.ntype = ntype

    def collate(self, items):
        """根据三元组中的学者id构造子图

        :param items: List[tensor(3)] 一个批次的三元组
        :return: tensor(N_src), tensor(N_dst), List[DGLBlock] (input_nodes, output_nodes, blocks)
        """
        items = torch.stack(items, dim=0)
        seed_nodes = items[:, 1:].flatten().unique()
        blocks = self.block_sampler.sample_blocks(self.g, {self.ntype: seed_nodes})
        output_nodes = blocks[-1].dstnodes[self.ntype].data[dgl.NID]
        return items, output_nodes, blocks

    @property
    def dataset(self):
        return self.triplets


class TripletNodeDataLoader(DataLoader):

    def __init__(self, g, triplets, block_sampler, device=None, **kwargs):
        """用于OAGCSAuthorRankDataset数据集的NodeDataLoader

        :param g: DGLGraph 异构图
        :param triplets: tensor(N, 3) (t, ap, an)三元组
        :param block_sampler: BlockSampler 邻居采样器
        :param device: torch.device
        :param kwargs: DataLoader的其他参数
        """
        if device is None:
            device = g.device
        block_sampler.set_output_context(to_dgl_context(device))
        self.collator = TripletNodeCollator(g, triplets, block_sampler, 'author')
        super().__init__(triplets, collate_fn=self.collator.collate, **kwargs)
