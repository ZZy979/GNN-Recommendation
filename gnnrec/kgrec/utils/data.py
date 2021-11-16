import json

import dgl
from dgl.dataloading import Collator


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
