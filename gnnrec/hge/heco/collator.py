import torch
from dgl.dataloading import Collator


class PositiveSampleCollator(Collator):

    def __init__(self, g, block_sampler, pos, ntype=None):
        """用于HeCo模型的collator

        对于每个batch的目标顶点，添加其正样本顶点作为目标顶点并生成block

        :param g: DGLGraph 原图
        :param block_sampler: BlockSampler 邻居采样器
        :param pos: tensor(N, T_pos) 每个顶点的正样本id
        :param ntype: str, optional 如果g是异构图则需要指定目标顶点类型
        """
        if not g.is_homogeneous and not ntype:
            raise ValueError('异构图必须指定目标顶点类型')
        self.g = g
        self.block_sampler = block_sampler
        self.pos = pos
        self.ntype = ntype

    @property
    def dataset(self):
        return None

    def collate(self, items):
        """生成以items及其正样本为目标顶点的block

        真正的目标顶点items一定出现在block顶点集合的开头，即output_nodes[:items.shape[0]] == items

        :param items: tensor(B) 一个batch的目标顶点id
        :return: List[DGLBlock]
        """
        pos_samples = self.pos[items].flatten()  # (B, T_pos) -> (B*T_pos,)
        added = list(set(pos_samples.tolist()) - set(items.tolist()))
        output_nodes = torch.cat([items, torch.tensor(added)])
        if not self.g.is_homogeneous:
            output_nodes = {self.ntype: output_nodes}
        return self.block_sampler.sample_blocks(self.g, output_nodes)
