import torch
from dgl.dataloading import MultiLayerNeighborSampler


class PositiveSampler(MultiLayerNeighborSampler):

    def __init__(self, fanouts, pos):
        """用于HeCo模型的邻居采样器

        对于每个batch的目标顶点，将其正样本添加到目标顶点并生成block

        :param fanouts: 每层的邻居采样数（见MultiLayerNeighborSampler）
        :param pos: tensor(N, T_pos) 每个顶点的正样本id，N是目标顶点数
        """
        super().__init__(fanouts)
        self.pos = pos

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        # 如果g是异构图则seed_nodes是字典，应当只有目标顶点类型
        if not g.is_homogeneous:
            assert len(seed_nodes) == 1, 'PositiveSampler: 异构图只能指定目标顶点这一种类型'
            ntype, seed_nodes = next(iter(seed_nodes.items()))
        pos_samples = self.pos[seed_nodes].flatten()  # (B, T_pos) -> (B*T_pos,)
        added = list(set(pos_samples.tolist()) - set(seed_nodes.tolist()))
        seed_nodes = torch.cat([seed_nodes, torch.tensor(added, device=seed_nodes.device)])
        if not g.is_homogeneous:
            seed_nodes = {ntype: seed_nodes}
        return super().sample_blocks(g, seed_nodes, exclude_eids)
