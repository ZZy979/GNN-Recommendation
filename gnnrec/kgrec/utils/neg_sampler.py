import torch
from dgl.dataloading.negative_sampler import _BaseNegativeSampler


class RecallNegativeSampler(_BaseNegativeSampler):

    def __init__(self, k, g, field_ids):
        """用于论文召回的负采样器

        对于每个领域，仅从该领域关联的论文中采样负样本

        :param k: int 每条边采样的负样本边数量
        :param g: DGLGraph 异构图
        :param field_ids: List[int] 训练集领域id列表
        """
        self.k = k
        self.out_edges = {f: g.out_edges(f, etype='has_field_rev')[1] for f in field_ids}
        self.out_degrees = {f: g.out_degrees(f, etype='has_field_rev') for f in field_ids}

    def _generate(self, g, eids, canonical_etype):
        assert canonical_etype == ('field', 'has_field_rev', 'paper'), '该采样器仅用于F-P边'
        src, _ = g.find_edges(eids, etype=canonical_etype)
        dst = torch.cat([
            self.out_edges[f][torch.randint(self.out_degrees[f], (self.k,))]
            for f in src.tolist()
        ])
        src = src.repeat_interleave(self.k)
        return src, dst
