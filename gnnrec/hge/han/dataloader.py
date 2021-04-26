import dgl
from dgl.dataloading import Collator, MultiLayerNeighborSampler

from gnnrec.hge.utils import metapath_adj


class MetapathNodeCollator(Collator):

    def __init__(self, g, nids, metapath, neighbor_size=-1):
        """用于生成ABA型元路径的邻居构成的同构子图的collator

        :param g: DGLGraph 异构图，应当只包含A-B和B-A两种边类型
        :param nids: tensor A类型目标顶点id
        :param metapath: (str, str) 元路径(A-B, B-A)
        :param neighbor_size: int, optional A-B邻居采样数量，默认为-1（不采样）
        """
        self.g = g
        self._dataset = nids
        self.metapath = metapath
        ab, ba = metapath
        self.dtype = g.to_canonical_etype(ba)[2]
        self.block_sampler = MultiLayerNeighborSampler([{ab: neighbor_size, ba: -1}, None])

    @property
    def dataset(self):
        return self._dataset

    def collate(self, items):
        """计算指定顶点及其基于元路径的邻居构成的block

        :param items: tensor或List[int] 目标顶点id
        :return: DGLBlock
        """
        bipartite_block = self.block_sampler.sample_blocks(self.g, {self.dtype: items})[0]
        adj = metapath_adj(bipartite_block, self.metapath)  # (N_src, N_dst)
        mg = dgl.graph(adj.nonzero(), num_nodes=bipartite_block.num_nodes(self.dtype))
        block = dgl.to_block(mg)
        block.srcdata.update(bipartite_block.srcnodes[self.dtype].data)
        block.dstdata.update(bipartite_block.dstnodes[self.dtype].data)
        return block
