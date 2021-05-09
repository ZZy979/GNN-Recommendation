import torch.nn as nn

from gnnrec.hge.hetgnn.model import ContentAggregation
from gnnrec.hge.hgconv.model import HGConvLayer


class MyGNN(nn.Module):

    def __init__(
            self, in_dim, hidden_dim, out_dim, num_heads, g, predict_ntype,
            num_layers, dropout=0.0):
        """My GNN

        :param in_dim: int 输入特征维数
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param g: DGLGraph 异构图
        :param predict_ntype: str 待预测顶点类型
        :param num_layers: int 层数
        :param dropout: float, optional Dropout概率，默认为0
        """
        super().__init__()
        self.predict_ntype = predict_ntype
        self.content_aggs = nn.ModuleDict({
            ntype: ContentAggregation(in_dim, num_heads * hidden_dim) for ntype in g.ntypes
        })
        self.layers = nn.ModuleList([
            HGConvLayer(num_heads * hidden_dim, hidden_dim, num_heads, g, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(num_heads * hidden_dim, out_dim)

    def forward(self, blocks, feats):
        """
        :param blocks: List[DGLBlock]
        :param feats: Dict[str, tensor(N_i, C_i, d_in)] 顶点类型到输入特征的映射
        :return: tensor(N_dst, d_out) 待预测顶点的输出特征
        """
        feats = {ntype: self.content_aggs[ntype](feats[ntype]) for ntype in feats}
        for i in range(len(self.layers)):
            feats = self.layers[i](blocks[i], feats)  # {ntype: tensor(N_i, K*d_hid)}
        return self.classifier(feats[self.predict_ntype])
