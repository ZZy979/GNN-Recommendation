import torch
import torch.nn as nn

from gnnrec.hge.hgconv.model import HGConvLayer
from gnnrec.hge.hgt.model import HGTAttention


class MyGNNLayer(HGConvLayer):

    def __init__(self, in_dim, out_dim, num_heads, ntypes, etypes, dropout=0.0):
        """MyGNN层

        :param in_dim: int 输入特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param ntypes: List[str] 顶点类型列表
        :param etypes: List[(str, str, str)] 规范边类型列表
        :param dropout: float, optional Dropout概率，默认为0
        """
        super().__init__(in_dim, out_dim, num_heads, ntypes, etypes, dropout)
        del self.micro_conv
        k_linear = {ntype: nn.Linear(in_dim, num_heads * out_dim) for ntype in ntypes}
        q_linear = {ntype: nn.Linear(in_dim, num_heads * out_dim) for ntype in ntypes}
        v_linear = {ntype: nn.Linear(in_dim, num_heads * out_dim) for ntype in ntypes}
        w_att = {r[1]: nn.Parameter(torch.Tensor(num_heads, out_dim, out_dim)) for r in etypes}
        w_msg = {r[1]: nn.Parameter(torch.Tensor(num_heads, out_dim, out_dim)) for r in etypes}
        mu = {r[1]: nn.Parameter(torch.ones(num_heads)) for r in etypes}
        self.micro_conv = nn.ModuleDict({
            etype: HGTAttention(
                num_heads * out_dim, num_heads, k_linear[stype], q_linear[dtype], v_linear[stype],
                w_att[etype], w_msg[etype], mu[etype]
            ) for stype, etype, dtype in etypes
        })
        self.reset_parameters2(w_att, w_msg)

    def reset_parameters2(self, w_att, w_msg):
        for etype in w_att:
            nn.init.xavier_uniform_(w_att[etype])
            nn.init.xavier_uniform_(w_msg[etype])


class MyGNN(nn.Module):

    def __init__(
            self, in_dims, hidden_dim, out_dim, num_heads, ntypes, etypes, predict_ntype,
            num_layers, dropout=0.0):
        """My GNN

        :param in_dims: Dict[str, int] 顶点类型到输入特征维数的映射
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param num_heads: int 注意力头数K
        :param ntypes: List[str] 顶点类型列表
        :param etypes: List[(str, str, str)] 规范边类型列表
        :param predict_ntype: str 待预测顶点类型
        :param num_layers: int 层数
        :param dropout: float, optional Dropout概率，默认为0
        """
        super().__init__()
        self.predict_ntype = predict_ntype
        self.fc_in = nn.ModuleDict({
            ntype: nn.Linear(in_dim, num_heads * hidden_dim) for ntype, in_dim in in_dims.items()
        })
        self.layers = nn.ModuleList([
            MyGNNLayer(
                num_heads * hidden_dim, hidden_dim, num_heads, ntypes, etypes, dropout
            ) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(num_heads * hidden_dim, out_dim)

    def forward(self, blocks, feats):
        """
        :param blocks: List[DGLBlock]
        :param feats: Dict[str, tensor(N_i, d_in_i)] 顶点类型到输入顶点特征的映射
        :return: tensor(N_i, d_out) 待预测顶点的最终嵌入
        """
        feats = {ntype: self.fc_in[ntype](feat) for ntype, feat in feats.items()}
        for i in range(len(self.layers)):
            feats = self.layers[i](blocks[i], feats)  # {ntype: tensor(N_i, K*d_hid)}
        return self.classifier(feats[self.predict_ntype])
