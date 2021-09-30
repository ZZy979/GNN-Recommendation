import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import HeteroGraphConv, GraphConv


class RelGraphConv(nn.Module):

    def __init__(self, in_dim, out_dim, ntypes, etypes, activation=None, dropout=0.0):
        """R-GCN层（用于异构图）

        :param in_dim: 输入特征维数
        :param out_dim: 输出特征维数
        :param ntypes: List[str] 顶点类型列表
        :param etypes: List[str] 边类型列表
        :param activation: callable, optional 激活函数，默认为None
        :param dropout: float, optional Dropout概率，默认为0
        """
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.conv = HeteroGraphConv({
            etype: GraphConv(in_dim, out_dim, norm='right', bias=False)
            for etype in etypes
        }, 'sum')
        self.loop_weight = nn.ModuleDict({
            ntype: nn.Linear(in_dim, out_dim, bias=False) for ntype in ntypes
        })

    def forward(self, g, feats):
        """
        :param g: DGLGraph 异构图
        :param feats: Dict[str, tensor(N_i, d_in)] 顶点类型到输入特征的映射
        :return: Dict[str, tensor(N_i, d_out)] 顶点类型到输出特征的映射
        """
        if g.is_block:
            feats_dst = {ntype: feat[:g.num_dst_nodes(ntype)] for ntype, feat in feats.items()}
        else:
            feats_dst = feats
        out = self.conv(g, (feats, feats_dst))  # Dict[ntype, (N_i, d_out)]
        for ntype in out:
            out[ntype] += self.loop_weight[ntype](feats_dst[ntype])
            if self.activation:
                out[ntype] = self.activation(out[ntype])
            out[ntype] = self.dropout(out[ntype])
        return out


class RGCN(nn.Module):

    def __init__(
            self, in_dim, hidden_dim, out_dim, input_ntypes, num_nodes, etypes, predict_ntype,
            num_layers=2, dropout=0.0):
        """R-GCN模型

        :param in_dim: int 输入特征维数
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param input_ntypes: List[str] 有输入特征的顶点类型列表
        :param num_nodes: Dict[str, int] 顶点类型到顶点数的映射
        :param etypes: List[str] 边类型列表
        :param predict_ntype: str 待预测顶点类型
        :param num_layers: int, optional 层数，默认为2
        :param dropout: float, optional Dropout概率，默认为0
        """
        super().__init__()
        self.embeds = nn.ModuleDict({
            ntype: nn.Embedding(num_nodes[ntype], in_dim)
            for ntype in num_nodes if ntype not in input_ntypes
        })
        ntypes = list(num_nodes)
        self.layers = nn.ModuleList()
        self.layers.append(RelGraphConv(in_dim, hidden_dim, ntypes, etypes, F.relu, dropout))
        for i in range(num_layers - 2):
            self.layers.append(RelGraphConv(hidden_dim, hidden_dim, ntypes, etypes, F.relu, dropout))
        self.layers.append(RelGraphConv(hidden_dim, out_dim, ntypes, etypes))
        self.predict_ntype = predict_ntype
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for k in self.embeds:
            nn.init.xavier_uniform_(self.embeds[k].weight, gain=gain)

    def forward(self, g, feats):
        """
        :param g: DGLGraph 异构图
        :param feats: Dict[str, tensor(N_i, d_in_i)] （部分）顶点类型到输入特征的映射
        :return: Dict[str, tensor(N_i, d_out)] 顶点类型到顶点嵌入的映射
        """
        for k in self.embeds:
            feats[k] = self.embeds[k].weight
        for i in range(len(self.layers)):
            feats = self.layers[i](g, feats)  # Dict[ntype, (N_i, d_hid)]
        return feats[self.predict_ntype]
