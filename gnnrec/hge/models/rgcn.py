import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import HeteroGraphConv, GraphConv, WeightBasis


class RelGraphConv(nn.Module):

    def __init__(
            self, in_dim, out_dim, rel_names, num_bases=None,
            weight=True, self_loop=True, activation=None, dropout=0.0):
        """R-GCN层（用于异构图）

        :param in_dim: 输入特征维数
        :param out_dim: 输出特征维数
        :param rel_names: List[str] 关系名称
        :param num_bases: int, optional 基的个数，默认使用关系个数
        :param weight: bool, optional 是否进行线性变换，默认为True
        :param self_loop: 是否包括自环消息，默认为True
        :param activation: callable, optional 激活函数，默认为None
        :param dropout: float, optional Dropout概率，默认为0
        """
        super().__init__()
        self.rel_names = rel_names
        self.self_loop = self_loop
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        self.conv = HeteroGraphConv({
            rel: GraphConv(in_dim, out_dim, norm='right', weight=False, bias=False)
            for rel in rel_names
        })

        self.use_weight = weight
        if not num_bases:
            num_bases = len(rel_names)
        self.use_basis = weight and 0 < num_bases < len(rel_names)
        if self.use_weight:
            if self.use_basis:
                self.basis = WeightBasis((in_dim, out_dim), num_bases, len(rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(rel_names), in_dim, out_dim))
                nn.init.xavier_uniform_(self.weight, nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
            nn.init.xavier_uniform_(self.loop_weight, nn.init.calculate_gain('relu'))

    def forward(self, g, inputs):
        """
        :param g: DGLGraph 异构图
        :param inputs: Dict[str, tensor(N_i, d_in)] 顶点类型到输入特征的映射
        :return: Dict[str, tensor(N_i, d_out)] 顶点类型到输出特征的映射
        """
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight  # (R, d_in, d_out)
            kwargs = {rel: {'weight': weight[i]} for i, rel in enumerate(self.rel_names)}
        else:
            kwargs = {}
        hs = self.conv(g, inputs, mod_kwargs=kwargs)  # Dict[ntype, (N_i, d_out)]
        for ntype in hs:
            if self.self_loop:
                hs[ntype] += torch.matmul(inputs[ntype], self.loop_weight)
            if self.activation:
                hs[ntype] = self.activation(hs[ntype])
            hs[ntype] = self.dropout(hs[ntype])
        return hs


class RGCN(nn.Module):

    def __init__(
            self, num_nodes, in_dims, hidden_dim, out_dim, rel_names,
            num_hidden_layers=1, num_bases=None, self_loop=True, dropout=0.0):
        """R-GCN实体分类模型

        :param num_nodes: Dict[str, int] 顶点类型到顶点数的映射
        :param in_dims: Dict[str, int] 顶点类型到输入特征维数的映射
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param rel_names: List[str] 关系名称
        :param num_hidden_layers: int, optional R-GCN隐藏层数，默认为1
        :param num_bases: int, optional 基的个数，默认使用关系个数
        :param self_loop: bool 是否包括自环消息，默认为True
        :param dropout: float, optional Dropout概率，默认为0
        """
        super().__init__()
        self.fc = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_dim) for ntype, in_dim in in_dims.items()
        })
        self.embeds = nn.ModuleDict({
            ntype: nn.Embedding(num_nodes[ntype], hidden_dim)
            for ntype in num_nodes if ntype not in in_dims
        })
        self.layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.layers.append(RelGraphConv(
                hidden_dim, hidden_dim, rel_names, num_bases, False, self_loop, F.relu, dropout
            ))
        self.layers.append(RelGraphConv(
            hidden_dim, out_dim, rel_names, num_bases, True, self_loop, dropout=dropout
        ))
        self.reset_parameters()

    def reset_parameters(self):
        for k in self.embeds:
            nn.init.xavier_uniform_(self.embeds[k].weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, g, h):
        """
        :param g: DGLGraph 异构图
        :param h: Dict[str, tensor(N_i, d_in_i)] （部分）顶点类型到输入特征的映射
        :return: Dict[str, tensor(N_i, d_out)] 顶点类型到顶点嵌入的映射
        """
        h = {k: self.fc[k](feat) for k, feat in h.items()}
        for k in self.embeds:
            h[k] = self.embeds[k].weight
        for layer in self.layers:
            h = layer(g, h)  # Dict[ntype, (N_i, d_hid)]
        return h