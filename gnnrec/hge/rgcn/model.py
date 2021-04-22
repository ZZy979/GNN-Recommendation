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

    def forward(self, g, feats):
        """
        :param g: DGLGraph 异构图
        :param feats: Dict[str, tensor(N_i, d_in)] 顶点类型到输入特征的映射
        :return: Dict[str, tensor(N_i, d_out)] 顶点类型到输出特征的映射
        """
        if g.is_block:
            feats_dst = {ntype: feats[ntype][:g.num_dst_nodes(ntype)] for ntype in feats}
        else:
            feats_dst = feats
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight  # (R, d_in, d_out)
            kwargs = {rel: {'weight': weight[i]} for i, rel in enumerate(self.rel_names)}
        else:
            kwargs = {}
        hs = self.conv(g, feats, mod_kwargs=kwargs)  # Dict[ntype, (N_i, d_out)]
        for ntype in hs:
            if self.self_loop:
                hs[ntype] += torch.matmul(feats_dst[ntype], self.loop_weight)
            if self.activation:
                hs[ntype] = self.activation(hs[ntype])
            hs[ntype] = self.dropout(hs[ntype])
        return hs


class RGCN(nn.Module):

    def __init__(
            self, num_nodes, in_dims, hidden_dim, out_dim, rel_names, predict_ntype,
            num_hidden_layers=1, num_bases=None, self_loop=True, dropout=0.0):
        """R-GCN模型

        :param num_nodes: Dict[str, int] 顶点类型到顶点数的映射
        :param in_dims: Dict[str, int] 顶点类型到输入特征维数的映射
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param rel_names: List[List[str]] 每一层的关系名称，长度等于num_hidden_layers+1
        :param predict_ntype: str 待预测顶点类型
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
        for i in range(num_hidden_layers):
            self.layers.append(RelGraphConv(
                hidden_dim, hidden_dim, rel_names[i], num_bases, False, self_loop, F.relu, dropout
            ))
        self.layers.append(RelGraphConv(
            hidden_dim, out_dim, rel_names[-1], num_bases, True, self_loop, dropout=dropout
        ))
        self.predict_ntype = predict_ntype
        self.reset_parameters()

    def reset_parameters(self):
        for k in self.embeds:
            nn.init.xavier_uniform_(self.embeds[k].weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, blocks, h):
        """
        :param blocks: List[DGLBlock]
        :param h: Dict[str, tensor(N_i, d_in_i)] （部分）顶点类型到输入特征的映射
        :return: Dict[str, tensor(N_i, d_out)] 顶点类型到顶点嵌入的映射
        """
        h = {k: self.fc[k](feat) for k, feat in h.items()}
        for k in self.embeds:
            h[k] = self.embeds[k](blocks[0].srcnodes(k))
        for i in range(len(self.layers)):
            h = self.layers[i](blocks[i], h)  # Dict[ntype, (N_i, d_hid)]
        return h[self.predict_ntype]


class RGCNFull(RGCN):

    def __init__(
            self, num_nodes, in_dims, hidden_dim, out_dim, rel_names, predict_ntype,
            num_hidden_layers=1, num_bases=None, self_loop=True, dropout=0.0):
        """RGCN模型（全图训练）

        :param num_nodes: Dict[str, int] 顶点类型到顶点数的映射
        :param in_dims: Dict[str, int] 顶点类型到输入特征维数的映射
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param rel_names: List[str] 关系名称
        :param predict_ntype: str 待预测顶点类型
        :param num_hidden_layers: int, optional R-GCN隐藏层数，默认为1
        :param num_bases: int, optional 基的个数，默认使用关系个数
        :param self_loop: bool 是否包括自环消息，默认为True
        :param dropout: float, optional Dropout概率，默认为0
        """
        super().__init__(
            num_nodes, in_dims, hidden_dim, out_dim, [rel_names] * (num_hidden_layers + 1),
            predict_ntype, num_hidden_layers, num_bases, self_loop, dropout
        )

    def forward(self, g, h):
        """
        :param g: DGLGraph 异构图
        :param h: Dict[str, tensor(N_i, d_in_i)] （部分）顶点类型到输入特征的映射
        :return: Dict[str, tensor(N_i, d_out)] 顶点类型到顶点嵌入的映射
        """
        h = {k: self.fc[k](feat) for k, feat in h.items()}
        for k in self.embeds:
            h[k] = self.embeds[k].weight
        for i in range(len(self.layers)):
            h = self.layers[i](g, h)  # Dict[ntype, (N_i, d_hid)]
        return h[self.predict_ntype]
