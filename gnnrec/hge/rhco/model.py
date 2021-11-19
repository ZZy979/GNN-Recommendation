import torch
import torch.nn as nn
from dgl.dataloading import MultiLayerFullNeighborSampler, MultiLayerNeighborSampler, NodeDataLoader

from ..heco.model import PositiveGraphEncoder, Contrast
from ..rhgnn.model import RHGNN


class RHCO(nn.Module):

    def __init__(
            self, in_dims, hidden_dim, out_dim, rel_hidden_dim, num_heads,
            ntypes, etypes, predict_ntype, num_layers, dropout, num_pos_graphs, tau, lambda_):
        """基于对比学习的关系感知异构图神经网络RHCO

        :param in_dims: Dict[str, int] 顶点类型到输入特征维数的映射
        :param hidden_dim: int 隐含特征维数
        :param out_dim: int 输出特征维数
        :param rel_hidden_dim: int 关系隐含特征维数
        :param num_heads: int 注意力头数K
        :param ntypes: List[str] 顶点类型列表
        :param etypes – List[(str, str, str)] 规范边类型列表
        :param predict_ntype: str 目标顶点类型
        :param num_layers: int 网络结构编码器层数
        :param dropout: float 输入特征dropout
        :param num_pos_graphs: int 正样本图个数M
        :param tau: float 温度参数τ
        :param lambda_: float 0~1之间，网络结构视图损失的系数λ（元路径视图损失的系数为1-λ）
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.predict_ntype = predict_ntype
        self.sc_encoder = RHGNN(
            in_dims, hidden_dim, hidden_dim, rel_hidden_dim, rel_hidden_dim, num_heads,
            ntypes, etypes, predict_ntype, num_layers, dropout
        )
        self.pg_encoder = PositiveGraphEncoder(
            num_pos_graphs, in_dims[predict_ntype], hidden_dim, dropout
        )
        self.contrast = Contrast(hidden_dim, tau, lambda_)
        self.predict = nn.Linear(hidden_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.predict.weight, gain)

    def forward(self, blocks, feats, mgs, mg_feats, pos):
        """
        :param blocks: List[DGLBlock]
        :param feats: Dict[str, tensor(N_i, d_in)] 顶点类型到输入特征的映射
        :param mgs: List[DGLBlock] 正样本图，len(mgs)=元路径数量=目标顶点邻居类型数S≠模型层数
        :param mg_feats: List[tensor(N_pos_src, d_in)] 正样本图源顶点的输入特征
        :param pos: tensor(B, N) 布尔张量，每个顶点的正样本
            （B是batch大小，真正的目标顶点；N是B个目标顶点加上其正样本后的顶点数）
        :return: float, tensor(B, d_out) 对比损失，目标顶点输出特征
        """
        z_sc = self.sc_encoder(blocks, feats)  # (N, d_hid)
        z_pg = self.pg_encoder(mgs, mg_feats)  # (N, d_hid)
        loss = self.contrast(z_sc, z_pg, pos)
        return loss, self.predict(z_sc[:pos.shape[0]])

    @torch.no_grad()
    def get_embeds(self, g, mgs, neighbor_size, batch_size, device):
        """计算目标顶点的最终嵌入(z_sc)

        :param g: DGLGraph 异构图
        :param mgs: List[DGLGraph] 正样本图
        :param neighbor_size: int 邻居采样数
        :param batch_size: int 批大小
        :param device torch.device GPU设备
        :return: tensor(N_tgt, d_out) 目标顶点的最终嵌入
        """
        sampler = MultiLayerNeighborSampler([neighbor_size] * len(self.sc_encoder.layers))
        loader = NodeDataLoader(
            g, {self.predict_ntype: g.nodes(self.predict_ntype)}, sampler,
            device=device, batch_size=batch_size
        )
        embeds = torch.zeros(g.num_nodes(self.predict_ntype), self.hidden_dim, device=device)
        for input_nodes, output_nodes, blocks in loader:
            z_sc = self.sc_encoder(blocks, blocks[0].srcdata['feat'])
            embeds[output_nodes[self.predict_ntype]] = z_sc
        return self.predict(embeds)


class RHCOFull(RHCO):
    """Full-batch RHCO"""

    def forward(self, g, feats, mgs, mg_feat, pos):
        return super().forward(
            [g] * len(self.sc_encoder.layers), feats, mgs, [mg_feat] * len(mgs), pos
        )

    @torch.no_grad()
    def get_embeds(self, g, *args):
        return self.predict(self.sc_encoder([g] * len(self.sc_encoder.layers), g.ndata['feat']))


class RHCOsc(RHCO):
    """RHCO消融实验变体：仅使用网络结构编码器"""

    def forward(self, blocks, feats, mgs, mg_feats, pos):
        z_sc = self.sc_encoder(blocks, feats)  # (N, d_hid)
        loss = self.contrast(z_sc, z_sc, pos)
        return loss, self.predict(z_sc[:pos.shape[0]])


class RHCOpg(RHCO):
    """RHCO消融实验变体：仅使用正样本图编码器"""

    def forward(self, blocks, feats, mgs, mg_feats, pos):
        z_pg = self.pg_encoder(mgs, mg_feats)  # (N, d_hid)
        loss = self.contrast(z_pg, z_pg, pos)
        return loss, self.predict(z_pg[:pos.shape[0]])

    def get_embeds(self, g, mgs, neighbor_size, batch_size, device):
        feat = g.nodes[self.predict_ntype].data['feat']
        sampler = MultiLayerFullNeighborSampler(1)
        mg_loaders = [
            NodeDataLoader(mg, g.nodes(self.predict_ntype), sampler, device=device, batch_size=batch_size)
            for mg in mgs
        ]
        embeds = torch.zeros(g.num_nodes(self.predict_ntype), self.hidden_dim, device=device)
        for mg_blocks in zip(*mg_loaders):
            output_nodes = mg_blocks[0][1]
            mg_feats = [feat[i] for i, _, _ in mg_blocks]
            mg_blocks = [b[0] for _, _, b in mg_blocks]
            embeds[output_nodes] = self.pg_encoder(mg_blocks, mg_feats)
        return self.predict(embeds)
