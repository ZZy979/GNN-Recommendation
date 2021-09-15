import torch
import torch.nn as nn
from dgl.dataloading import MultiLayerNeighborSampler
from dgl.nn import GraphConv
from torch.utils.data import DataLoader

from ..heco.collator import PositiveSampleCollator
from ..heco.model import Contrast
from ..rhgnn.model import RHGNN


class RHCO(nn.Module):

    def __init__(
            self, in_dims, hidden_dim, out_dim, rel_hidden_dim, num_heads,
            ntypes, etypes, predict_ntype, num_layers, dropout, tau, lambda_):
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
        self.mp_encoder = GraphConv(
            in_dims[predict_ntype], hidden_dim, norm='right', activation=nn.PReLU()
        )
        self.contrast = Contrast(hidden_dim, tau, lambda_)
        self.predict = nn.Linear(hidden_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.predict.weight, gain)

    def forward(self, blocks, feats, pos_g, pos_feat, pos):
        """
        :param blocks: List[DGLBlock]
        :param feats: Dict[str, tensor(N_i, d_in)] 顶点类型到输入特征的映射
        :param pos_g: DGLGraph 正样本图
        :param pos_feat: tensor(N_pos_src, d_in) 正样本图源顶点的输入特征
        :param pos: tensor(B, N) 布尔张量，每个顶点的正样本
            （B是batch大小，真正的目标顶点；N是B个目标顶点加上其正样本后的顶点数）
        :return: float, tensor(B, d_out) 对比损失，目标顶点输出特征
        """
        z_sc = self.sc_encoder(blocks, feats)  # (N, d_hid)
        z_mp = self.mp_encoder(pos_g, pos_feat)  # (N, d_hid)
        loss = self.contrast(z_sc, z_mp, pos)
        return loss, self.predict(z_sc[:pos.shape[0]])

    @torch.no_grad()
    def get_embeds(self, g, pos, neighbor_size, batch_size, device):
        """计算目标顶点的最终嵌入(z_sc)

        :param g: DGLGraph 异构图
        :param pos: tensor(N_tgt, T_pos) 每个目标顶点的正样本id
        :param neighbor_size: int 邻居采样数
        :param batch_size: int 批大小
        :param device torch.device GPU设备
        :return: tensor(N_tgt, d_out) 目标顶点的最终嵌入
        """
        with g.local_scope():
            sampler = MultiLayerNeighborSampler([neighbor_size] * len(self.sc_encoder.layers))
            collator = PositiveSampleCollator(g, sampler, pos, self.predict_ntype)
            loader = DataLoader(g.nodes(self.predict_ntype), batch_size=batch_size)
            embeds = torch.zeros(g.num_nodes(self.predict_ntype), self.hidden_dim, device=device)
            for batch in loader:
                blocks = [b.to(device) for b in collator.collate(batch)]
                z_sc = self.sc_encoder(blocks, blocks[0].srcdata['feat'])
                embeds[batch] = z_sc[:batch.shape[0]]
            return self.predict(embeds)
