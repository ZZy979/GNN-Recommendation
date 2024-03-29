import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
from dgl.ops import edge_softmax


class HeCoGATConv(nn.Module):

    def __init__(self, hidden_dim, attn_drop=0.0, negative_slope=0.01, activation=None):
        """HeCo作者代码中使用的GAT

        :param hidden_dim: int 隐含特征维数
        :param attn_drop: float 注意力dropout
        :param negative_slope: float, optional LeakyReLU负斜率，默认为0.01
        :param activation: callable, optional 激活函数，默认为None
        """
        super().__init__()
        self.attn_l = nn.Parameter(torch.FloatTensor(1, hidden_dim))
        self.attn_r = nn.Parameter(torch.FloatTensor(1, hidden_dim))
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.attn_l, gain)
        nn.init.xavier_normal_(self.attn_r, gain)

    def forward(self, g, feat_src, feat_dst):
        """
        :param g: DGLGraph 邻居-目标顶点二分图
        :param feat_src: tensor(N_src, d) 邻居顶点输入特征
        :param feat_dst: tensor(N_dst, d) 目标顶点输入特征
        :return: tensor(N_dst, d) 目标顶点输出特征
        """
        with g.local_scope():
            # HeCo作者代码中使用attn_drop的方式与原始GAT不同，这样是不对的，却能顶点聚类提升性能……
            attn_l = self.attn_drop(self.attn_l)
            attn_r = self.attn_drop(self.attn_r)
            el = (feat_src * attn_l).sum(dim=-1).unsqueeze(dim=-1)  # (N_src, 1)
            er = (feat_dst * attn_r).sum(dim=-1).unsqueeze(dim=-1)  # (N_dst, 1)
            g.srcdata.update({'ft': feat_src, 'el': el})
            g.dstdata['er'] = er
            g.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(g.edata.pop('e'))
            g.edata['a'] = edge_softmax(g, e)  # (E, 1)

            # 消息传递
            g.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            ret = g.dstdata['ft']
            if self.activation:
                ret = self.activation(ret)
            return ret


class Attention(nn.Module):

    def __init__(self, hidden_dim, attn_drop):
        """语义层次的注意力

        :param hidden_dim: int 隐含特征维数
        :param attn_drop: float 注意力dropout
        """
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.Parameter(torch.FloatTensor(1, hidden_dim))
        self.attn_drop = nn.Dropout(attn_drop)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain)
        nn.init.xavier_normal_(self.attn, gain)

    def forward(self, h):
        """
        :param h: tensor(N, M, d) 顶点基于不同元路径/类型的嵌入，N为顶点数，M为元路径/类型数
        :return: tensor(N, d) 顶点的最终嵌入
        """
        attn = self.attn_drop(self.attn)
        # (N, M, d) -> (M, d) -> (M, 1)
        w = torch.tanh(self.fc(h)).mean(dim=0).matmul(attn.t())
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((h.shape[0],) + beta.shape)  # (N, M, 1)
        z = (beta * h).sum(dim=1)  # (N, d)
        return z


class NetworkSchemaEncoder(nn.Module):

    def __init__(self, hidden_dim, attn_drop, relations):
        """网络结构视图编码器

        :param hidden_dim: int 隐含特征维数
        :param attn_drop: float 注意力dropout
        :param relations: List[(str, str, str)] 目标顶点关联的关系列表，长度为邻居类型数S
        """
        super().__init__()
        self.relations = relations
        self.dtype = relations[0][2]
        self.gats = nn.ModuleDict({
            r[0]: HeCoGATConv(hidden_dim, attn_drop, activation=F.elu)
            for r in relations
        })
        self.attn = Attention(hidden_dim, attn_drop)

    def forward(self, g, feats):
        """
        :param g: DGLGraph 异构图
        :param feats: Dict[str, tensor(N_i, d)] 顶点类型到输入特征的映射
        :return: tensor(N_dst, d) 目标顶点的最终嵌入
        """
        feat_dst = feats[self.dtype][:g.num_dst_nodes(self.dtype)]
        h = []
        for stype, etype, dtype in self.relations:
            h.append(self.gats[stype](g[stype, etype, dtype], feats[stype], feat_dst))
        h = torch.stack(h, dim=1)  # (N_dst, S, d)
        z_sc = self.attn(h)  # (N_dst, d)
        return z_sc


class PositiveGraphEncoder(nn.Module):

    def __init__(self, num_metapaths, in_dim, hidden_dim, attn_drop):
        """正样本视图编码器

        :param num_metapaths: int 元路径数量M
        :param hidden_dim: int 隐含特征维数
        :param attn_drop: float 注意力dropout
        """
        super().__init__()
        self.gcns = nn.ModuleList([
            GraphConv(in_dim, hidden_dim, norm='right', activation=nn.PReLU())
            for _ in range(num_metapaths)
        ])
        self.attn = Attention(hidden_dim, attn_drop)

    def forward(self, mgs, feats):
        """
        :param mgs: List[DGLGraph] 正样本图
        :param feats: List[tensor(N, d)] 输入顶点特征
        :return: tensor(N, d) 输出顶点特征
        """
        h = [gcn(mg, feat) for gcn, mg, feat in zip(self.gcns, mgs, feats)]
        h = torch.stack(h, dim=1)  # (N, M, d)
        z_pg = self.attn(h)  # (N, d)
        return z_pg


class Contrast(nn.Module):

    def __init__(self, hidden_dim, tau, lambda_):
        """对比损失模块

        :param hidden_dim: int 隐含特征维数
        :param tau: float 温度参数
        :param lambda_: float 0~1之间，网络结构视图损失的系数（元路径视图损失的系数为1-λ）
        """
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lambda_ = lambda_
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for model in self.proj:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain)

    def sim(self, x, y):
        """计算相似度矩阵

        :param x: tensor(N, d)
        :param y: tensor(N, d)
        :return: tensor(N, N) S[i, j] = exp(cos(x[i], y[j]))
        """
        x_norm = torch.norm(x, dim=1, keepdim=True)
        y_norm = torch.norm(y, dim=1, keepdim=True)
        numerator = torch.mm(x, y.t())
        denominator = torch.mm(x_norm, y_norm.t())
        return torch.exp(numerator / denominator / self.tau)

    def forward(self, z_sc, z_mp, pos):
        """
        :param z_sc: tensor(N, d) 目标顶点在网络结构视图下的嵌入
        :param z_mp: tensor(N, d) 目标顶点在元路径视图下的嵌入
        :param pos: tensor(B, N) 0-1张量，每个目标顶点的正样本
            （B是batch大小，真正的目标顶点；N是B个目标顶点加上其正样本后的顶点数）
        :return: float 对比损失
        """
        z_sc_proj = self.proj(z_sc)
        z_mp_proj = self.proj(z_mp)
        sim_sc2mp = self.sim(z_sc_proj, z_mp_proj)
        sim_mp2sc = sim_sc2mp.t()

        batch = pos.shape[0]
        sim_sc2mp = sim_sc2mp / (sim_sc2mp.sum(dim=1, keepdim=True) + 1e-8)  # 不能改成/=
        loss_sc = -torch.log(torch.sum(sim_sc2mp[:batch] * pos, dim=1)).mean()

        sim_mp2sc = sim_mp2sc / (sim_mp2sc.sum(dim=1, keepdim=True) + 1e-8)
        loss_mp = -torch.log(torch.sum(sim_mp2sc[:batch] * pos, dim=1)).mean()
        return self.lambda_ * loss_sc + (1 - self.lambda_) * loss_mp


class HeCo(nn.Module):

    def __init__(self, in_dims, hidden_dim, feat_drop, attn_drop, relations, tau, lambda_):
        """HeCo模型

        :param in_dims: Dict[str, int] 顶点类型到输入特征维数的映射
        :param hidden_dim: int 隐含特征维数
        :param feat_drop: float 输入特征dropout
        :param attn_drop: float 注意力dropout
        :param relations: List[(str, str, str)] 目标顶点关联的关系列表，长度为邻居类型数S
        :param tau: float 温度参数
        :param lambda_: float 0~1之间，网络结构视图损失的系数（元路径视图损失的系数为1-λ）
        """
        super().__init__()
        self.dtype = relations[0][2]
        self.fcs = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden_dim) for ntype, in_dim in in_dims.items()
        })
        self.feat_drop = nn.Dropout(feat_drop)
        self.sc_encoder = NetworkSchemaEncoder(hidden_dim, attn_drop, relations)
        self.mp_encoder = PositiveGraphEncoder(len(relations), hidden_dim, hidden_dim, attn_drop)
        self.contrast = Contrast(hidden_dim, tau, lambda_)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for ntype in self.fcs:
            nn.init.xavier_normal_(self.fcs[ntype].weight, gain)

    def forward(self, g, feats, mgs, mg_feats, pos):
        """
        :param g: DGLGraph 异构图
        :param feats: Dict[str, tensor(N_i, d_in)] 顶点类型到输入特征的映射
        :param mgs: List[DGLBlock] 正样本图，len(mgs)=元路径数量=目标顶点邻居类型数S≠模型层数
        :param mg_feats: List[tensor(N_pos_src, d_in)] 正样本图源顶点的输入特征
        :param pos: tensor(B, N) 布尔张量，每个顶点的正样本
            （B是batch大小，真正的目标顶点；N是B个目标顶点加上其正样本后的顶点数）
        :return: float, tensor(B, d_hid) 对比损失，元路径编码器输出的目标顶点特征
        """
        h = {ntype: F.elu(self.feat_drop(self.fcs[ntype](feat))) for ntype, feat in feats.items()}
        mg_h = [F.elu(self.feat_drop(self.fcs[self.dtype](mg_feat))) for mg_feat in mg_feats]
        z_sc = self.sc_encoder(g, h)  # (N, d_hid)
        z_mp = self.mp_encoder(mgs, mg_h)  # (N, d_hid)
        loss = self.contrast(z_sc, z_mp, pos)
        return loss, z_mp[:pos.shape[0]]

    @torch.no_grad()
    def get_embeds(self, mgs, feats):
        """计算目标顶点的最终嵌入(z_mp)

        :param mgs: List[DGLBlock] 正样本图
        :param feats: List[tensor(N_pos_src, d_in)] 正样本图源顶点的输入特征
        :return: tensor(N_tgt, d_hid) 目标顶点的最终嵌入
        """
        h = [F.elu(self.fcs[self.dtype](feat)) for feat in feats]
        z_mp = self.mp_encoder(mgs, h)
        return z_mp
