import dgl
import pandas as pd
import torch
from dgl.dataloading.negative_sampler import Uniform

from gnnrec.kgrec.data import OAGCoreDataset
from gnnrec.kgrec.utils import load_author_rank


class RatingKnowledgeGraphDataset:
    """基于知识图谱的用户评分数据集

    读取用户-物品评分数据和知识图谱三元组，并分别构造为两个图：

    * user_item_graph: DGLGraph (user, rate, item)二分图，由正向（评分大于等于阈值）的交互关系组成
    * knowledge_graph: DGLGraph 同构图，其中0~N_item-1对应user_item_graph中的item顶点（即物品集合是实体集合的子集）
      ，边特征relation表示关系类型
    """

    def __init__(self):
        g = OAGCoreDataset()[0]
        author_rank = load_author_rank()
        rating = pd.DataFrame(
            [[i, a] for i, (f, r) in enumerate(author_rank.items()) for a in r],
            columns=['user_id', 'item_id']
        )
        user_item_graph = dgl.heterograph(
            {('user', 'rate', 'item'): (rating['user_id'], rating['item_id'])},
            num_nodes_dict={'user': len(author_rank), 'item': g.num_nodes('author')}
        )

        # 负采样
        neg_sampler = Uniform(1)
        nu, nv = neg_sampler(user_item_graph, torch.arange(user_item_graph.num_edges()))
        u, v = user_item_graph.edges()
        self.user_item_graph = dgl.heterograph(
            {('user', 'rate', 'item'): (torch.cat([u, nu]), torch.cat([v, nv]))},
            num_nodes_dict={ntype: user_item_graph.num_nodes(ntype) for ntype in user_item_graph.ntypes}
        )
        self.user_item_graph.edata['label'] = torch.cat([torch.ones(u.shape[0]), torch.zeros(nu.shape[0])])

        knowledge_graph = dgl.to_homogeneous(dgl.node_type_subgraph(g, ['author', 'institution', 'paper']))
        knowledge_graph.edata['relation'] = knowledge_graph.edata[dgl.NTYPE]
        self.knowledge_graph = dgl.add_reverse_edges(knowledge_graph, copy_edata=True)

    def get_num(self):
        return self.user_item_graph.num_nodes('user'), self.knowledge_graph.num_nodes(), self.knowledge_graph.edata['relation'].max().item() + 1
