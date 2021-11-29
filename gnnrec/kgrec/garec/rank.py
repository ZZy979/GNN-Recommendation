import dgl
import torch

from gnnrec.config import DATA_DIR
from gnnrec.hge.utils import add_reverse_edges
from gnnrec.kgrec.data import OAGCSDataset
from gnnrec.kgrec.garec import recall
from gnnrec.kgrec.utils import iter_json, calc_author_citation


class Context:

    def __init__(self, recall_ctx, g, author_embeds, field2id):
        """学者排名模块上下文对象

        :param recall_ctx: recall.Context
        :param g: DGLGraph 异构图
        :param author_embeds: tensor(N_author, d) 学者嵌入
        :param field2id: Dict[str, int] 领域名称到id的映射
        """
        self.recall_ctx = recall_ctx
        g.nodes['paper'].data['citation'] = g.nodes['paper'].data['citation'].float()
        g.edges['writes'].data['order'] = g.edges['writes'].data['order'].float()
        self.g = g
        self.apg = g['author', 'writes', 'paper']
        self.author_embeds = author_embeds
        self.field2id = field2id


def get_context(recall_ctx):
    g = OAGCSDataset()[0]
    author_embeds = torch.load(DATA_DIR / 'rank/author_embed.pkl', map_location='cpu')
    field2id = {f['name']: i for i, f in enumerate(iter_json(DATA_DIR / 'oag/cs/mag_fields.txt'))}
    return Context(recall_ctx, g, author_embeds, field2id)


def rank(ctx, query, k=100):
    """根据输入的查询词在oag-cs数据集计算学者排名

    :param ctx: Context 上下文对象
    :param query: str 查询词
    :param k: int, optional 返回top学者数量，默认为100
    :return: List[float], List[int] 学者得分和id，按得分降序排序
    """
    if query in ctx.field2id:
        pid, _ = ctx.g.in_edges(ctx.field2id[query], etype='has_field')
    else:
        _, pid = recall.recall(ctx.recall_ctx, query, 200)
    sg = add_reverse_edges(dgl.in_subgraph(ctx.apg, {'paper': pid}, relabel_nodes=True))
    author_citation = calc_author_citation(sg)
    citation, idx = author_citation.topk(k)
    aid = sg.nodes['author'].data[dgl.NID][idx]
    return citation.tolist(), aid.tolist()


def rank_author(ctx, query, k=100):
    """根据输入的查询词在oag-cs数据集计算学者排名

    :param ctx: Context 上下文对象
    :param query: str 查询词
    :param k: int, optional 返回top学者数量，默认为100
    :return: List[float], List[int] 学者得分和id，按得分降序排序
    """
    q = ctx.recall_ctx.scibert_model.get_embeds(query).squeeze(dim=0)  # (d,)
    q = q / q.norm()
    _, pid = recall.recall(ctx.recall_ctx, query)
    aid, _ = ctx.apg.in_edges(pid)
    similarity = torch.matmul(ctx.author_embeds[aid], q)
    score, idx = similarity.topk(k)
    return score.tolist(), aid[idx].tolist()
