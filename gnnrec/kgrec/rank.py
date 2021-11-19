import json

import torch

from gnnrec.config import DATA_DIR
from gnnrec.kgrec import recall
from gnnrec.kgrec.data import OAGCSDataset
from gnnrec.kgrec.utils import iter_json


class Context:

    def __init__(self, recall_ctx, apg, author_embeds, field2id, author_rank):
        """学者排名模块上下文对象

        :param recall_ctx: gnnrec.kgrec.recall.Context
        :param apg: DGLGraph 学者-论文二分图
        :param author_embeds: tensor(N_author, d) 学者嵌入
        :param field2id: Dict[str, int] 领域名称到id的映射
        :param author_rank: Dict[int, List[int] 学者真实排名{field_id: [author_id]}
        """
        self.recall_ctx = recall_ctx
        self.apg = apg
        self.author_embeds = author_embeds
        self.field2id = field2id
        self.author_rank = author_rank


def get_context(recall_ctx):
    apg = OAGCSDataset()[0]['author', 'writes', 'paper']
    author_embeds = torch.load(DATA_DIR / 'rank/author_embed.pkl', map_location='cpu')
    field2id = {f['name']: i for i, f in enumerate(iter_json(DATA_DIR / 'oag/cs/mag_fields.txt'))}
    with open(DATA_DIR / 'rank/author_rank_train.json') as f:
        author_rank = {int(k): v for k, v in json.load(f).items()}
    with open(DATA_DIR / 'rank/author_rank_val.json') as f:
        author_rank.update({int(k): v for k, v in json.load(f).items()})
    return Context(recall_ctx, apg, author_embeds, field2id, author_rank)


def rank(ctx, query, k=100):
    """根据输入的查询词在oag-cs数据集计算学者排名

    :param ctx: Context 上下文对象
    :param query: str 查询词
    :param k: int, optional 返回top学者数量，默认为100
    :return: List[float], List[int] 学者得分和id，按得分降序排序
    """
    # if query in ctx.field2id and ctx.field2id[query] in ctx.author_rank:
    #     aid = ctx.author_rank[ctx.field2id[query]]
    #     return list(range(len(aid), 0, -1)), aid
    # else:
    q = ctx.recall_ctx.scibert_model.get_embeds(query).squeeze(dim=0)  # (d,)
    q = q / q.norm()
    _, pid = recall.recall(ctx.recall_ctx, query)
    aid, _ = ctx.apg.in_edges(pid)
    similarity = torch.matmul(ctx.author_embeds[aid], q)
    score, idx = similarity.topk(k)
    return score.tolist(), aid[idx].tolist()
