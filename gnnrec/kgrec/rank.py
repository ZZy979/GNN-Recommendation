import json

from gnnrec.config import DATA_DIR


class Context:

    def __init__(self, recall_ctx, author_rank):
        """学者排名模块上下文对象

        :param recall_ctx: gnnrec.kgrec.recall.Context
        :param author_rank: {field_id: [author_id]} 领域学者排名
        """
        self.recall_ctx = recall_ctx
        # 之后需要：author_embeds
        self.author_rank = author_rank


def get_context(recall_ctx):
    with open(DATA_DIR / 'rank/author_rank_train.json') as f:
        author_rank = json.load(f)
    return Context(recall_ctx, author_rank)


def rank(ctx, query):
    """根据输入的查询词在oag-cs数据集计算学者排名

    :param ctx: Context 上下文对象
    :param query: str 查询词
    :return: List[float], List[int] 学者得分和id，按得分降序排序
    """
    return [], ctx.author_rank.get(query, [])
