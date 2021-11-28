import torch

from gnnrec.config import DATA_DIR, MODEL_DIR
from gnnrec.kgrec.data import OAGCSContrastDataset
from gnnrec.kgrec.garec.model import ContrastiveSciBERT


class Context:

    def __init__(self, paper_embeds, scibert_model):
        """论文召回模块上下文对象

        :param paper_embeds: tensor(N_paper, d) 论文标题向量
        :param scibert_model: ContrastiveSciBERT 微调后的SciBERT模型
        """
        self.paper_embeds = paper_embeds
        self.scibert_model = scibert_model


def get_context():
    paper_embeds = torch.load(DATA_DIR / 'oag/cs/paper_feat.pkl', map_location='cpu')
    scibert_model = ContrastiveSciBERT(128, 0.07)
    scibert_model.load_state_dict(torch.load(MODEL_DIR / 'scibert.pt', map_location='cpu'))
    return Context(paper_embeds, scibert_model)


def recall(ctx, query, k=1000):
    """根据输入的查询词在oag-cs数据集召回论文

    :param ctx: Context 上下文对象
    :param query: str 查询词
    :param k: int, optional 召回论文数量，默认为1000
    :return: List[float], List[int] Top k论文的相似度和id，按相似度降序排序
    """
    q = ctx.scibert_model.get_embeds(query).squeeze(dim=0)  # (d,)
    q = q / q.norm()
    similarity = torch.matmul(ctx.paper_embeds, q)
    score, pid = similarity.topk(k)
    return score.tolist(), pid.tolist()


def main():
    ctx = get_context()
    paper_titles = OAGCSContrastDataset(DATA_DIR / 'oag/cs/mag_papers.txt', 'all')
    while True:
        query = input('query> ').strip()
        score, pid = recall(ctx, query, 10)
        for i in range(len(pid)):
            print('{:.4f}\t{}'.format(score[i], paper_titles[pid[i]][0]))


if __name__ == '__main__':
    main()
