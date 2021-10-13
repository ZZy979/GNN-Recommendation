import argparse

import torch

from gnnrec.kgrec.data import OAGCSContrastDataset
from gnnrec.kgrec.scibert import ContrastiveSciBERT


class Context:

    def __init__(self, paper_embeds, scibert_model):
        """论文召回模块上下文对象

        :param paper_embeds: tensor(N, d) 论文标题向量
        :param scibert_model: ContrastiveSciBERT 微调后的SciBERT模型
        """
        self.paper_embeds = paper_embeds
        self.scibert_model = scibert_model


def get_context(paper_embeds_file, scibert_model_file):
    paper_embeds = torch.load(paper_embeds_file, map_location='cpu')
    scibert_model = ContrastiveSciBERT(128, 0.07)
    scibert_model.load_state_dict(torch.load(scibert_model_file, map_location='cpu'))
    return Context(paper_embeds, scibert_model)


def recall(ctx, query, k=1000):
    """根据输入的查询词在oag-cs数据集召回论文

    :param ctx: Context 上下文对象
    :param query: str 查询词
    :param k: int, optional 召回论文数量，默认为1000
    :return: tensor(k), tensor(k) Top k论文的相似度和id
    """
    q = ctx.scibert_model.get_embeds(query)  # (1, d)
    q = q / q.norm()
    similarity = torch.mm(ctx.paper_embeds, q.t()).squeeze(dim=1)  # (N,)
    return similarity.topk(k, dim=0)


def main():
    parser = argparse.ArgumentParser(description='oag-cs数据集 论文召回模块')
    parser.add_argument('paper_embeds_file', help='预训练的论文标题向量文件')
    parser.add_argument('scibert_model_file', help='微调后的SciBERT模型文件')
    parser.add_argument('raw_paper_file', help='原始论文数据文件')
    args = parser.parse_args()

    ctx = get_context(args.paper_embeds_file, args.scibert_model_file)
    data = OAGCSContrastDataset(args.raw_paper_file, 'all')
    while True:
        query = input('query> ').strip()
        score, pid = recall(ctx, query, 10)
        for i in range(len(pid)):
            print(score[i].item(), data[pid[i].item()][0])


if __name__ == '__main__':
    main()
