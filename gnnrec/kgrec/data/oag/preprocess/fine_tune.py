import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from gnnrec.hge.utils import set_random_seed, get_device


class ContrastiveSciBERT(nn.Module):

    def __init__(self, out_dim, tau, device):
        """用于对比学习的SciBERT模型

        :param out_dim: int 输出特征维数
        :param tau: float 温度参数τ
        :param device: torch.device
        """
        super().__init__()
        self.tau = tau
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(device)
        self.linear = nn.Linear(self.model.config.hidden_size, out_dim)

    def get_embeds(self, texts, max_length=64):
        """将文本编码为向量

        :param texts: List[str] 输入文本列表，长度为N
        :param max_length: int, optional padding最大长度，默认为64
        :return: tensor(N, d_out)
        """
        encoded = self.tokenizer(
            texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt'
        ).to(self.device)
        return self.linear(self.model(**encoded).pooler_output)

    def calc_sim(self, texts_a, texts_b):
        """计算两组文本的相似度

        :param texts_a: List[str] 输入文本A列表，长度为N
        :param texts_b: List[str] 输入文本B列表，长度为N
        :return: tensor(N, N) 相似度矩阵，S[i, j] = cos(a[i], b[j]) / τ
        """
        embeds_a = self.get_embeds(texts_a)  # (N, d_out)
        embeds_b = self.get_embeds(texts_b)  # (N, d_out)
        embeds_a = embeds_a / embeds_a.norm(dim=1, keepdim=True)
        embeds_b = embeds_b / embeds_b.norm(dim=1, keepdim=True)
        return embeds_a @ embeds_b.t() / self.tau

    def forward(self, texts_a, texts_b):
        """计算两组文本的对比损失

        :param texts_a: List[str] 输入文本A列表，长度为N
        :param texts_b: List[str] 输入文本B列表，长度为N
        :return: tensor(N, N), float A对B的相似度矩阵，对比损失
        """
        # logits_ab等价于预测概率，对比损失等价于交叉熵损失
        logits_ab = self.calc_sim(texts_a, texts_b)
        logits_ba = logits_ab.t()
        labels = torch.arange(len(texts_a), device=self.device)
        loss_ab = F.cross_entropy(logits_ab, labels)
        loss_ba = F.cross_entropy(logits_ba, labels)
        return logits_ab, (loss_ab + loss_ba) / 2


class OAGCSPaperTitleKeywordsDataset(Dataset):
    SPLIT_YEAR = 2016

    def __init__(self, raw_file, split='train'):
        """oag-cs论文标题和关键词数据集

        :param raw_file: str 原始论文数据文件
        :param split: str "train", "valid", "all"
        """
        self.titles = []
        self.keywords = []
        with open(raw_file, encoding='utf8') as f:
            for line in f:
                p = json.loads(line)
                if split == 'train' and p['year'] <= self.SPLIT_YEAR \
                        or split == 'valid' and p['year'] > self.SPLIT_YEAR \
                        or split == 'all':
                    self.titles.append(p['title'])
                    self.keywords.append(p['keywords'])

    def __getitem__(self, item):
        return self.titles[item], self.keywords[item]

    def __len__(self):
        return len(self.titles)


def collate(samples):
    return map(list, zip(*samples))


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)

    train_dataset = OAGCSPaperTitleKeywordsDataset(args.raw_file, split='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_dataset = OAGCSPaperTitleKeywordsDataset(args.raw_file, split='valid')
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    model = ContrastiveSciBERT(args.num_hidden, args.tau, device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps * 0.1, num_training_steps=total_steps
    )
    for epoch in range(args.epochs):
        model.train()
        losses, scores = [], []
        for titles, keywords in tqdm(train_loader):
            logits, loss = model(titles, keywords)
            labels = torch.arange(len(titles), device=device)
            losses.append(loss.item())
            scores.append(score(logits, labels))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        val_score = evaluate(valid_loader, model, device)
        print('Epoch {:d} | Train Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f}'.format(
            epoch, sum(losses) / len(losses), sum(scores) / len(scores), val_score
        ))
    torch.save(model.state_dict(), args.model_save_path)
    print('模型已保存到', args.model_save_path)


@torch.no_grad()
def evaluate(loader, model, device):
    model.eval()
    scores = []
    for titles, keywords in tqdm(loader):
        logits = model.calc_sim(titles, keywords)
        labels = torch.arange(len(titles), device=device)
        scores.append(score(logits, labels))
    return sum(scores) / len(scores)


def score(logits, labels):
    return (accuracy(logits, labels) + accuracy(logits.t(), labels)) / 2


def accuracy(logits, labels):
    return torch.sum(torch.argmax(logits, dim=1) == labels).item() * 1.0 / len(labels)


@torch.no_grad()
def infer(args):
    device = get_device(args.device)
    dataset = OAGCSPaperTitleKeywordsDataset(args.raw_file, split='all')
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)
    model = ContrastiveSciBERT(args.num_hidden, args.tau, device).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print('正在推断...')
    h = []
    for titles, _ in tqdm(loader):
        h.append(model.get_embeds(titles).detach().cpu())
    torch.save(torch.cat(h), args.vec_save_path)
    print('结果已保存到', args.vec_save_path)


def main():
    parser = argparse.ArgumentParser(description='通过论文标题和关键词的对比学习对SciBERT模型进行fine-tune')
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train', help='训练')
    train_parser.add_argument('--seed', type=int, default=42, help='随机数种子')
    train_parser.add_argument('--device', type=int, default=0, help='GPU设备')
    train_parser.add_argument('--num-hidden', type=int, default=128, help='隐藏层维数')
    train_parser.add_argument('--tau', type=float, default=0.07, help='温度参数')
    train_parser.add_argument('--epochs', type=int, default=5, help='训练epoch数')
    train_parser.add_argument('--batch-size', type=int, default=64, help='批大小')
    train_parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
    train_parser.add_argument('raw_file', help='原始论文数据文件')
    train_parser.add_argument('model_save_path', help='模型保存路径')
    train_parser.set_defaults(func=train)

    infer_parser = subparsers.add_parser('infer', help='推断')
    infer_parser.add_argument('--device', type=int, default=0, help='GPU设备')
    infer_parser.add_argument('--num-hidden', type=int, default=128, help='隐藏层维数')
    infer_parser.add_argument('--tau', type=float, default=0.07, help='温度参数')
    infer_parser.add_argument('--batch-size', type=int, default=64, help='批大小')
    infer_parser.add_argument('raw_file', help='原始论文数据文件')
    infer_parser.add_argument('model_path', help='模型文件路径')
    infer_parser.add_argument('vec_save_path', help='论文向量文件保存路径')
    infer_parser.set_defaults(func=infer)

    args = parser.parse_args()
    print(args)
    args.func(args)


if __name__ == '__main__':
    main()
