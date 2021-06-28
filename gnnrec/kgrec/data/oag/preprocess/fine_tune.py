import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import IterableDataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

from gnnrec.kgrec.data.oag.config import CS_FIELD_L2
from gnnrec.hge.utils import set_random_seed, get_device


class SciBERT(nn.Module):

    def __init__(self, out_dim, device):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased').to(device)
        self.linear = nn.Linear(self.model.config.hidden_size, out_dim)

    def forward(self, text):
        encoded = self.tokenizer(
            text, padding='max_length', truncation=True, max_length=256, return_tensors='pt'
        ).to(self.device)
        return self.linear(self.model(**encoded).pooler_output)


class OAGCSPaperFieldDataset(IterableDataset):

    def __init__(self, raw_paper_file, split='train'):
        """用于fine-tune SciBERT模型的OAG论文领域数据集

        :param raw_paper_file: str 原始论文数据文件
        :param split: str "train", "valid", "all"
        """
        self.raw_paper_file = raw_paper_file
        self.split = split
        self._split_year = 2016
        self.field_ids = {f: i for i, f in enumerate(CS_FIELD_L2)}

    def __iter__(self):
        with open(self.raw_paper_file, encoding='utf8') as f:
            for line in f:
                p = json.loads(line)
                if self.split == 'train' and p['year'] <= self._split_year \
                        or self.split == 'valid' and p['year'] > self._split_year \
                        or self.split == 'all':
                    yield p['title'] + ' ' + p['abstract'], [self.field_ids[f] for f in p['fos']]

    def __len__(self):
        return 1189957 if self.split == 'train' else 288826 if self.split == 'valid' else 1478783

    @property
    def num_classes(self):
        return 34


def collate(samples):
    return map(list, zip(*samples))


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)

    train_dataset = OAGCSPaperFieldDataset(args.raw_paper_file, split='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate)
    valid_dataset = OAGCSPaperFieldDataset(args.raw_paper_file, split='valid')
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=collate)
    mlb = MultiLabelBinarizer().fit([list(range(train_dataset.num_classes))])

    model = nn.Sequential(
        SciBERT(args.num_hidden, device),
        nn.Linear(args.num_hidden, train_dataset.num_classes)
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps * 0.1, num_training_steps=total_steps
    )
    for epoch in range(args.epochs):
        model.train()
        losses, scores = [], []
        for texts, labels in tqdm(train_loader):
            logits = model(texts)
            labels = mlb.transform(labels)
            loss = F.binary_cross_entropy_with_logits(logits, torch.from_numpy(labels).float().to(device))

            losses.append(loss.item())
            scores.append(micro_f1(logits, labels))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        val_score = evaluate(valid_loader, model, mlb)
        print('Epoch {:d} | Train Loss {:.4f} | Train Mirco F1 {:.4f} | Val Mirco F1 {:.4f}'.format(
            epoch, sum(losses) / len(losses), sum(scores) / len(scores), val_score
        ))
    torch.save(model[0].cpu(), args.model_save_path)
    print('模型已保存到', args.model_save_path)


def micro_f1(logits, labels):
    return f1_score(labels, logits.detach().cpu().numpy() > 0.5, average='micro')


@torch.no_grad()
def evaluate(loader, model, mlb):
    model.eval()
    scores = []
    for texts, labels in loader:
        logits = model(texts)
        labels = mlb.transform(labels)
        scores.append(micro_f1(logits, labels))
    return sum(scores) / len(scores)


@torch.no_grad()
def inference(args):
    device = get_device(args.device)
    dataset = OAGCSPaperFieldDataset(args.raw_paper_file, split='all')
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)
    model = torch.load(args.model_path).to(device)
    print('正在推断...')
    model.eval()
    h = []
    for texts, _ in tqdm(loader):
        h.append(model(texts).cpu())
    torch.save(torch.cat(h), args.vec_save_path)
    print('结果已保存到', args.vec_save_path)


def main():
    parser = argparse.ArgumentParser(description='通过论文二级领域分类任务fine-tune SciBERT模型')
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser('train', help='训练')
    train_parser.add_argument('--seed', type=int, default=42, help='随机数种子')
    train_parser.add_argument('--device', type=int, default=0, help='GPU设备')
    train_parser.add_argument('--num-hidden', type=int, default=128, help='隐藏层维数')
    train_parser.add_argument('--epochs', type=int, default=5, help='训练epoch数')
    train_parser.add_argument('--batch-size', type=int, default=64, help='批大小')
    train_parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
    train_parser.add_argument('raw_paper_file', help='论文原始数据文件路径')
    train_parser.add_argument('model_save_path', help='模型文件保存路径')
    train_parser.set_defaults(func=train)

    infer_parser = subparsers.add_parser('infer', help='推断')
    infer_parser.add_argument('--device', type=int, default=0, help='GPU设备')
    infer_parser.add_argument('--batch-size', type=int, default=64, help='批大小')
    infer_parser.add_argument('raw_paper_file', help='论文原始数据文件路径')
    infer_parser.add_argument('model_path', help='模型文件路径')
    infer_parser.add_argument('vec_save_path', help='论文向量文件保存路径')
    infer_parser.set_defaults(func=inference)

    args = parser.parse_args()
    print(args)
    args.func(args)


if __name__ == '__main__':
    main()
