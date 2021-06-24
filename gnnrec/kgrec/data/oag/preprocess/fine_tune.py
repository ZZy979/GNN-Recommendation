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

    def __init__(self, raw_paper_file):
        self.raw_paper_file = raw_paper_file
        self.field_ids = {f: i for i, f in enumerate(CS_FIELD_L2)}

    def __iter__(self):
        with open(self.raw_paper_file, encoding='utf8') as f:
            for line in f:
                p = json.loads(line)
                yield p['title'] + ' ' + p['abstract'], [self.field_ids[f] for f in p['fos']]

    def __len__(self):
        return 1478783

    @property
    def num_classes(self):
        return 34


def collate(samples):
    return map(list, zip(*samples))


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)

    dataset = OAGCSPaperFieldDataset(args.raw_paper_file)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)
    mlb = MultiLabelBinarizer().fit([list(range(dataset.num_classes))])

    model = nn.Sequential(
        SciBERT(args.num_hidden, device),
        nn.Linear(args.num_hidden, dataset.num_classes)
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps * 0.1, num_training_steps=total_steps
    )
    for epoch in range(args.epochs):
        model.train()
        losses, scores = [], []
        for texts, labels in tqdm(loader):
            logits = model(texts)
            labels = mlb.transform(labels)
            loss = F.binary_cross_entropy_with_logits(logits, torch.from_numpy(labels).float().to(device))

            losses.append(loss.item())
            scores.append(f1_score(labels, logits.detach().cpu().numpy() > 0.5, average='micro'))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        print('Epoch {:d} | Loss {:.4f} | Mirco F1 {:.4f}'.format(
            epoch, sum(losses) / len(losses), sum(scores) / len(scores)
        ))
    print('正在推断...')
    inference(model, loader, args.save_path)
    print('结果已保存到', args.save_path)


@torch.no_grad()
def inference(model, loader, save_path):
    model.eval()
    h = []
    for texts, _ in tqdm(loader):
        h.append(model[0](texts).cpu())
    torch.save(torch.cat(h), save_path)


def main():
    parser = argparse.ArgumentParser(description='通过论文二级领域分类任务fine-tune SciBERT模型')
    parser.add_argument('--seed', type=int, default=42, help='随机数种子')
    parser.add_argument('--device', type=int, default=0, help='GPU设备')
    parser.add_argument('--num-hidden', type=int, default=128, help='隐藏层维数')
    parser.add_argument('--epochs', type=int, default=10, help='训练epoch数')
    parser.add_argument('--batch-size', type=int, default=64, help='批大小')
    parser.add_argument('--lr', type=float, default=5e-5, help='学习率')
    parser.add_argument('raw_paper_file', help='论文原始数据文件路径')
    parser.add_argument('save_path', help='论文向量文件保存路径')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
