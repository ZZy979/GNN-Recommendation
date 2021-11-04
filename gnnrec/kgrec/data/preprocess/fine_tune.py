import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from gnnrec.config import DATA_DIR, MODEL_DIR
from gnnrec.hge.utils import set_random_seed, get_device, accuracy
from gnnrec.kgrec.data import OAGCSContrastDataset
from gnnrec.kgrec.scibert import ContrastiveSciBERT
from gnnrec.kgrec.utils import iter_json


def collate(samples):
    return map(list, zip(*samples))


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)

    raw_file = DATA_DIR / 'oag/cs/mag_papers.txt'
    train_dataset = OAGCSContrastDataset(raw_file, split='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_dataset = OAGCSContrastDataset(raw_file, split='valid')
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
        print('Epoch {:d} | Loss {:.4f} | Train Acc {:.4f} | Val Acc {:.4f}'.format(
            epoch, sum(losses) / len(losses), sum(scores) / len(scores), val_score
        ))
    model_save_path = MODEL_DIR / 'scibert.pt'
    torch.save(model.state_dict(), model_save_path)
    print('模型已保存到', model_save_path)


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
    return (accuracy(logits.argmax(dim=1), labels) + accuracy(logits.argmax(dim=0), labels)) / 2


@torch.no_grad()
def infer(args):
    device = get_device(args.device)
    model = ContrastiveSciBERT(args.num_hidden, args.tau, device).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / 'scibert.pt', map_location=device))
    model.eval()

    raw_path = DATA_DIR / 'oag/cs'
    dataset = OAGCSContrastDataset(raw_path / 'mag_papers.txt', split='all')
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)
    print('正在推断论文向量...')
    h = []
    for titles, _ in tqdm(loader):
        h.append(model.get_embeds(titles).detach().cpu())
    h = torch.cat(h)  # (N_paper, d_hid)
    h = h / h.norm(dim=1, keepdim=True)
    torch.save(h, raw_path / 'paper_feat.pkl')
    print('论文向量已保存到', raw_path / 'paper_feat.pkl')

    fields = [f['name'] for f in iter_json(raw_path / 'mag_fields.txt')]
    loader = DataLoader(fields, batch_size=args.batch_size)
    print('正在推断领域向量...')
    h = []
    for fields in tqdm(loader):
        h.append(model.get_embeds(fields).detach().cpu())
    h = torch.cat(h)  # (N_field, d_hid)
    h = h / h.norm(dim=1, keepdim=True)
    torch.save(h, raw_path / 'field_feat.pkl')
    print('领域向量已保存到', raw_path / 'field_feat.pkl')


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
    train_parser.set_defaults(func=train)

    infer_parser = subparsers.add_parser('infer', help='推断')
    infer_parser.add_argument('--device', type=int, default=0, help='GPU设备')
    infer_parser.add_argument('--num-hidden', type=int, default=128, help='隐藏层维数')
    infer_parser.add_argument('--tau', type=float, default=0.07, help='温度参数')
    infer_parser.add_argument('--batch-size', type=int, default=64, help='批大小')
    infer_parser.set_defaults(func=infer)

    args = parser.parse_args()
    print(args)
    args.func(args)


if __name__ == '__main__':
    main()
