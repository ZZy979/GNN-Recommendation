import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from gnnrec.hge.utils import set_random_seed, get_device
from gnnrec.kgrec.data import OAGCSContrastDataset
from gnnrec.kgrec.scibert import ContrastiveSciBERT


def collate(samples):
    return map(list, zip(*samples))


def train(args):
    set_random_seed(args.seed)
    device = get_device(args.device)

    train_dataset = OAGCSContrastDataset(args.raw_file, split='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    valid_dataset = OAGCSContrastDataset(args.raw_file, split='valid')
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
    dataset = OAGCSContrastDataset(args.raw_file, split='all')
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)
    model = ContrastiveSciBERT(args.num_hidden, args.tau, device).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print('正在推断...')
    h = []
    for titles, _ in tqdm(loader):
        h.append(model.get_embeds(titles).detach().cpu())
    h = torch.cat(h)  # (N_paper, d_hid)
    h = h / h.norm(dim=1, keepdim=True)
    torch.save(h, args.vec_save_path)
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
