import json

from torch.utils.data import Dataset


class OAGCSContrastDataset(Dataset):
    SPLIT_YEAR = 2016

    def __init__(self, raw_file, split='train'):
        """oag-cs论文标题-关键词对比学习数据集

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
