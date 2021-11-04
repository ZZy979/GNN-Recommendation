from torch.utils.data import Dataset

from gnnrec.kgrec.utils import iter_json


class OAGCSContrastDataset(Dataset):
    SPLIT_YEAR = 2016

    def __init__(self, raw_file, split='train'):
        """oag-cs论文标题-关键词对比学习数据集

        由于原始数据不包含关键词，因此使用研究领域（fos字段）作为关键词

        :param raw_file: str 原始论文数据文件
        :param split: str "train", "valid", "all"
        """
        self.titles = []
        self.keywords = []
        for p in iter_json(raw_file):
            if split == 'train' and p['year'] <= self.SPLIT_YEAR \
                    or split == 'valid' and p['year'] > self.SPLIT_YEAR \
                    or split == 'all':
                self.titles.append(p['title'])
                self.keywords.append('; '.join(p['fos']))

    def __getitem__(self, item):
        return self.titles[item], self.keywords[item]

    def __len__(self):
        return len(self.titles)
