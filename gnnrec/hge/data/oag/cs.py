import json
import os

import dgl
import pandas as pd
import torch
from dgl.data import DGLDataset, extract_archive
from dgl.data.utils import save_graphs, save_info, load_graphs, load_info

from .config import CS_FIELD_L2


class OAGCSDataset(DGLDataset):
    """OAG MAG数据集计算机领域的子集，只有一个异构图

    https://www.aminer.cn/oag-2-1

    统计数据
    -----
    顶点

    * 1973365 author
    * 1478783 paper
    * 10806 venue
    * 13138 institution
    * 34 field

    边

    * 4830908 author-writes->paper
    * 1478783 paper-published_at->paper
    * 2872948 paper-has_field->field
    * 5377236 paper-cites->paper
    * 1528195 author-affiliated_with->institution

    不包含标签（标签由具体子类提供）

    属性
    -----
    * num_classes: int 类别数（具体子类提供）
    * author_names: List[str] 学者姓名
    * paper_titles: List[str] 论文标题
    * venue_names: List[str] 期刊名称
    * inst_names: List[str] 机构名称
    * field_names: List[str] 领域名称

    paper顶点属性
    -----
    * feat: tensor(1478783, 128) 预训练的标题和摘要词向量
    * year: tensor(1478783) 发表年份（1944~2021）
    """

    def __init__(self):
        # TODO 更新下载链接
        super().__init__('oag-cs', 'https://pan.baidu.com/s/10mTwTer21XPzIahn1elzFA')

    def download(self):
        zip_file_path = os.path.join(self.raw_dir, 'oag-cs.zip')
        if not os.path.exists(zip_file_path):
            raise FileNotFoundError('请手动下载文件 {} 提取码：d15v 并保存为 {}'.format(
                self.url, zip_file_path
            ))
        extract_archive(zip_file_path, self.raw_path)

    def save(self):
        save_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'), [self.g])
        save_info(os.path.join(self.save_path, self.name + '_info.pkl'), {
            'author_names': self.author_names,
            'paper_titles': self.paper_titles,
            'venue_names': self.venue_names,
            'inst_names': self.inst_names,
            'field_names': self.field_names
        })

    def load(self):
        graphs, _ = load_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))
        self.g = graphs[0]
        info = load_info(os.path.join(self.save_path, self.name + '_info.pkl'))
        self.author_names = info['author_names']
        self.paper_titles = info['paper_titles']
        self.venue_names = info['venue_names']
        self.inst_names = info['inst_names']
        self.field_names = info['field_names']

    def process(self):
        self._venue_ids, self.venue_names = self._read_venue()
        self._inst_ids, self.inst_names = self._read_institutions()
        self._author_ids, self.author_names, author_inst = self._read_authors()
        paper_author, paper_venue, paper_field, paper_paper, \
            self.paper_titles, paper_abstracts, paper_years = self._read_papers()
        self.field_names = CS_FIELD_L2

        self.g = self._build_graph(paper_author, paper_venue, paper_field, paper_paper, author_inst)
        self.g.nodes['paper'].data['feat'] = torch.load(os.path.join(self.raw_path, 'paper_feat.pkl'))
        self.g.nodes['paper'].data['year'] = torch.tensor(paper_years)

    def _iter_json(self, filename):
        with open(os.path.join(self.raw_path, filename), encoding='utf8') as f:
            for line in f:
                yield json.loads(line)

    def _read_venue(self):
        print('正在读取期刊数据...')
        venue_ids, venue_names = {}, []
        for i, v in enumerate(self._iter_json('mag_venues.txt')):
            venue_ids[v['id']] = i
            venue_names.append(v['name'])
        return venue_ids, venue_names

    def _read_institutions(self):
        print('正在读取机构数据...')
        inst_ids, inst_names = {}, []
        for i, o in enumerate(self._iter_json('mag_institutions.txt')):
            inst_ids[o['id']] = i
            inst_names.append(o['name'])
        return inst_ids, inst_names

    def _read_authors(self):
        print('正在读取学者数据...')
        author_ids, author_names, author_inst = {}, [], []
        for i, a in enumerate(self._iter_json('mag_authors.txt')):
            author_ids[a['id']] = i
            author_names.append(a['name'])
            if a['org'] is not None:
                author_inst.append([i, self._inst_ids[a['org']]])
        return author_ids, author_names, pd.DataFrame(author_inst, columns=['aid', 'oid'])

    def _read_papers(self):
        print('正在读取论文数据...')
        paper_ids, paper_author, paper_venue = {}, [], []
        paper_titles, paper_abstracts, paper_years = [], [], []
        for i, p in enumerate(self._iter_json('mag_papers.txt')):
            paper_ids[p['id']] = i
            paper_author.extend([i, self._author_ids[a]] for a in p['authors'])
            paper_venue.append([i, self._venue_ids[p['venue']]])
            paper_titles.append(p['title'])
            paper_abstracts.append(p['abstract'])
            paper_years.append(p['year'])

        field_ids = {f: i for i, f in enumerate(CS_FIELD_L2)}
        paper_field, paper_paper = [], []
        for i, p in enumerate(self._iter_json('mag_papers.txt')):
            paper_field.extend([i, field_ids[f]] for f in p['fos'])
            paper_paper.extend([i, paper_ids[r]] for r in p['references'] if r in paper_ids)
        return (
            pd.DataFrame(paper_author, columns=['pid', 'aid']),
            pd.DataFrame(paper_venue, columns=['pid', 'vid']),
            pd.DataFrame(paper_field, columns=['pid', 'fid']),
            pd.DataFrame(paper_paper, columns=['pid', 'rid']),
            paper_titles, paper_abstracts, paper_years
        )

    def _build_graph(self, paper_author, paper_venue, paper_field, paper_paper, author_inst):
        print('正在构造异构图...')
        pa_p, pa_a = paper_author['pid'].to_list(), paper_author['aid'].to_list()
        pv_p, pv_v = paper_venue['pid'].to_list(), paper_venue['vid'].to_list()
        pf_p, pf_f = paper_field['pid'].to_list(), paper_field['fid'].to_list()
        pp_p, pp_r = paper_paper['pid'].to_list(), paper_paper['rid'].to_list()
        ai_a, ai_i = author_inst['aid'].to_list(), author_inst['oid'].to_list()
        return dgl.heterograph({
            ('author', 'writes', 'paper'): (pa_a, pa_p),
            ('paper', 'published_at', 'venue'): (pv_p, pv_v),
            ('paper', 'has_field', 'field'): (pf_p, pf_f),
            ('paper', 'cites', 'paper'): (pp_p, pp_r),
            ('author', 'affiliated_with', 'institution'): (ai_a, ai_i)
        })

    def has_cache(self):
        return os.path.exists(os.path.join(self.save_path, self.name + '_dgl_graph.bin')) \
               and os.path.exists(os.path.join(self.save_path, self.name + '_info.pkl'))

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError('This dataset has only one graph')
        return self.g

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        raise NotImplementedError
