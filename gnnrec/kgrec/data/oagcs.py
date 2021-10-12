import json
import os

import dgl
import pandas as pd
import torch
from dgl.data import DGLDataset, extract_archive
from dgl.data.utils import save_graphs, load_graphs


class OAGCSDataset(DGLDataset):
    """OAG MAG数据集(https://www.aminer.cn/oag-2-1)计算机领域的子集，只有一个异构图

    统计数据
    -----
    顶点

    * 1598084 author
    * 1108605 paper
    * 9958 venue
    * 12248 institution
    * 99879 field

    边

    * 3433354 author-writes->paper
    * 1108605 paper-published_at->venue
    * 10352775 paper-has_field->field
    * 2440117 paper-cites->paper
    * 1231953 author-affiliated_with->institution

    paper顶点属性
    -----
    * feat: tensor(N_paper, 128) 预训练的标题和摘要词向量
    * year: tensor(N_paper) 发表年份（1937~2021）
    * 不包含标签

    field顶点属性
    -----
    * feat: tensor(N_field, 128) 预训练的领域向量
    """

    def __init__(self):
        super().__init__('oag-cs', 'https://pan.baidu.com/s/1EjafRKBBDr96IycoNHDHuQ')

    def download(self):
        zip_file_path = os.path.join(self.raw_dir, 'oag-cs.zip')
        if not os.path.exists(zip_file_path):
            raise FileNotFoundError('请手动下载文件 {} 提取码：k1v4 并保存为 {}'.format(
                self.url, zip_file_path
            ))
        extract_archive(zip_file_path, self.raw_path)

    def save(self):
        save_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'), [self.g])

    def load(self):
        self.g = load_graphs(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))[0][0]

    def process(self):
        self._vid_map = self._read_venues()  # {原始id: 顶点id}
        self._oid_map = self._read_institutions()  # {原始id: 顶点id}
        self._fid_map = self._read_fields()  # {领域名称: 顶点id}
        self._aid_map, author_inst = self._read_authors()  # {原始id: 顶点id}, R(aid, oid)
        # [原始id], R(pid, aid), R(pid, vid), R(pid, fid), R(pid, rid), [年份]
        paper_author, paper_venue, paper_field, paper_ref, paper_years = self._read_papers()
        self.g = self._build_graph(paper_author, paper_venue, paper_field, paper_ref, author_inst, paper_years)

    def _iter_json(self, filename):
        with open(os.path.join(self.raw_path, filename), encoding='utf8') as f:
            for line in f:
                yield json.loads(line)

    def _read_venues(self):
        print('正在读取期刊数据...')
        # 行号=索引=顶点id
        return {v['id']: i for i, v in enumerate(self._iter_json('mag_venues.txt'))}

    def _read_institutions(self):
        print('正在读取机构数据...')
        return {o['id']: i for i, o in enumerate(self._iter_json('mag_institutions.txt'))}

    def _read_fields(self):
        print('正在读取领域数据...')
        return {f['name']: f['id'] for f in self._iter_json('mag_fields.txt')}

    def _read_authors(self):
        print('正在读取学者数据...')
        author_id_map, author_inst = {}, []
        for i, a in enumerate(self._iter_json('mag_authors.txt')):
            author_id_map[a['id']] = i
            if a['org'] is not None:
                author_inst.append([i, self._oid_map[a['org']]])
        return author_id_map, pd.DataFrame(author_inst, columns=['aid', 'oid'])

    def _read_papers(self):
        print('正在读取论文数据...')
        paper_id_map, paper_author, paper_venue, paper_field, paper_years = {}, [], [], [], []
        for i, p in enumerate(self._iter_json('mag_papers.txt')):
            paper_id_map[p['id']] = i
            paper_author.extend([i, self._aid_map[a]] for a in p['authors'])
            paper_venue.append([i, self._vid_map[p['venue']]])
            paper_field.extend([i, self._fid_map[f]] for f in p['fos'] if f in self._fid_map)
            paper_years.append(p['year'])

        paper_ref = []
        for i, p in enumerate(self._iter_json('mag_papers.txt')):
            paper_ref.extend([i, paper_id_map[r]] for r in p['references'] if r in paper_id_map)
        return (
            pd.DataFrame(paper_author, columns=['pid', 'aid']).drop_duplicates(),
            pd.DataFrame(paper_venue, columns=['pid', 'vid']),
            pd.DataFrame(paper_field, columns=['pid', 'fid']),
            pd.DataFrame(paper_ref, columns=['pid', 'rid']),
            paper_years
        )

    def _build_graph(self, paper_author, paper_venue, paper_field, paper_ref, author_inst, paper_years):
        print('正在构造异构图...')
        pa_p, pa_a = paper_author['pid'].to_list(), paper_author['aid'].to_list()
        pv_p, pv_v = paper_venue['pid'].to_list(), paper_venue['vid'].to_list()
        pf_p, pf_f = paper_field['pid'].to_list(), paper_field['fid'].to_list()
        pp_p, pp_r = paper_ref['pid'].to_list(), paper_ref['rid'].to_list()
        ai_a, ai_i = author_inst['aid'].to_list(), author_inst['oid'].to_list()
        g = dgl.heterograph({
            ('author', 'writes', 'paper'): (pa_a, pa_p),
            ('paper', 'published_at', 'venue'): (pv_p, pv_v),
            ('paper', 'has_field', 'field'): (pf_p, pf_f),
            ('paper', 'cites', 'paper'): (pp_p, pp_r),
            ('author', 'affiliated_with', 'institution'): (ai_a, ai_i)
        })
        g.nodes['paper'].data['feat'] = torch.load(os.path.join(self.raw_path, 'paper_feat.pkl'))
        g.nodes['paper'].data['year'] = torch.tensor(paper_years)
        g.nodes['field'].data['feat'] = torch.load(os.path.join(self.raw_path, 'field_feat.pkl'))
        return g

    def has_cache(self):
        return os.path.exists(os.path.join(self.save_path, self.name + '_dgl_graph.bin'))

    def __getitem__(self, idx):
        if idx != 0:
            raise IndexError('This dataset has only one graph')
        return self.g

    def __len__(self):
        return 1
