import dgl
import torch

from .oagcs import OAGCSDataset


class OAGVenueDataset(OAGCSDataset):
    """oag-cs期刊分类数据集，删除了venue顶点，作为paper顶点的标签

    属性
    -----
    * num_classes: 类别数
    * predict_ntype: 目标顶点类型

    增加的paper顶点属性
    -----
    * label: tensor(N_paper) 论文所属期刊(-1~176)
    * train_mask, val_mask, test_mask: tensor(N_paper) 数量分别为402457, 280762, 255387，划分方式：年份
    """

    def load(self):
        super().load()
        for k in ('train_mask', 'val_mask', 'test_mask'):
            self.g.nodes['paper'].data[k] = self.g.nodes['paper'].data[k].bool()

    def process(self):
        super().process()
        venue_in_degrees = self.g.in_degrees(etype='published_at')
        drop_venue_id = torch.nonzero(venue_in_degrees < 1000, as_tuple=True)[0]
        # 删除论文数1000以下的期刊，剩余360种
        tmp_g = dgl.remove_nodes(self.g, drop_venue_id, 'venue')

        pv_p, pv_v = tmp_g.edges(etype='published_at')
        labels = torch.full((tmp_g.num_nodes('paper'),), -1)
        mask = torch.full((tmp_g.num_nodes('paper'),), False)
        labels[pv_p] = pv_v
        mask[pv_p] = True

        g = dgl.heterograph({etype: tmp_g.edges(etype=etype) for etype in [
            ('author', 'writes', 'paper'), ('paper', 'has_field', 'field'),
            ('paper', 'cites', 'paper'), ('author', 'affiliated_with', 'institution')
        ]})
        for ntype in g.ntypes:
            g.nodes[ntype].data.update(self.g.nodes[ntype].data)
        for etype in g.canonical_etypes:
            g.edges[etype].data.update(self.g.edges[etype].data)

        year = g.nodes['paper'].data['year']
        g.nodes['paper'].data.update({
            'label': labels,
            'train_mask': mask & (year < 2015),
            'val_mask': mask & (year >= 2015) & (year < 2018),
            'test_mask': mask & (year >= 2018)
        })
        self.g = g

    @property
    def name(self):
        return 'oag-venue'

    @property
    def num_classes(self):
        return 360

    @property
    def predict_ntype(self):
        return 'paper'
