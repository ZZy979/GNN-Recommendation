import dgl
import torch
from dgl.utils import extract_node_subframes, set_new_frames

from . import OAGCSDataset


class OAGCSVenueDataset(OAGCSDataset):
    """oag-cs期刊分类数据集，删除了venue顶点，作为paper顶点的标签

    属性
    -----
    * num_classes: 类别数

    增加的paper顶点属性
    -----
    * label: tensor(N_paper) 论文所属期刊(-1~176)
    * train_mask, val_mask, test_mask: tensor(N_paper) 数量分别为261837, 111863, 65131，划分方式：年份
    """

    def load(self):
        super().load()
        for k in ('train_mask', 'val_mask', 'test_mask'):
            self.g.nodes['paper'].data[k] = self.g.nodes['paper'].data[k].bool()

    def process(self):
        super().process()
        venue_in_degrees = self.g.in_degrees(etype='published_at')
        drop_venue_id = torch.nonzero(venue_in_degrees < 1000, as_tuple=True)[0]
        # 删除论文数1000以下的期刊，剩余177种
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
        node_frames = extract_node_subframes(self.g, None)
        del node_frames[self.g.ntypes.index('venue')]
        set_new_frames(g, node_frames=node_frames)

        year = g.nodes['paper'].data['year']
        g.nodes['paper'].data.update({
            'label': labels,
            'train_mask': mask & (year < 2012),
            'val_mask': mask & (year >= 2012) & (year < 2017),
            'test_mask': mask & (year >= 2017)
        })
        self.g = g

    @property
    def name(self):
        return 'oag-cs-venue'

    @property
    def num_classes(self):
        return 177
