import dgl
import dgl.function as fn
import torch

from .oagcs import OAGCSDataset


class OAGCoreDataset(OAGCSDataset):
    """oag-cs核心数据集，仅保留引用数对数和>=10的学者及其关联的其他顶点

    统计数据
    -----
    顶点

    * 160350 author
    * 1302162 paper
    * 8048 venue
    * 5556 institution
    * 98560 field

    边

    * 3058876 author-writes->paper
    * 1302162 paper-published_at->venue
    * 12217340 paper-has_field->field
    * 7555285 paper-cites->paper
    * 158685 author-affiliated_with->institution

    增加的顶点属性
    -----
    * dgl.NID: tensor(N_i) 原始顶点id
    """

    def process(self):
        super().process()
        g = self.g
        g.nodes['paper'].data['citation'] = g.nodes['paper'].data['citation'].float().log1p()
        apg = dgl.reverse(g['author', 'writes', 'paper'])
        apg.update_all(fn.copy_u('citation', 'c'), fn.sum('c', 'c'))
        author_citation = apg.nodes['author'].data['c']

        keep_authors = (author_citation >= 10).nonzero(as_tuple=True)[0].tolist()
        drop_authors = torch.tensor(list(set(range(g.num_nodes('author'))) - set(keep_authors)))
        g = dgl.remove_nodes(g, drop_authors, 'author', True)
        nid = {'author': g.nodes['author'].data[dgl.NID]}

        for ntype, etype in [
            ('institution', 'affiliated_with'), ('paper', 'writes'),
            ('field', 'has_field'), ('venue', 'published_at')
        ]:
            drop_nodes = torch.nonzero(g.in_degrees(etype=etype) == 0, as_tuple=True)[0]
            g = dgl.remove_nodes(g, drop_nodes, ntype, store_ids=True)
            nid[ntype] = g.nodes[ntype].data[dgl.NID]
        g.ndata[dgl.NID] = nid
        for etype in g.etypes:
            del g.edges[etype].data[dgl.EID]
        self.g = g

    @property
    def name(self):
        return 'oag-core'
