import dgl
import dgl.function as fn
from django.core.management import BaseCommand
from tqdm import trange

from gnnrec.config import DATA_DIR
from gnnrec.kgrec.data import OAGCSDataset
from gnnrec.kgrec.utils import iter_json
from rank.models import Venue, Institution, Field, Author, Paper, Writes


class Command(BaseCommand):
    help = '将oag-cs数据集导入数据库'

    def add_arguments(self, parser):
        parser.add_argument('--batch-size', type=int, default=2000, help='批大小')

    def handle(self, *args, **options):
        batch_size = options['batch_size']
        raw_path = DATA_DIR / 'oag/cs'

        print('正在导入期刊数据...')
        Venue.objects.bulk_create([
            Venue(id=i, name=v['name'])
            for i, v in enumerate(iter_json(raw_path / 'mag_venues.txt'))
        ], batch_size=batch_size)
        vid_map = {v['id']: i for i, v in enumerate(iter_json(raw_path / 'mag_venues.txt'))}

        print('正在导入机构数据...')
        Institution.objects.bulk_create([
            Institution(id=i, name=o['name'])
            for i, o in enumerate(iter_json(raw_path / 'mag_institutions.txt'))
        ], batch_size=batch_size)
        oid_map = {o['id']: i for i, o in enumerate(iter_json(raw_path / 'mag_institutions.txt'))}

        print('正在导入领域数据...')
        Field.objects.bulk_create([
            Field(id=i, name=f['name'])
            for i, f in enumerate(iter_json(raw_path / 'mag_fields.txt'))
        ], batch_size=batch_size)

        data = OAGCSDataset()
        g = data[0]
        apg = dgl.reverse(g['author', 'writes', 'paper'], copy_ndata=False)
        apg.nodes['paper'].data['c'] = g.nodes['paper'].data['citation'].float()
        apg.update_all(fn.copy_u('c', 'm'), fn.sum('m', 'c'))
        author_citation = apg.nodes['author'].data['c'].int().tolist()

        print('正在导入学者数据...')
        Author.objects.bulk_create([
            Author(
                id=i, name=a['name'], n_citation=author_citation[i],
                institution_id=oid_map[a['org']] if a['org'] is not None else None
            ) for i, a in enumerate(iter_json(raw_path / 'mag_authors.txt'))
        ], batch_size=batch_size)

        print('正在导入论文数据...')
        Paper.objects.bulk_create([
            Paper(
                id=i, title=p['title'], venue_id=vid_map[p['venue']], year=p['year'],
                abstract=p['abstract'], n_citation=p['n_citation']
            ) for i, p in enumerate(iter_json(raw_path / 'mag_papers.txt'))
        ], batch_size=batch_size)

        print('正在导入论文关联数据（很慢）...')
        print('writes')
        u, v = g.edges(etype='writes')
        order = g.edges['writes'].data['order']
        edges = list(zip(u.tolist(), v.tolist(), order.tolist()))
        for i in trange(0, len(edges), batch_size):
            Writes.objects.bulk_create([
                Writes(author_id=a, paper_id=p, order=r)
                for a, p, r in edges[i:i + batch_size]
            ])

        print('has_field')
        u, v = g.edges(etype='has_field')
        edges = list(zip(u.tolist(), v.tolist()))
        HasField = Paper.fos.through
        for i in trange(0, len(edges), batch_size):
            HasField.objects.bulk_create([
                HasField(paper_id=p, field_id=f)
                for p, f in edges[i:i + batch_size]
            ])

        print('cites')
        u, v = g.edges(etype='cites')
        edges = list(zip(u.tolist(), v.tolist()))
        Cites = Paper.references.through
        for i in trange(0, len(edges), batch_size):
            Cites.objects.bulk_create([
                Cites(from_paper_id=p, to_paper_id=r)
                for p, r in edges[i:i + batch_size]
            ])
        print('导入完成')
