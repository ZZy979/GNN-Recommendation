import json
import os

from django.core.management import BaseCommand
from django.db import connection
from tqdm import tqdm

from rank.models import Venue, Institution, Field, Author, Paper


class Command(BaseCommand):
    help = '将oag-cs数据集导入数据库'

    def add_arguments(self, parser):
        parser.add_argument('--batch-size', type=int, default=1000, help='批大小')
        parser.add_argument('raw_path', help='原始数据所在目录')

    def handle(self, *args, **options):
        batch_size = options['batch_size']
        raw_path = options['raw_path']

        print('正在导入期刊数据...')
        Venue.objects.bulk_create([
            Venue(id=i, name=v['name'])
            for i, v in enumerate(iter_json(raw_path, 'mag_venues.txt'))
        ], batch_size=batch_size)
        vid_map = {v['id']: i for i, v in enumerate(iter_json(raw_path, 'mag_venues.txt'))}

        print('正在导入机构数据...')
        Institution.objects.bulk_create([
            Institution(id=i, name=o['name'])
            for i, o in enumerate(iter_json(raw_path, 'mag_institutions.txt'))
        ], batch_size=batch_size)
        oid_map = {o['id']: i for i, o in enumerate(iter_json(raw_path, 'mag_institutions.txt'))}

        print('正在导入领域数据...')
        Field.objects.bulk_create([
            Field(id=i, name=f['name'])
            for i, f in enumerate(iter_json(raw_path, 'mag_fields.txt'))
        ], batch_size=batch_size)
        fid_map = {f['name']: i for i, f in enumerate(iter_json(raw_path, 'mag_fields.txt'))}

        print('正在导入学者数据...')
        Author.objects.bulk_create([
            Author(id=i, name=a['name'], institution_id=oid_map[a['org']] if a['org'] is not None else None)
            for i, a in enumerate(iter_json(raw_path, 'mag_authors.txt'))
        ], batch_size=batch_size)
        aid_map = {a['id']: i for i, a in enumerate(iter_json(raw_path, 'mag_authors.txt'))}

        print('正在导入论文数据...')
        Paper.objects.bulk_create([
            Paper(id=i, title=p['title'], venue_id=vid_map[p['venue']], year=p['year'], abstract=p['abstract'])
            for i, p in enumerate(iter_json(raw_path, 'mag_papers.txt'))
        ], batch_size=batch_size)
        pid_map = {p['id']: i for i, p in enumerate(iter_json(raw_path, 'mag_papers.txt'))}

        print('正在导入论文关联数据（很慢）...')
        for i, p in tqdm(enumerate(iter_json(raw_path, 'mag_papers.txt'))):
            paper = Paper.objects.get(id=i)
            paper.authors.set([aid_map[a] for a in p['authors']])
            paper.fos.set([fid_map[f] for f in p['fos'] if f in fid_map])
            paper.references.set([pid_map[r] for r in p['references'] if r in pid_map])
            paper.save()

        print('正在更新论文引用数...')
        with connection.cursor() as cursor:
            cursor.execute(
                'UPDATE rank_paper, (SELECT to_paper_id, COUNT(*) AS n FROM rank_paper_references GROUP BY to_paper_id) r'
                ' SET rank_paper.n_citation = r.n'
                ' WHERE rank_paper.id = r.to_paper_id'
            )
        print('导入完成')


def iter_json(raw_path, filename):
    with open(os.path.join(raw_path, filename), encoding='utf8') as f:
        for line in f:
            yield json.loads(line)
