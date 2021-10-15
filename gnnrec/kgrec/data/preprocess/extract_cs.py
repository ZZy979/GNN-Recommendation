import argparse
import json
import os

from gnnrec.kgrec.data.config import CS, CS_FIELD_L2
from gnnrec.kgrec.data.preprocess.utils import iter_lines


def extract_papers(raw_path):
    valid_keys = ['title', 'authors', 'venue', 'year', 'indexed_abstract', 'fos', 'references']
    cs_fields = set(CS_FIELD_L2)
    for p in iter_lines(raw_path, 'paper'):
        if not all(p.get(k) for k in valid_keys):
            continue
        fos = {f['name'] for f in p['fos']}
        abstract = parse_abstract(p['indexed_abstract'])
        if CS in fos and not fos.isdisjoint(cs_fields) \
                and 2010 <= p['year'] <= 2021 \
                and len(p['title']) <= 200 and len(abstract) <= 4000 \
                and 1 <= len(p['authors']) <= 20 and 1 <= len(p['references']) <= 100:
            try:
                yield {
                    'id': p['id'],
                    'title': p['title'],
                    'authors': [a['id'] for a in p['authors']],
                    'venue': p['venue']['id'],
                    'year': p['year'],
                    'abstract': abstract,
                    'fos': list(fos),
                    'references': p['references'],
                    'n_citation': p.get('n_citation', 0),
                }
            except KeyError:
                pass


def parse_abstract(indexed_abstract):
    try:
        abstract = json.loads(indexed_abstract)
        words = [''] * abstract['IndexLength']
        for w, idx in abstract['InvertedIndex'].items():
            for i in idx:
                words[i] = w
        return ' '.join(words)
    except json.JSONDecodeError:
        return ''


def extract_authors(raw_path, author_ids):
    for a in iter_lines(raw_path, 'author'):
        if a['id'] in author_ids:
            yield {
                'id': a['id'],
                'name': a['name'],
                'org': int(a['last_known_aff_id']) if 'last_known_aff_id' in a else None
            }


def extract_venues(raw_path, venue_ids):
    for v in iter_lines(raw_path, 'venue'):
        if v['id'] in venue_ids:
            yield {'id': v['id'], 'name': v['DisplayName']}


def extract_institutions(raw_path, institution_ids):
    for i in iter_lines(raw_path, 'affiliation'):
        if i['id'] in institution_ids:
            yield {'id': i['id'], 'name': i['DisplayName']}


def extract(args):
    print('正在抽取计算机领域的论文...')
    paper_ids, author_ids, venue_ids, fields = set(), set(), set(), set()
    with open(os.path.join(args.output_path, 'mag_papers.txt'), 'w', encoding='utf8') as f:
        for p in extract_papers(args.raw_path):
            paper_ids.add(p['id'])
            author_ids.update(p['authors'])
            venue_ids.add(p['venue'])
            fields.update(p['fos'])
            json.dump(p, f, ensure_ascii=False)
            f.write('\n')
    print(f'论文抽取完成，已保存到{f.name}')
    print(f'论文数{len(paper_ids)}，学者数{len(author_ids)}，期刊数{len(venue_ids)}，领域数{len(fields)}')

    print('正在抽取学者...')
    institution_ids = set()
    with open(os.path.join(args.output_path, 'mag_authors.txt'), 'w', encoding='utf8') as f:
        for a in extract_authors(args.raw_path, author_ids):
            if a['org']:
                institution_ids.add(a['org'])
            json.dump(a, f, ensure_ascii=False)
            f.write('\n')
    print(f'学者抽取完成，已保存到{f.name}')
    print(f'机构数{len(institution_ids)}')

    print('正在抽取期刊...')
    with open(os.path.join(args.output_path, 'mag_venues.txt'), 'w', encoding='utf8') as f:
        for v in extract_venues(args.raw_path, venue_ids):
            json.dump(v, f, ensure_ascii=False)
            f.write('\n')
    print(f'期刊抽取完成，已保存到{f.name}')

    print('正在抽取机构...')
    with open(os.path.join(args.output_path, 'mag_institutions.txt'), 'w', encoding='utf8') as f:
        for i in extract_institutions(args.raw_path, institution_ids):
            json.dump(i, f, ensure_ascii=False)
            f.write('\n')
    print(f'机构抽取完成，已保存到{f.name}')

    print('正在抽取领域...')
    fields.remove(CS)
    fields = sorted(fields)
    with open(os.path.join(args.output_path, 'mag_fields.txt'), 'w', encoding='utf8') as f:
        for i, field in enumerate(fields):
            json.dump({'id': i, 'name': field}, f, ensure_ascii=False)
            f.write('\n')
    print(f'领域抽取完成，已保存到{f.name}')


def main():
    parser = argparse.ArgumentParser(description='抽取OAG数据集计算机领域的子集')
    parser.add_argument('raw_path', help='原始zip文件所在目录')
    parser.add_argument('output_path', help='输出目录')
    args = parser.parse_args()
    extract(args)


if __name__ == '__main__':
    main()
