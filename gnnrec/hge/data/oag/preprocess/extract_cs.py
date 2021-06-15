import argparse
import json
import os

from gnnrec.hge.data.oag.cs import CS_FIELD_L2
from gnnrec.hge.data.oag.preprocess.iter_raw import iter_lines


def extract_papers(raw_path):
    valid_keys = ['title', 'authors', 'venue', 'year', 'indexed_abstract', 'fos', 'references']
    cs = 'computer science'
    cs_fields = set(CS_FIELD_L2)
    for p in iter_lines(raw_path, 'paper'):
        if all(p.get(k) for k in valid_keys) and any(f['name'] == cs for f in p['fos']) \
                and any(f['name'] in cs_fields for f in p['fos']):
            try:
                abstract = parse_abstract(p['indexed_abstract'])
                if 60 <= len(p['title']) <= 100 and 500 <= len(abstract) <= 1500:
                    yield {
                        'id': p['id'],
                        'title': p['title'],
                        'authors': [a['id'] for a in p['authors']],
                        'venue': p['venue']['id'],
                        'year': p['year'],
                        'abstract': abstract,
                        'fos': [f['name'] for f in p['fos'] if f['name'] in cs_fields],
                        'references': p['references'],
                    }
            except (KeyError, json.JSONDecodeError):
                pass


def parse_abstract(indexed_abstract):
    abst = json.loads(indexed_abstract)
    words = [''] * abst['IndexLength']
    for w, idx in abst['InvertedIndex'].items():
        for i in idx:
            words[i] = w
    return ' '.join(words)


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
    paper_ids, author_ids, venue_ids = set(), set(), set()
    with open(os.path.join(args.output_path, 'mag_papers.txt'), 'w', encoding='utf8') as f:
        for p in extract_papers(args.raw_path):
            paper_ids.add(p['id'])
            author_ids.update(p['authors'])
            venue_ids.add(p['venue'])
            json.dump(p, f)
            f.write('\n')
    print(f'论文抽取完成，已保存到{f.name}')
    print(f'论文数{len(paper_ids)}，学者数{len(author_ids)}，期刊数{len(venue_ids)}')

    print('正在抽取学者...')
    institution_ids = set()
    with open(os.path.join(args.output_path, 'mag_authors.txt'), 'w', encoding='utf8') as f:
        for a in extract_authors(args.raw_path, author_ids):
            if a['org']:
                institution_ids.add(a['org'])
            json.dump(a, f)
            f.write('\n')
    print(f'学者抽取完成，已保存到{f.name}')
    print(f'机构数{len(institution_ids)}')

    print('正在抽取期刊...')
    with open(os.path.join(args.output_path, 'mag_venues.txt'), 'w', encoding='utf8') as f:
        for v in extract_venues(args.raw_path, venue_ids):
            json.dump(v, f)
            f.write('\n')
    print(f'期刊抽取完成，已保存到{f.name}')

    print('正在抽取机构...')
    with open(os.path.join(args.output_path, 'mag_institutions.txt'), 'w', encoding='utf8') as f:
        for i in extract_institutions(args.raw_path, institution_ids):
            json.dump(i, f)
            f.write('\n')
    print(f'机构抽取完成，已保存到{f.name}')


def main():
    parser = argparse.ArgumentParser(description='抽取OAG数据集计算机领域的子集')
    parser.add_argument('raw_path', help='原始zip文件所在目录')
    parser.add_argument('output_path', help='输出目录')
    args = parser.parse_args()
    extract(args)


if __name__ == '__main__':
    main()
