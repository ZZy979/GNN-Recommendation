import argparse
import json
import os
import zipfile
from collections import Counter


def lines(root, data_type):
    extract_dir = os.path.join(root, 'tmp')
    for zip_file in filter(lambda z: z.startswith(f'mag_{data_type}s'), os.listdir(root)):
        with zipfile.ZipFile(os.path.join(root, zip_file), 'r') as z:
            for txt_file in z.namelist():
                print(f'{zip_file}\\{txt_file}')
                z.extract(txt_file, extract_dir)
                with open(os.path.join(extract_dir, txt_file), encoding='utf8') as f:
                    yield from f
                os.remove(os.path.join(extract_dir, txt_file))


def analyze(args):
    n = 0
    max_fields = set()
    min_fields = None
    field_count = Counter()
    sample = ''
    for line in lines(args.root, args.type):
        d = json.loads(line)
        n += 1
        max_fields.update(d)
        if min_fields is None:
            min_fields = set(d)
        else:
            min_fields.intersection_update(d)
        field_count.update(d.keys())
        if len(d) == len(max_fields):
            sample = line
    print('数据类型：', args.type)
    print('总量：', n)
    print('最大字段集合：', max_fields)
    print('最小字段集合：', min_fields)
    print('字段出现比例：', {k: v / n for k, v in field_count.items()})
    print('示例：', sample)


def main():
    parser = argparse.ArgumentParser(description='分析OAG MAG数据集的字段')
    parser.add_argument('type', choices=['author', 'paper', 'venue', 'affiliation'], help='数据类型')
    parser.add_argument('root', help='原始文件所在目录')
    args = parser.parse_args()
    analyze(args)


if __name__ == '__main__':
    main()
