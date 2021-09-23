import argparse
from collections import Counter

from gnnrec.kgrec.data.preprocess.utils import iter_lines


def analyze(args):
    total = 0
    max_fields = set()
    min_fields = None
    field_count = Counter()
    sample = None
    for d in iter_lines(args.raw_path, args.type):
        total += 1
        keys = [k for k in d if d[k]]
        max_fields.update(keys)
        if min_fields is None:
            min_fields = set(keys)
        else:
            min_fields.intersection_update(keys)
        field_count.update(keys)
        if len(keys) == len(max_fields):
            sample = d
    print('数据类型：', args.type)
    print('总量：', total)
    print('最大字段集合：', max_fields)
    print('最小字段集合：', min_fields)
    print('字段出现比例：', {k: v / total for k, v in field_count.items()})
    print('示例：', sample)


def main():
    parser = argparse.ArgumentParser(description='分析OAG MAG数据集的字段')
    parser.add_argument('type', choices=['author', 'paper', 'venue', 'affiliation'], help='数据类型')
    parser.add_argument('raw_path', help='原始zip文件所在目录')
    args = parser.parse_args()
    analyze(args)


if __name__ == '__main__':
    main()
