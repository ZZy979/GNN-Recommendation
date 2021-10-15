import argparse

from gnnrec.hge.metapath2vec.random_walk import random_walk
from gnnrec.hge.utils import add_reverse_edges
from gnnrec.kgrec.data import OAGCSDataset


def main():
    parser = argparse.ArgumentParser(description='oag-cs数据集 metapath2vec基于元路径的随机游走')
    parser.add_argument('--num-walks', type=int, default=4, help='每个顶点游走次数')
    parser.add_argument('--walk-length', type=int, default=10, help='元路径重复次数')
    parser.add_argument('output_file', help='输出文件名')
    args = parser.parse_args()

    data = OAGCSDataset()
    g = add_reverse_edges(data[0])
    metapaths = {
        'author': ['writes', 'published_at', 'published_at_rev', 'writes_rev'],  # APVPA
        'paper': ['writes_rev', 'writes', 'published_at', 'published_at_rev', 'has_field', 'has_field_rev'],  # PAPVPFP
        'venue': ['published_at_rev', 'writes_rev', 'writes', 'published_at'],  # VPAPV
        'field': ['has_field_rev', 'writes_rev', 'writes', 'has_field'],  # FPAPF
        'institution': ['affiliated_with_rev', 'writes', 'writes_rev', 'affiliated_with']  # IAPAI
    }
    random_walk(g, metapaths, args.num_walks, args.walk_length, args.output_file)


if __name__ == '__main__':
    main()
