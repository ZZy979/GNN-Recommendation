import argparse
import logging

from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description='metapath2vec训练word2vec')
    parser.add_argument('--size', type=int, default=128, help='词向量维数')
    parser.add_argument('--workers', type=int, default=3, help='工作线程数')
    parser.add_argument('--iter', type=int, default=10, help='迭代次数')
    parser.add_argument('corpus_file', help='语料库文件路径')
    parser.add_argument('save_path', help='保存word2vec模型文件名')
    args = parser.parse_args()
    print(args)

    model = Word2Vec(
        corpus_file=args.corpus_file, size=args.size, min_count=1,
        workers=args.workers, sg=1, iter=args.iter
    )
    model.save(args.save_path)


if __name__ == '__main__':
    main()
