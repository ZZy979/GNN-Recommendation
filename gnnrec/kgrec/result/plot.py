import matplotlib.pyplot as plt
import pandas as pd

from gnnrec.config import BASE_DIR
from gnnrec.kgrec.utils import K

OUTPUT_DIR = BASE_DIR / 'output'
RESULT_DIR = BASE_DIR / 'gnnrec/kgrec/result'


def plot_rank():
    df = pd.read_csv(RESULT_DIR / 'rank.csv')
    metrics = ['nDCG@k', 'Recall@k']
    models = ['SciBERT', 'KGCN', 'GARec']
    formats = ['.-', '*--', 'x-.']
    n = len(K)
    for i, metric in enumerate(metrics):
        fig, ax = plt.subplots()
        for model, fmt in zip(models, formats):
            ax.plot(K, df[model][i * n:(i + 1) * n].to_numpy(), fmt, label=model)
        ax.set_xlabel('k')
        ax.set_xticks(K, K)
        ax.legend()
        ax.set_title(metric)
        fig.savefig(RESULT_DIR / f'rank_{metric[:-2]}.png')


def main():
    plot_rank()


if __name__ == '__main__':
    main()
