import os

import matplotlib.pyplot as plt
import pandas as pd

from gnnrec.config import BASE_DIR


def plot_param_analysis():
    df = pd.read_csv(os.path.join(BASE_DIR, 'gnnrec/hge/result/param_analysis.csv'))
    params = ['alpha', 'Tpos', 'dimension']

    for p in params:
        fig, ax = plt.subplots()
        x = df[p].dropna().to_numpy()
        acc = df[f'Accuracy_{p}'].dropna().to_numpy()
        f1 = df[f'Macro-F1_{p}'].dropna().to_numpy()
        ax.plot(x, acc, '.-', label='Accuracy')
        ax.plot(x, f1, '.--', label='Macro-F1')
        ax.set_xlabel(p)
        ax.legend()
        fig.savefig(os.path.join(BASE_DIR, f'gnnrec/hge/result/param_analysis_{p}.png'))


def main():
    plot_param_analysis()


if __name__ == '__main__':
    main()
