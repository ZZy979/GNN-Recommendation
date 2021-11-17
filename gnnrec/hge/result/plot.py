import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gnnrec.config import BASE_DIR

RESULT_DIR = BASE_DIR / 'gnnrec/hge/result'


def plot_param_analysis():
    df = pd.read_csv(RESULT_DIR / 'param_analysis.csv')
    params = ['alpha', 'Tpos', 'dimension']

    for p in params:
        fig, ax = plt.subplots()
        x = df[p].dropna().to_numpy()
        ax.plot(x, df[f'Accuracy_{p}'].dropna().to_numpy(), '.-', label='Accuracy')
        ax.plot(x, df[f'Macro-F1_{p}'].dropna().to_numpy(), '*--', label='Macro-F1')
        ax.set_xlabel(p)
        ax.set_ylabel('Accuracy / Macro-F1')

        ax2 = ax.twinx()
        ax2.plot(x, df[f'Train-time_{p}(h)'].dropna().to_numpy(), 'x-.', label='Train time')
        ax2.set_ylabel('Train time(h)')

        fig.legend(loc='upper center')
        fig.savefig(RESULT_DIR / f'param_analysis_{p}.png')


def plot_ablation_study():
    df = pd.read_csv(RESULT_DIR / 'ablation_study.csv')
    labels = ['Accuracy', 'Macro-F1']
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots()
    for d, model in zip(range(-1, 2), ('RHCO_pg', 'RHCO_sc', 'RHCO')):
        ax.bar(x + d * width, df[model].to_numpy(), width, label=model)
    ax.set_xticks(x, labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULT_DIR / 'ablation_study.png')


def main():
    plot_param_analysis()
    plot_ablation_study()


if __name__ == '__main__':
    main()
