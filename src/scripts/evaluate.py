from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import auc, roc_curve, roc_auc_score


class Evaluator:
    def __init__(
            self,
            oof: pd.DataFrame,
            save_path: Optional[Path]=None
        ):

        self.oof = oof
        self.save_path = save_path

        assert 'WEEK_NUM' in self.oof.columns
        assert 'target' in self.oof.columns
        assert 'probability' in self.oof.columns

    def plot_pred(self, is_log: bool=False) -> None:
        _, ax = plt.subplots()
        sns.histplot(data=self.oof, x='probability', hue='target', bins=50, ax=ax)
        if is_log:
            ax.set_yscale('log')
        if self.save_path is not None:
            plt.savefig(Path.joinpath(self.save_path, 'hist_pred.png'))
        plt.show()

    def plot_roc(self) -> None:
        fpr, tpr, _ = roc_curve(self.oof['target'], self.oof['probability'])
        _, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {auc(fpr, tpr):.2f})')
        ax.plot([0, 1], [0, 1], linestyle='--', color='k', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        if self.save_path is not None:
            plt.savefig(Path.joinpath(self.save_path, 'roc_curve.png'))
        plt.show()

    def plot_gini(self) -> Tuple[pd.DataFrame, float]:
        gini_per_week = (
            self.oof.
            groupby('WEEK_NUM')
            .apply(
                lambda g: 2 * roc_auc_score(g['target'], g['probability']) - 1,
                include_groups=False
            )
        )
        gini_per_week.name = 'gini'
        gini_per_week = gini_per_week.reset_index().sort_values('WEEK_NUM')

        linear_regression = LinearRegression()
        linear_regression.fit(gini_per_week[['WEEK_NUM']], gini_per_week[['gini']])
        a = linear_regression.coef_[0].item()
        b = linear_regression.intercept_.item()

        gini_per_week['regression'] = a * gini_per_week['WEEK_NUM'] + b
        gini_per_week['residuals'] = gini_per_week['gini'] - gini_per_week['regression']
        stability = gini_per_week['gini'].mean() + 88.0 * min([0, a]) - 0.5 * gini_per_week['residuals'].std()

        _, ax = plt.subplots()
        ax.scatter(gini_per_week['WEEK_NUM'], gini_per_week['gini'], alpha=0.5, label='Gini coefficient')
        ax.plot(
            a * gini_per_week['WEEK_NUM'] + b,
            label=f'y = {a:.4f}x + {b:.4f}',
            color='tab:orange'
        )
        ax.set(
            xlabel='WEEK_NUM',
            ylabel='Gini coefficient',
            ylim=[0, 1],
            title='stability: {:.4f}'.format(stability)
        )
        ax.legend()
        if self.save_path is not None:
            plt.savefig(Path.joinpath(self.save_path, 'gini_weeks.png'))
        plt.show()

        outcome = pd.DataFrame([[stability, a, b]], columns=['stability', 'slope', 'intercept'])
        return gini_per_week, outcome