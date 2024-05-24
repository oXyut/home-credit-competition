import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


class StabilityMetric(object):
    def __init__(self, weeks):
        self.weeks = weeks

    def get_final_error(self, error, weight):
        return error / (weight + 1e-38)

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])

        approx = approxes[0]

        probas = pd.DataFrame({'WEEK_NUM': self.weeks, 'probability': approx, 'target': target})

        gini_per_week = (
            probas
            .groupby('WEEK_NUM')
            .apply(
                lambda g: 2 * roc_auc_score(g['target'], g['probability']) - 1,
                include_groups=False
            )
        )
        print(gini_per_week)
        gini_per_week.name = 'gini'
        gini_per_week = gini_per_week.reset_index().sort_values('WEEK_NUM')

        a, b = np.linalg.lstsq(gini_per_week[['WEEK_NUM']], gini_per_week[['gini']], rcond=None)[0]
        gini_per_week['residuals'] = gini_per_week['gini'] - (a * gini_per_week['WEEK_NUM'] + b)
        stability = gini_per_week['gini'].mean() + 88.0 * min([0, a]) - 0.5 * gini_per_week['residuals'].std()

        return stability