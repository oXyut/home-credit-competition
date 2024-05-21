from typing import List, Tuple, Dict, Literal, Optional
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.linear_model import LinearRegression

DIR_INPUTS = Path('../../data/inputs')
DIR_PATH_CONFIGS = Path('../../data/configs/paths.yaml')
COLUMNS_BASE = [
    'case_id',
    'date_decision',
    'MONTH',
    'WEEK_NUM',
    'target',
]
COLUMNS_BASE_TEST = [
    'case_id',
    'date_decision',
    'MONTH',
    'WEEK_NUM',
]

class DataFrameMerger:
    def __init__(
            self,
            dict_tables_all:Dict,
            dict_tabels_use:Dict,
            category_columns_use:List[str],
            dir_path:Path,
        ):
        self.dict_tables_all = dict_tables_all
        self.dict_tabels_use = dict_tabels_use
        self.category_columns_use = category_columns_use
        self.dir_path = dir_path
        self.df_train = None
        self.df_test = None
        self.df_base = {'train':None, 'test':None}

    def select_columns(self, df:pd.DataFrame, depth:Literal[0, 1, 2])->pd.DataFrame:
        df_ = df.copy()
        cols_use = []
        for cat in self.category_columns_use:
            cols_ = [col for col in df_.columns if col.endswith(cat)]
            cols_use.extend(cols_)
        if depth == 0:
            cols_use.extend(['case_id'])
        elif depth == 1:
            cols_use.extend(['case_id', 'num_group1'])
        elif depth == 2:
            cols_use.extend(['case_id', 'num_group1', 'num_group2'])
        
        return df_[cols_use]

    def pre_aggregate(self, df:pd.DataFrame, depth:Literal[0, 1, 2])->pd.DataFrame:
        if depth == 0:
            return df
        elif depth == 1:
            # print("NOTE: for now, only leave num_group1==0 for depth==1")
            df_ = df.query('num_group1==0')
            df_ = df_.drop(['num_group1'], axis=1)
            return df_
        elif depth == 2:
            # print("NOTE: for now, only leave num_group1==0 and num_group2==0 for depth==2")
            df_ = df.query('num_group1==0 & num_group2==0')
            df_ = df_.drop(['num_group1', 'num_group2'], axis=1)
            return df_

    def add_depth_prefix(self, df:pd.DataFrame, depth:Literal[0, 1, 2])->pd.DataFrame:
        df_ = df.copy()
        if depth == 0:
            return df_
        elif depth == 1:
            df_.columns = [f'1_{col}' if col != 'case_id' else col for col in df_.columns]
            return df_
        elif depth == 2:
            df_.columns = [f'2_{col}' if col != 'case_id' else col for col in df_.columns]
            return df_        


    def load_concat(self, mode:Literal['train', 'test'], is_pre_aggregate:bool=True)->Tuple[pd.DataFrame, Dict[int, Dict[str, pd.DataFrame]]]:
        dict_df = {0:{}, 1:{}, 2:{}}
        df_base = pd.read_csv(Path.joinpath(self.dir_path, mode, f'{mode}_base.csv'))
        self.df_base[mode] = df_base
        print('base', df_base.shape)
        for depth in [0, 1, 2]:
            for k in self.dict_tabels_use[depth]:
                dfs = []
                for p in self.dict_tables_all[depth][k]:
                    df_ = pd.read_csv(Path.joinpath(self.dir_path, mode, f'{mode}_{p}'))
                    df_ = self.select_columns(df_, depth)
                    if is_pre_aggregate:
                        df_ = self.pre_aggregate(df_, depth)
                    dfs.append(df_)
                dfs = pd.concat(dfs, axis=0)
                print(k, dfs.shape)
                dict_df[depth][k] = dfs
                # df_ = pd.concat([pd.read_csv(Path.joinpath(self.dir_path, mode, f'{mode}_{p}')) for p in self.dict_tables_all[depth][k]], axis=0)

        return df_base, dict_df
    
    def aggregate(self, dict_df:Dict[int, Dict[str, pd.DataFrame]])->Dict[int, Dict[str, pd.DataFrame]]:
        # print('Aggregate by case_id for tables with depth=1, 2')
        # print('NOT IMPLEMENTED YET')
        return dict_df
    
    def merge(self, df_base:pd.DataFrame, dict_df:Dict[int, Dict[str, pd.DataFrame]])->pd.DataFrame:
        df = df_base
        for depth in [0, 1, 2]:
            for k in self.dict_tabels_use[depth]:
                df = pd.merge(df, self.add_depth_prefix(dict_df[depth][k], depth), on='case_id', how='left')
        return df
    
def preprocess(df: pd.DataFrame, category_columns_use: List[str], mode: Literal['train', 'test'] = 'train') -> pd.DataFrame:
    df_ = df.copy()

    columns_base = COLUMNS_BASE.copy()
    if mode == 'test':
        columns_base.remove('target')

    cols_wo_base = [col for col in df_.columns if col not in columns_base]
    cols_use = columns_base.copy()

    # 日付関連の特徴量を格納するための空のDataFrameを準備
    date_features = []

    for cat in category_columns_use:
        cols_ = [col for col in cols_wo_base if col.endswith(cat)]
        cols_use.extend(cols_)
        if cat == 'P':  # numerical features
            pass
        elif cat == 'M':  # categorical features
            # df_[cols_] = df_[cols_].astype('category')
            # Label Encoding
            le = LabelEncoder()
            for col in cols_:
                df_[col] = le.fit_transform(df_[col])
            ## ちゃんとLabelEncoder使ったほうがいいかも
        elif cat == 'D':  # date features s.t. YYYY-MM-DD
            for col in cols_:
                df_[col] = pd.to_datetime(df_[col])
                # 日付関連の新しい特徴量を作成
                col_features = pd.DataFrame({
                    f'{col}_year': df_[col].dt.year,
                    f'{col}_month': df_[col].dt.month,
                    f'{col}_day': df_[col].dt.day,
                    f'{col}_weekday': df_[col].dt.weekday
                }, index=df_.index)
                date_features.append(col_features)
                # 元の日付列を使用リストから削除
                if col in cols_use:
                    cols_use.remove(col)
        elif cat == 'T':
            pass
        elif cat == 'L':
            pass

    # 全ての日付関連の特徴量を結合
    if date_features:
        df_date_features = pd.concat(date_features, axis=1)
        df_ = pd.concat([df_, df_date_features], axis=1)
        cols_use.extend(df_date_features.columns)

    # 最終的に使用する列のみを選択
    return df_[cols_use]

class Evaluator:
    def __init__(
            self,
            y_true:pd.DataFrame,
            y_pred:np.ndarray,
            save_path:Optional[Path]=None
        ):
        if len(y_true) != len(y_pred):
            raise ValueError('Length of y_true and y_pred must be same')

        self.y_true = y_true
        self.y_pred = y_pred
        self.save_path = save_path
        self.save_name = self.save_path.name if self.save_path else None


    def plot_pred(self, is_log:bool=False)->None:
        fig, ax = plt.subplots()
        bins = np.linspace(0, 1, 100)
        ax.hist(self.y_pred[self.y_true['target'] == 0], bins=bins, alpha=0.5, label='0')
        ax.hist(self.y_pred[self.y_true['target'] == 1], bins=bins, alpha=0.5, label='1')
        ax.legend()
        if is_log:
            ax.set_yscale('log')
        if self.save_path:
            ax.set_title(self.save_name)
            plt.savefig(Path.joinpath(self.save_path, 'hist_pred.png'))
        plt.show()

    def plot_roc(self)->None:
        fpr, tpr, thresholds = roc_curve(self.y_true['target'], self.y_pred)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {auc(fpr, tpr):.2f})')
        ax.plot([0, 1], [0, 1], linestyle='--', color='k', label='Random')
        ax.set_title(self.save_name)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.legend()
        if self.save_path:
            plt.savefig(Path.joinpath(self.save_path, 'roc_curve.png'))
        plt.show()

    def plot_gini(self)->Tuple[pd.DataFrame, float]:
        gini_weeks = []

        for week in range(0, 92):
            idx_week = (self.y_true['WEEK_NUM'] == week)
            y_pred_week = self.y_pred[idx_week]
            y_test_week = self.y_true[idx_week]
            gini_week = 2 * roc_auc_score(y_test_week['target'], y_pred_week) - 1
            gini_weeks.append(gini_week)

        gini_weeks = np.array(gini_weeks)
        linear_regression = LinearRegression()
        linear_regression.fit(np.arange(0, 92).reshape(-1, 1), gini_weeks)
        a, b = linear_regression.coef_[0], linear_regression.intercept_

        residuals = gini_weeks - linear_regression.predict(np.arange(0, 92).reshape(-1, 1))
        stability = gini_weeks.mean() + 88.0 * np.amin([0, a]) - 0.5 * residuals.std()
        print(stability)

        fig, ax = plt.subplots()
        ax.plot(gini_weeks, marker='o', linestyle='', label='Gini coefficient')
        ax.plot(
            linear_regression.predict(np.arange(0, 92).reshape(-1, 1)),
            label=f'y = {a:.4f}x + {b:.4f}'
        )
        ax.text(0, 0.9, f'stability: {stability:.4f}')
        ax.set(
            title=self.save_name,
            xlabel='WEEK_NUM',
            ylabel='Gini coefficient',
            ylim=[0, 1],
        )
        ax.legend()
        if self.save_path:
            plt.savefig(Path.joinpath(self.save_path, 'gini_weeks.png'))
        plt.show()

        df_gini_weeks = pd.DataFrame({
            'WEEK_NUM': np.arange(0, 92),
            'Gini': gini_weeks,
        })
        return df_gini_weeks, stability

