import gc
from itertools import combinations
import pathlib
from typing import Dict, List

import numpy as np
import pandas as pd
import polars as pl

from .process_features import process_date_features


def merge_dataset(
        base_data: pl.DataFrame,
        depth_paths: Dict[str, List[pathlib.Path]],
        bool_features: pd.DataFrame,
        float64_features: pd.DataFrame,
        string_features: pd.DataFrame,
        date_features: pd.DataFrame,
        useful_features: pd.DataFrame = None,
        depth: str = '012',
    ) -> pd.DataFrame:
    
    assert depth in ['0', '1', '2', '012']

    aggs = [
        pl.col(pl.Float32).max().cast(pl.Float32).name.prefix('max_'),
        pl.col(pl.Float32).median().cast(pl.Float32).name.prefix('median_'),
        pl.col(pl.Float32).sum().cast(pl.Float32).name.prefix('sum_'),
        pl.col(pl.Int32, pl.Int64).max().cast(pl.Int32).name.prefix('max_'),
        pl.col(pl.Int32, pl.Int64).median().cast(pl.Int32).name.prefix('median_'),
        pl.col(pl.Int32, pl.Int64).sum().cast(pl.Int32).name.prefix('sum_'),
        pl.col(pl.Date, pl.String).first().name.prefix('first_'),
    ]
    
    for i, (k, path_list) in enumerate(depth_paths.items()):
        
        if depth == '012':
            pass
        elif depth != k[-1]:
            continue
            
        print(f'loading `{k}`')
        depth_data = []
        for p in path_list:
            sub_data = pl.read_parquet(p).cast({'case_id': pl.Int64})

            '''
            cast dtypes
            '''
            for _, row in bool_features[['Variable', 'cast_dtype']].iterrows():
                col = row['Variable']
                cast_dtype = row['cast_dtype']
                if col in sub_data.columns:
                    sub_data = sub_data.with_columns(pl.col(col).fill_null(2).fill_nan(2).cast(cast_dtype))
                
            for _, row in float64_features[['Variable', 'cast_dtype']].iterrows():
                col = row['Variable']
                cast_dtype = row['cast_dtype']
                if col in sub_data.columns:
                    sub_data = sub_data.with_columns(pl.col(col).cast(cast_dtype))
                    
            for _, row in string_features[['Variable', 'cast_dtype']].iterrows():
                col = row['Variable']
                cast_dtype = row['cast_dtype']
                if col in sub_data.columns:
                    sub_data = sub_data.with_columns(pl.col(col).fill_null('none').cast(cast_dtype))

            for _, row in date_features[['Variable', 'cast_dtype']].iterrows():
                col = row['Variable']
                cast_dtype = row['cast_dtype']
                if col in sub_data.columns:
                    sub_data = sub_data.with_columns(pl.col(col).cast(cast_dtype))

            '''
            rename columns
            '''
            sub_data = sub_data.rename(lambda c: rename_column(c, k[-1]))
            
            '''
            delete useless features
            '''
            if useful_features is not None:
                sub_data = sub_data.drop(
                    [
                        col
                        for col in sub_data.columns
                        if (
                            (col not in useful_features['Variable'].to_list())
                            and
                            (col != 'case_id' and col != 'date_decision' and col != 'MONTH' and col != 'WEEK_NUM')
                        )
                    ]
                )
            
            '''
            aggregation
            '''
            if k[-1] == '1' :
                aggs_12 = aggs + [
                    (pl.col(col).first() / pl.col(col).max()).cast(pl.Float32).name.prefix('ratio_')
                    for col in sub_data.select(pl.col(pl.Float32)).columns
                ]
                
                sub_data = sub_data.group_by('case_id').agg(aggs_12).sort('case_id')
                # for col in sub_data.columns:
                #     if col.startswith('head'):
                #         sub_data = sub_data.with_columns(
                #             pl.col(col).list.get(0).name.prefix('first_'),
                #             pl.col(col).list.get(1).name.prefix('second_'),
                #         )
                #         sub_data = sub_data.drop(col)
            elif k[-1] == '2':
                aggs_12 = aggs + [
                    (pl.col(col).first() / pl.col(col).max()).cast(pl.Float32).name.prefix('ratio_')
                    for col in sub_data.select(pl.col(pl.Float32)).columns
                ]
                sub_data = sub_data.group_by(['case_id', 'num_group1']).agg(aggs_12).group_by('case_id').agg(aggs).sort('case_id')
            sub_data = sub_data.drop([col for col in sub_data.columns if 'num_group' in col])

            depth_data.append(sub_data)
            print(f'\t{sub_data.shape}')
            
            del sub_data
            gc.collect()
        
        depth_data = pl.concat(depth_data, how='vertical_relaxed')
        base_data = base_data.join(depth_data, how='left', on='case_id', suffix=f'_{i}')
        
        del depth_data
        gc.collect()

    # '''
    # add new features
    # '''
    # depth_0_P_high_fimp_features = [
    #     'avgdpdtolclosure24_3658938P_0',
    #     'maxdbddpdtollast12m_3658940P_0',
    #     'maxdpdlast3m_392P_0',
    # ]

    # depth_0_A_high_fimp_features = [
    #     'price_1097A_0',
    #     'pmtssum_45A_0',
    #     'annuity_780A_0',
    #     'credamount_770A_0',
    # ]

    # depth_0_L_high_fimp_features = [
    #     'pmtnum_254L_0',
    #     'mobilephncnt_593L_0',
    #     'days180_256L_0',
    #     'days120_123L_0',
    #     'eir_270L_0',
    #     'numrejects9m_859L_0',
    #     'isbidproduct_1095L_0',
    #     'days90_310L_0',
    #     'days360_512L_0',
    #     'pctinstlsallpaidlate1d_3546856L_0',
    #     'numinstpaidearly3d_3546850L_0',
    #     'pmtscount_423L_0',
    #     'numinstunpaidmax_3546851L_0',
    #     'cntpmts24_3658933L_0',
    #     'numinstlsallpaid_934L_0',
    # ]

    # display(base_data.select(depth_0_L_high_fimp_features))

    # P_aggs = []
    # for col1, col2 in combinations(depth_0_P_high_fimp_features, 2):
    #     P_aggs.append((pl.col(col1) - pl.col(col2)).alias(f'diff_{col1}_{col2}'))
    #     P_aggs.append((pl.col(col1) + pl.col(col2)).alias(f'sum_{col1}_{col2}'))

    # A_aggs = []
    # for col1, col2 in combinations(depth_0_A_high_fimp_features, 2):
    #     A_aggs.append((pl.col(col1) - pl.col(col2)).alias(f'diff_{col1}_{col2}'))
    #     A_aggs.append((pl.col(col1) + pl.col(col2)).alias(f'sum_{col1}_{col2}'))

    # L_aggs = []
    # for col1, col2 in combinations(depth_0_L_high_fimp_features, 2):
    #     L_aggs.append((pl.col(col1) - pl.col(col2)).alias(f'diff_{col1}_{col2}'))
    #     L_aggs.append((pl.col(col1) + pl.col(col2)).alias(f'sum_{col1}_{col2}'))

    # base_data = base_data.with_columns(*A_aggs)

    '''
    process date features
    '''
    base_data = process_date_features(base_data)

    '''
    convert polars DataFrame into pandas DataFrame
    '''
    base_data = base_data.to_pandas()

    return base_data


def rename_column(column, depth):
    if column in ['case_id', 'num_group1', 'num_group2']:
        return column
    else:
        return column + f'_{depth}'