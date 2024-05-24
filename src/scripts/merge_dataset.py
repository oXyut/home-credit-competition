import gc
from itertools import combinations
import pathlib
from typing import Dict, List

import numpy as np
import pandas as pd
import polars as pl
from polars import selectors as cs


def merge_dataset(
        base_data: pl.DataFrame,
        depth_paths: Dict[str, List[pathlib.Path]],
        depth: str = '012',
    ) -> pd.DataFrame:
    
    assert depth in ['0', '1', '2', '012']
    
    drop_features = [
        'employername_160M',
        'name_4527232M',
        'name_4917606M',
        'empls_employer_name_740M',
        'registaddr_zipcode_184M',
        'contaddr_zipcode_807M',
        'empladdr_zipcode_114M',
        'profession_152M',
        'addres_zip_823M',
    ]

    aggs = [
        cs.ends_with('P', 'A', 'D', 'M', 'T', 'L').last().name.prefix('last_'),
        cs.ends_with('P', 'A', 'D', 'M', 'T', 'L').max().name.prefix('max_'),
        cs.ends_with('P', 'A', 'D').mean().name.prefix('mean_'),
        # (cs.ends_with('P', 'A').std() / cs.ends_with('P', 'A')).mean().name.prefix('cv_'),
        # cs.ends_with('P', 'A').sum().name.prefix('sum_'),
        #cs.ends_with('M').mode().first().name.prefix('mode_'),
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
            drop columns
            '''
            drop_features = [
                'employername_160M',
                'name_4527232M',
                'name_4917606M',
                'empls_employer_name_740M',
                'registaddr_zipcode_184M',
                'contaddr_zipcode_807M',
                'empladdr_zipcode_114M',
                'profession_152M',
                'addres_zip_823M',
            ]

            sub_data = sub_data.drop(drop_features)

            '''
            cast raw data dtypes
            '''
            sub_data = sub_data.pipe(cast_dtypes)

            '''
            aggregation
            '''
            if k[-1] == '1':
                sub_data = sub_data.group_by('case_id').agg(aggs).sort('case_id')

            elif k[-1] == '2':
                sub_data = sub_data.group_by(['case_id', 'num_group1']).agg(aggs).group_by('case_id').agg(aggs).sort('case_id')

            '''
            cast aggregated data dtypes
            '''
            sub_data = sub_data.with_columns(
                pl.col(pl.Float32, pl.Float64).cast(pl.Float32),
                pl.col(pl.Int32, pl.Int64).exclude('case_id').cast(pl.Int32)
            )
            
            '''
            rename columns
            '''
            sub_data = sub_data.rename(lambda c: rename_column(c, k[-1]))

            '''
            drop num_groupN features
            '''
            sub_data = sub_data.drop([col for col in sub_data.columns if 'num_group' in col])

            depth_data.append(sub_data)
            print(f'\t{sub_data.shape}')
            
            del sub_data
            gc.collect()
        
        depth_data = pl.concat(depth_data, how='vertical_relaxed')
        base_data = base_data.join(depth_data, how='left', on='case_id', suffix=f'_{i}')
        
        del depth_data
        gc.collect()

    '''
    add new features
    '''
    depth_0_P_high_fimp_features = [
        'avgdpdtolclosure24_3658938P_0',
        'maxdbddpdtollast12m_3658940P_0',
        'maxdpdlast3m_392P_0',
    ]

    depth_0_A_high_fimp_features = [
        'price_1097A_0',
        'pmtssum_45A_0',
        'annuity_780A_0',
        'credamount_770A_0',
    ]

    depth_0_L_high_fimp_features = [
        'pmtnum_254L_0',
        'mobilephncnt_593L_0',
        'days180_256L_0',
        'days120_123L_0',
        'eir_270L_0',
        'numrejects9m_859L_0',
        'isbidproduct_1095L_0',
        'days90_310L_0',
        'days360_512L_0',
        'pctinstlsallpaidlate1d_3546856L_0',
        'numinstpaidearly3d_3546850L_0',
        'pmtscount_423L_0',
        'numinstunpaidmax_3546851L_0',
        'cntpmts24_3658933L_0',
        'numinstlsallpaid_934L_0',
    ]

    P_aggs = []
    for col1, col2 in combinations(depth_0_P_high_fimp_features, 2):
        P_aggs.append((pl.col(col1) - pl.col(col2)).alias(f'diff_{col1}_{col2}'))
        P_aggs.append((pl.col(col1) + pl.col(col2)).alias(f'sum_{col1}_{col2}'))

    A_aggs = []
    for col1, col2 in combinations(depth_0_A_high_fimp_features, 2):
        A_aggs.append((pl.col(col1) - pl.col(col2)).alias(f'diff_{col1}_{col2}'))
        A_aggs.append((pl.col(col1) + pl.col(col2)).alias(f'sum_{col1}_{col2}'))

    L_aggs = []
    for col1, col2 in combinations(depth_0_L_high_fimp_features, 2):
        L_aggs.append((pl.col(col1) - pl.col(col2)).alias(f'diff_{col1}_{col2}'))
        L_aggs.append((pl.col(col1) + pl.col(col2)).alias(f'sum_{col1}_{col2}'))

    base_data = base_data.with_columns([*P_aggs, *A_aggs, *L_aggs])

    '''
    process date_decision
    '''
    base_data = base_data.pipe(process_date)

    '''
    convert polars DataFrame into pandas DataFrame
    '''
    base_data = base_data.to_pandas()

    return base_data


def cast_dtypes(df):
    df = df.with_columns(cs.ends_with('P', 'A').cast(pl.Float64))
    df = df.with_columns(cs.ends_with('M').cast(pl.String))
    df = df.with_columns(cs.ends_with('D').cast(pl.Date))
    return df


def rename_column(column, depth):
    if column in ['case_id', 'num_group1', 'num_group2']:
        return column
    else:
        return column + f'_{depth}'
    

def process_date(df):
    df = df.with_columns(
        [
            pl.col('date_decision').dt.year().cast(pl.Int16).name.prefix('year_'),
            pl.col('date_decision').dt.month().cast(pl.Int8).name.prefix('month_'),
            pl.col('date_decision').dt.day().cast(pl.Int8).name.prefix('day_'),
            pl.col('date_decision').dt.weekday().cast(pl.Int8).name.prefix('weekday_'),
        ]
    )

    D_features = df.select(pl.col(pl.Date)).columns
    if 'date_decision' in D_features:
        D_features.remove('date_decision')
    df = df.with_columns(pl.col(D_features).sub(('date_decision')).dt.total_days().cast(pl.Float32))

    return df