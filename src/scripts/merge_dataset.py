import gc
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
        pl.all().exclude(pl.String).max().cast(pl.Float32).name.prefix('max_'),
        #pl.all().exclude(pl.String).min().cast(pl.Float32).name.prefix('min_'),
        #pl.col(pl.String).fill_null('none').sum(),
    ]

    drop_columns = [
        'empls_employer_name_740M',
        'purposeofcred_722M',
        'subjectrole_326M',
        'contracttype_653M',
        'empls_economicalst_849M',
        'pmtmethod_731M',
        'subjectrole_43M',
        'subjectroles_name_541M',
        'periodicityofpmts_997M',
        'subjectrole_93M',
        'maritalst_893M',
        'classificationofcontr_1114M',
        'subjectrole_182M',
        'profession_152M',
        'education_88M',
        'conts_role_79M',
        'rejectreasonclient_4145042M',
    ]
    
    for i, (k, path_list) in enumerate(depth_paths.items()):
        
        if depth == '012':
            pass
        elif depth != k[-1]:
            continue
            
        print(f'loading `{k}`')
        depth_data = []
        for p in path_list:
            sub_data = pl.read_parquet(p)

            '''
            cast dtypes
            '''
            for _, row in bool_features[['Variable', 'cast_dtype']].iterrows():
                col = row['Variable']
                cast_dtype = row['cast_dtype']
                if col in sub_data.columns:
                    sub_data = sub_data.with_columns(pl.col(col).fill_null(np.nan).cast(cast_dtype))
                
            for _, row in float64_features[['Variable', 'cast_dtype']].iterrows():
                col = row['Variable']
                cast_dtype = row['cast_dtype']
                if col in sub_data.columns:
                    sub_data = sub_data.with_columns(pl.col(col).cast(cast_dtype))
                    
            for _, row in string_features[['Variable', 'cast_dtype']].iterrows():
                col = row['Variable']
                cast_dtype = row['cast_dtype']
                if col in sub_data.columns:
                    sub_data = sub_data.with_columns(pl.col(col).cast(cast_dtype))

            for _, row in date_features[['Variable', 'cast_dtype']].iterrows():
                col = row['Variable']
                cast_dtype = row['cast_dtype']
                if col in sub_data.columns:
                    sub_data = sub_data.with_columns(pl.col(col).cast(cast_dtype))

            if k[-1] == '1':
                display(sub_data)
            elif k[-1] == '2':
                display(sub_data)
            
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
            # sub_data = sub_data.group_by('case_id').agg(aggs).sort('case_id')

            if k[-1] == '1':
                sub_data = sub_data.filter(pl.col('num_group1')==0)
            elif k[-1] == '2':
                sub_data = sub_data.filter(pl.col('num_group1')==0, pl.col('num_group2')==0)
            
            '''
            drop num_groups
            '''
            if k[-1] == '1':
                sub_data = sub_data.drop('num_group1')
            elif k[-1] == '2':
                sub_data = sub_data.drop(['num_group1', 'num_group2'])
            
            depth_data.append(sub_data)
            print(f'\t{sub_data.shape}')
            
            del sub_data
            gc.collect()
        
        depth_data = pl.concat(depth_data, how='vertical_relaxed')
        base_data = base_data.join(depth_data, how='left', on='case_id', suffix=f'_{i}')
        
        del depth_data
        gc.collect()

    print(base_data.shape)
    
    '''
    process date features
    '''
    base_data = process_date_features(base_data)

    '''
    convert polars DataFrame to pandas DataFrame
    '''
    base_data = base_data.to_pandas()

    return base_data


def rename_column(column, depth):
    if column in ['case_id', 'num_group1', 'num_group2']:
        return column
    else:
        return column + f'_{depth}'