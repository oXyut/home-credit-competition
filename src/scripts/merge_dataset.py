import gc
import pathlib
from typing import Dict, List

import pandas as pd
import polars as pl


def merge_dataset(
        base_data: pl.DataFrame,
        depth_paths: Dict[str, List[pathlib.Path]],
        cast_features: Dict[str, object],
        depth: str,
    ) -> pd.DataFrame:
    
    assert depth in ['0', '1', '2', '012']
    
    for i, (k, path_list) in enumerate(depth_paths.items()):
        
        if depth == '012':
            pass
        elif depth != k[-1]:
            continue
            
        print(f'loading `{k}`')
        depth_data = []
        for p in path_list:
            sub_data = pl.read_parquet(p)
            sub_data = sub_data.cast(
                {k: v for k, v in cast_features.items() if k in sub_data.columns}
            )
            if k[-1] == '1':
                sub_data = sub_data.drop('num_group1')
            elif k[-1] == '2':
                sub_data = sub_data.drop(['num_group1', 'num_group2'])
            if k[-1] == '1' or k[-1] == '2':
                sub_data = sub_data.group_by('case_id').sum().sort('case_id')
                
            depth_data.append(sub_data)
            
            print(f'\t{sub_data.shape}')
            
            del sub_data
            gc.collect()
        
        depth_data = pl.concat(depth_data, how='vertical_relaxed')
        base_data = base_data.join(depth_data, how='left', on='case_id', suffix=f'_{i}')
        
        del depth_data
        gc.collect()
    
    return base_data.to_pandas()