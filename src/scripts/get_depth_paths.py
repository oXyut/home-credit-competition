from collections import OrderedDict
import pathlib


def get_depth_paths(load_dir: pathlib.Path, prefix: str):
    
    assert prefix in ['test', 'train']
    
    depth_paths = OrderedDict()

    depth_paths['static_0'] = []
    depth_paths['static_cb_0'] = []
    depth_paths['applprev_1'] = []
    depth_paths['other_1'] = []
    depth_paths['tax_registry_a_1'] = []
    depth_paths['tax_registry_b_1'] = []
    depth_paths['tax_registry_c_1'] = []
    depth_paths['credit_bureau_a_1'] = []
    depth_paths['credit_bureau_b_1'] = []
    depth_paths['deposit_1'] = []
    depth_paths['person_1'] = []
    depth_paths['debitcard_1'] = []
    depth_paths['applprev_2'] = []
    depth_paths['person_2'] = []
    depth_paths['credit_bureau_a_2'] = []
    depth_paths['credit_bureau_b_2'] = []

    for k in depth_paths.keys():
        depth_paths[k] = sorted(
            [p for p in load_dir.joinpath(f'{prefix}').glob(f'{prefix}_{k}*.parquet')]
        )
    return depth_paths