import os
from pathlib import Path

from comet_ml import Experiment
from dotenv import load_dotenv
import yaml

DIR_INPUTS = Path('../../data/inputs')
dir_path = Path.joinpath(DIR_INPUTS, 'csv_files')
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


def run():
    load_dotenv()
    experiment = Experiment()

    with open(DIR_PATH_CONFIGS, 'r') as f:
        dict_tables_all = yaml.safe_load(f)


    experiment.log_metric("accuracy", 0.9)

    experiment.end()

if __name__ == "__main__":
    run()