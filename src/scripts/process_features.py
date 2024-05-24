from datetime import datetime
from typing import List

import polars as pl
import polars.selectors as cs


def process_date_features(data: pl.DataFrame) -> pl.DataFrame:
    data = data.with_columns(
        pl.col(pl.Date).dt.year().cast(pl.Float32).name.prefix('year_'),
        pl.col(pl.Date).dt.month().cast(pl.Float32).name.prefix('month_'),
        pl.col(pl.Date).dt.day().cast(pl.Float32).name.prefix('day_'),
        pl.col(pl.Date).dt.weekday().cast(pl.Float32).name.prefix('weekday_'),
    )
    # data = data.with_columns(
    #     [
    #         (pl.col('date_decision') - pl.col(feature)).dt.total_days().cast(pl.Float32).alias(f'total_days_diff_date_decision_{feature}')
    #         for feature in data.select(pl.col(pl.Date)).columns if feature != 'date_decision'
    #     ]
    # )
    data = data.drop(cs.date())
    return data