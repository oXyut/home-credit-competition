from datetime import datetime
from typing import List

import polars as pl
import polars.selectors as cs


def process_date_features(data: pl.DataFrame) -> pl.DataFrame:
    data = data.with_columns(
        pl.col(pl.Date).dt.year().cast(pl.Float32).name.prefix('year_'),
        pl.col(pl.Date).dt.month().cast(pl.Float32).name.prefix('month_'),
        pl.col(pl.Date).dt.day().cast(pl.Float32).name.prefix('day_'),
    )
    data = data.drop(cs.date())
    return data