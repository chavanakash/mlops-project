"""Simulates the data engineer's job: load historical source data into the
warehouse (Postgres) once, as a backfill.

Dataset: California housing (bundled with scikit-learn, 20,640 rows, real
1990 census data - median income, rooms, population, lat/long -> median
house value). We stamp every row with source='historical_batch' and spread
ingested_at over the last 180 days so the warehouse looks like it was
actually populated over time, not all at once - this matters later when we
simulate "new data arriving day to day" and query "what's new since last
training run".

Run: python -m src.data.load_historical_data
"""
import datetime as dt

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sqlalchemy import create_engine

from src.config import DATABASE_URL

COLUMN_MAP = {
    "MedInc": "med_inc",
    "HouseAge": "house_age",
    "AveRooms": "ave_rooms",
    "AveBedrms": "ave_bedrms",
    "Population": "population",
    "AveOccup": "ave_occup",
    "Latitude": "latitude",
    "Longitude": "longitude",
}


def load() -> pd.DataFrame:
    bunch = fetch_california_housing(as_frame=True)
    df = bunch.frame.rename(columns=COLUMN_MAP)
    df = df.rename(columns={"MedHouseVal": "med_house_val"})

    rng = np.random.default_rng(seed=42)
    days_ago = rng.integers(low=1, high=180, size=len(df))
    now = dt.datetime.utcnow()
    df["ingested_at"] = [now - dt.timedelta(days=int(d)) for d in days_ago]
    df["source"] = "historical_batch"
    return df


def main() -> None:
    df = load()
    engine = create_engine(DATABASE_URL)
    df.to_sql("houses", engine, if_exists="append", index=False, chunksize=2000)
    print(f"Loaded {len(df)} historical rows into 'houses' (source=historical_batch)")


if __name__ == "__main__":
    main()
