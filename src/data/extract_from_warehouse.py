"""DVC pipeline stage 1: EXTRACT.

Pulls a snapshot of the warehouse's `houses` table into a CSV under
data/raw/. Why snapshot to a file instead of hitting Postgres directly from
every later stage: DVC versions *files*, not live database state. By
extracting once and letting DVC hash the resulting CSV, every later stage
(featurize, train) becomes reproducible from that exact snapshot - `dvc
repro` can tell you "this model was trained on this exact data", which you
cannot say if training reads a mutable table directly.

`extract.as_of` in params.yaml lets a retraining run scope the pull to
"everything ingested up to this timestamp" - unset (null) means "all
historical data", which is what the first pipeline run uses.

Run: python -m src.data.extract_from_warehouse
"""
import pandas as pd
import yaml
from sqlalchemy import create_engine, text

from src.config import DATABASE_URL

RAW_PATH = "data/raw/houses.csv"


def main() -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    as_of = params["extract"].get("as_of")

    engine = create_engine(DATABASE_URL)
    if as_of:
        query = text("SELECT * FROM houses WHERE ingested_at <= :as_of")
        df = pd.read_sql(query, engine, params={"as_of": as_of})
    else:
        df = pd.read_sql(text("SELECT * FROM houses"), engine)
    df.to_csv(RAW_PATH, index=False)
    print(f"Extracted {len(df)} rows -> {RAW_PATH} (as_of={as_of or 'ALL'})")


if __name__ == "__main__":
    main()
