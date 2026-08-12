"""DVC pipeline stage 2: FEATURIZE.

Turns the raw warehouse snapshot into a model-ready train/test split.

Feature engineering choice: `bedrms_ratio` (bedrooms per room) is added
because it's a real predictive signal for house value that isn't directly
present as a column - a lower ratio (more living rooms per bedroom) tends to
correlate with higher-value homes. This is the "data scientist" judgment
call step: raw warehouse columns rarely go straight into a model unchanged.

Why split here (not inside train.py): so the *same* train/test split is a
versioned DVC artifact. If train.py did its own random split, two different
model-training runs could accidentally train/evaluate on different splits,
making metric comparisons meaningless.

Run: python -m src.features.build_features
"""
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/houses.csv"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

TARGET = "med_house_val"


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["bedrms_ratio"] = df["ave_bedrms"] / df["ave_rooms"]
    return df


def main() -> None:
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    split_cfg = params["split"]

    df = pd.read_csv(RAW_PATH)
    df = add_features(df)

    feature_cols = [
        "med_inc", "house_age", "ave_rooms", "ave_bedrms",
        "population", "ave_occup", "latitude", "longitude", "bedrms_ratio",
    ]
    df = df[feature_cols + [TARGET]]

    train_df, test_df = train_test_split(
        df,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
    )

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f"train={len(train_df)} rows -> {TRAIN_PATH}")
    print(f"test={len(test_df)} rows -> {TEST_PATH}")


if __name__ == "__main__":
    main()
