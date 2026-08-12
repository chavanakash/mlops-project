import pandas as pd

from src.features.build_features import add_features


def test_bedrms_ratio_computed_correctly():
    df = pd.DataFrame({"ave_rooms": [4.0, 5.0], "ave_bedrms": [2.0, 1.0]})
    result = add_features(df)
    assert list(result["bedrms_ratio"]) == [0.5, 0.2]
