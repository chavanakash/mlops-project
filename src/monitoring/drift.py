"""Statistical drift detection, implemented from scratch on purpose.

There are libraries (Evidently, etc.) that do this for you, but for a
learning project it's more valuable to see exactly what "drift" means as a
number: are the feature values the live application is seeing still drawn
from the same distribution as what the model was trained on?

Two complementary tests, both standard in industry monitoring:

- KS test (Kolmogorov-Smirnov): a p-value answering "how likely is it that
  these two samples come from the same distribution?" Low p-value (< 0.05)
  = probably not the same distribution anymore.
- PSI (Population Stability Index): buckets both distributions the same
  way and measures how much probability mass shifted between buckets.
  Widely used in credit-risk/fraud modeling because, unlike a p-value, it
  doesn't shrink to "significant" automatically as sample size grows - it
  directly measures *how much* things moved. Rule of thumb: <0.1 no
  meaningful shift, 0.1-0.25 moderate, >0.25 significant.
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def ks_drift(reference: pd.Series, current: pd.Series) -> tuple[float, float]:
    statistic, p_value = ks_2samp(reference.dropna(), current.dropna())
    return float(statistic), float(p_value)


def population_stability_index(
    reference: pd.Series, current: pd.Series, buckets: int = 10
) -> float:
    reference = reference.dropna()
    current = current.dropna()

    # bucket edges come from the REFERENCE distribution's quantiles - PSI
    # measures how current data falls into "the bins training data defined",
    # not some symmetric binning of both.
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, buckets + 1)))
    if len(edges) < 3:
        return 0.0  # reference has ~no spread; PSI is meaningless here

    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.clip(ref_counts / len(reference), 1e-6, None)
    cur_pct = np.clip(cur_counts / len(current), 1e-6, None)

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)
