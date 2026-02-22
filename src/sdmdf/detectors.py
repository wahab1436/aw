from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy, gaussian_kde, wasserstein_distance


def _sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x)))


def threshold_clustering_score(values: np.ndarray, threshold: float, bandwidth: float = 1.5) -> float:
    """Estimate bunching near incentive threshold using local KDE ratio."""
    if len(values) < 20:
        return 0.0
    std = max(float(np.std(values, ddof=1)), 1e-6)
    kde = gaussian_kde(values, bw_method=bandwidth / std)
    center = kde.evaluate([threshold])[0]
    flank = np.mean(kde.evaluate([threshold - 8, threshold + 8]))
    ratio = (center - flank) / max(flank, 1e-6)
    return max(0.0, min(1.0, _sigmoid(ratio * 2.5) - 0.5)) * 2


def variance_compression_score(values: np.ndarray, threshold: float, window: float = 6.0) -> float:
    near = values[(values >= threshold - window) & (values <= threshold + window)]
    far = values[(values < threshold - window) | (values > threshold + window)]
    if len(near) < 10 or len(far) < 10:
        return 0.0
    var_ratio = np.var(near) / max(np.var(far), 1e-6)
    hist_near, _ = np.histogram(near, bins=12, density=True)
    hist_far, _ = np.histogram(far, bins=12, density=True)
    ent_ratio = entropy(hist_near + 1e-6) / max(entropy(hist_far + 1e-6), 1e-6)
    compression = 1 - 0.5 * (var_ratio + ent_ratio)
    return float(np.clip(compression, 0, 1))


def temporal_incentive_alignment_score(df_group: pd.DataFrame) -> float:
    """Measure whether threshold-adjacent outcomes increase in deadline windows."""
    threshold = df_group["threshold"].iloc[0]
    near = (df_group["kpi"].between(threshold - 2, threshold + 2)).astype(float)
    deadline = df_group["deadline_window"].astype(float)
    if near.mean() == 0 or deadline.mean() == 0:
        return 0.0
    lift = near[deadline == 1].mean() - near[deadline == 0].mean()
    return float(np.clip(lift * 3.0, 0, 1))


def cross_group_divergence_score(df: pd.DataFrame) -> float:
    """Average Wasserstein divergence between group KPI distributions."""
    groups = list(df["group"].unique())
    if len(groups) < 2:
        return 0.0
    dists = []
    for i, g1 in enumerate(groups):
        for g2 in groups[i + 1 :]:
            v1 = df.loc[df["group"] == g1, "kpi"].to_numpy()
            v2 = df.loc[df["group"] == g2, "kpi"].to_numpy()
            dists.append(wasserstein_distance(v1, v2))
    avg_dist = float(np.mean(dists))
    # Normalize to [0,1] with soft cap at 10 KPI points
    return float(np.clip(avg_dist / 10.0, 0, 1))
