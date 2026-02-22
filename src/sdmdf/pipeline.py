from __future__ import annotations

import pandas as pd

from .detectors import (
    cross_group_divergence_score,
    temporal_incentive_alignment_score,
    threshold_clustering_score,
    variance_compression_score,
)
from .scoring import SDRSConfig, compute_sdrs


def run_analysis(df: pd.DataFrame, config: SDRSConfig | None = None) -> pd.DataFrame:
    """Compute component scores and SDRS per group."""
    g_score = cross_group_divergence_score(df)
    results = []

    for group, gdf in df.groupby("group"):
        values = gdf["kpi"].to_numpy()
        threshold = float(gdf["threshold"].iloc[0])
        t = threshold_clustering_score(values, threshold)
        v = variance_compression_score(values, threshold)
        a = temporal_incentive_alignment_score(gdf)
        sdrs = compute_sdrs(t=t, v=v, a=a, g=g_score, config=config)
        results.append(
            {
                "group": group,
                "T_threshold_cluster": t,
                "V_variance_compression": v,
                "A_temporal_alignment": a,
                "G_cross_group_divergence": g_score,
                "SDRS": sdrs,
            }
        )

    return pd.DataFrame(results).sort_values("SDRS", ascending=False).reset_index(drop=True)
