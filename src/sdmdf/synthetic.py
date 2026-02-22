from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_kpi_data(
    n_groups: int = 4,
    n_periods: int = 180,
    seed: int = 7,
    threshold: float = 100.0,
    distortion_rate: float = 0.25,
) -> pd.DataFrame:
    """Create labeled synthetic KPI data with optional strategic distortions."""
    rng = np.random.default_rng(seed)
    groups = [f"region_{i+1}" for i in range(n_groups)]
    periods = pd.date_range("2024-01-01", periods=n_periods, freq="D")

    rows: list[dict] = []
    for g_idx, group in enumerate(groups):
        phase = g_idx * np.pi / 8
        base = 88 + g_idx * 5
        seasonal = 8 * np.sin(np.linspace(0, 4 * np.pi, n_periods) + phase)
        noise = rng.normal(0, 6, size=n_periods)
        organic = base + seasonal + noise

        deadline_mask = (np.arange(n_periods) % 30) >= 25
        should_distort = rng.random(n_periods) < distortion_rate
        distort_mask = deadline_mask & should_distort

        distorted = organic.copy()
        near_threshold = np.abs(distorted - threshold) < 12
        target_mask = distort_mask & near_threshold
        distorted[target_mask] = threshold + rng.normal(0.3, 0.8, target_mask.sum())

        # variance compression near threshold and deadlines
        compress_mask = distort_mask & (distorted >= threshold - 5) & (distorted <= threshold + 5)
        distorted[compress_mask] = threshold + rng.normal(0.2, 0.3, compress_mask.sum())

        for i, ts in enumerate(periods):
            rows.append(
                {
                    "timestamp": ts,
                    "group": group,
                    "kpi": float(max(distorted[i], 0.0)),
                    "is_distorted": bool(distort_mask[i]),
                    "deadline_window": bool(deadline_mask[i]),
                    "threshold": threshold,
                }
            )

    return pd.DataFrame(rows)
