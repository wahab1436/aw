from __future__ import annotations

import argparse

from .pipeline import run_analysis
from .synthetic import generate_synthetic_kpi_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SDMDF synthetic analysis")
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--periods", type=int, default=180)
    parser.add_argument("--distortion-rate", type=float, default=0.25)
    args = parser.parse_args()

    df = generate_synthetic_kpi_data(
        n_groups=args.groups,
        n_periods=args.periods,
        distortion_rate=args.distortion_rate,
    )
    scores = run_analysis(df)
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
