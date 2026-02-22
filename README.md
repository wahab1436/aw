# Strategic Data Manipulation Detection Framework (SDMDF)

This repository implements a practical starter framework for detecting incentive-aligned KPI distortions and producing a **Strategic Distortion Risk Score (SDRS)** in `[0,1]`.

## What it does

- Generates synthetic KPI time-series with labeled strategic distortion behavior.
- Computes four interpretable detector components:
  - `T`: Threshold clustering score
  - `V`: Variance compression score
  - `A`: Temporal incentive alignment score
  - `G`: Cross-group divergence score
- Aggregates components into `SDRS = w1*T + w2*V + w3*A + w4*G`.
- Returns a per-group score table for reporting and downstream dashboarding.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
python -m sdmdf.cli --groups 5 --periods 240 --distortion-rate 0.3
```

## Repository layout

- `src/sdmdf/synthetic.py`: Organic + manipulated KPI simulation.
- `src/sdmdf/detectors.py`: Statistical detector modules.
- `src/sdmdf/scoring.py`: SDRS weighting and calibration object.
- `src/sdmdf/pipeline.py`: End-to-end scoring pipeline.
- `src/sdmdf/cli.py`: CLI runner for synthetic experiments.
- `tests/test_pipeline.py`: Basic behavioral tests.

## Next steps

- Add calibration routines for detector weights using controlled simulation sweeps.
- Integrate optional Bayesian evidence model to improve interpretability.
- Add notebook and dashboard visualizations for stakeholder communication.
