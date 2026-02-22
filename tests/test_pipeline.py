from sdmdf.pipeline import run_analysis
from sdmdf.synthetic import generate_synthetic_kpi_data


def test_pipeline_returns_expected_columns():
    df = generate_synthetic_kpi_data(n_groups=3, n_periods=120, seed=10, distortion_rate=0.3)
    out = run_analysis(df)
    expected = {
        "group",
        "T_threshold_cluster",
        "V_variance_compression",
        "A_temporal_alignment",
        "G_cross_group_divergence",
        "SDRS",
    }
    assert expected.issubset(set(out.columns))
    assert len(out) == 3


def test_scores_are_bounded():
    df = generate_synthetic_kpi_data(n_groups=2, n_periods=90, seed=11, distortion_rate=0.4)
    out = run_analysis(df)
    for col in [
        "T_threshold_cluster",
        "V_variance_compression",
        "A_temporal_alignment",
        "G_cross_group_divergence",
        "SDRS",
    ]:
        assert out[col].between(0, 1).all()
