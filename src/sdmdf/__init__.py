"""Strategic Data Manipulation Detection Framework (SDMDF)."""

from .pipeline import run_analysis
from .scoring import SDRSConfig, compute_sdrs

__all__ = ["run_analysis", "SDRSConfig", "compute_sdrs"]
