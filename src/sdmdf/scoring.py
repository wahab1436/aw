from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SDRSConfig:
    w_t: float = 0.3
    w_v: float = 0.25
    w_a: float = 0.25
    w_g: float = 0.2


def compute_sdrs(t: float, v: float, a: float, g: float, config: SDRSConfig | None = None) -> float:
    cfg = config or SDRSConfig()
    raw = cfg.w_t * t + cfg.w_v * v + cfg.w_a * a + cfg.w_g * g
    return max(0.0, min(1.0, raw))
