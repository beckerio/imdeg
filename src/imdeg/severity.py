# src/imgdeg/severity.py

from typing import List, Union
import numpy as np


def invert_poly_to_native(
    coeffs: List[float],
    target_strength: float,
    s_min: float = 1.0,
    s_max: float = 5.0,
    n_steps: int = 1000,
) -> float:
    """
    Numerically invert p(s) ~ target_strength on [s_min, s_max]
    by simple grid search (good enough for degree 2/3).
    """
    poly = np.poly1d(np.array(coeffs, dtype=np.float32))
    grid = np.linspace(s_min, s_max, n_steps)
    vals = poly(grid)
    idx = int(np.argmin(np.abs(vals - target_strength)))
    return float(grid[idx])


def native_for_canonical_level_poly(
    calib: dict,
    canonical_strength: float,
) -> float:
    """
    Given a calibration dict (with 'coeffs' and 'native_severities')
    and a target canonical strength, return native severity s* ∈ [min,max].
    """
    return invert_poly_to_native(
        coeffs=calib["coeffs"],
        target_strength=canonical_strength,
        s_min=min(calib["native_severities"]),
        s_max=max(calib["native_severities"]),
    )


def native_for_canonical_level(
    calib: dict,
    canonical_strength: float,
) -> float:
    """
    Map a canonical distortion strength back to a native severity value s*,
    using linear interpolation over measured distortion strengths.

    Unlike older versions, this no longer requires polynomial coefficients.
    Works even when calib["coeffs"] is absent.
    """
    native_s = np.array(calib["native_severities"], dtype=float)
    native_d = np.array(calib["distortion_strengths"], dtype=float)

    # sort by distortion (monotonic) to ensure consistent interpolation
    idx = np.argsort(native_d)
    native_d_sorted = native_d[idx]
    native_s_sorted = native_s[idx]

    # clamp canonical_strength to observed distortion range
    D = float(canonical_strength)
    if D <= native_d_sorted[0]:
        return float(native_s_sorted[0])
    if D >= native_d_sorted[-1]:
        return float(native_s_sorted[-1])

    # linear interpolation native_severity ← distortion_strength
    return float(np.interp(D, native_d_sorted, native_s_sorted))


Number = Union[int, float]
def map_severity_to_range(
    severity: int,
    lo: Number,
    hi: Number,
) -> float:
    """
    Map a canonical severity level s ∈ {1,...,5} to a scalar in [lo, hi].
    This is a *placeholder* mapping; calibration.py can later overwrite
    this behavior using severity.json or more complex curves.

    Args:
        severity: integer in [1..5]
        lo: lower bound of parameter range
        hi: upper bound of parameter range

    Returns:
        Scalar in [lo, hi] interpolated linearly.
    """
    if not (1 <= severity <= 5):
        raise ValueError(f"severity must be in [1..5], got {severity}")
    if lo == hi:
        return float(lo)

    alpha = (severity - 1) / 4.0  # 0 for s=1, 1 for s=5
    return float(lo + alpha * (hi - lo))
