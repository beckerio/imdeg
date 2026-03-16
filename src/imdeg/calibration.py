# src/imgdeg/calibration.py

from typing import List, Dict, Any
import numpy as np
import torch

from .apply import apply_degradation
from .metrics import distortion_metric


@torch.no_grad()
def measure_severity_curve(
    images: List[torch.Tensor],
    paper: str,
    term: str,
    metric: str = "1-ssim",  # or "psnr_drop"
    n_severities: int = 5,
) -> Dict[str, Any]:
    """
    Measure average distortion strength D_j(s) for s = 1..n_severities
    using the *native* severity definition of the paper/backend.

    Returns a dict with:
        - "paper", "term", "metric"
        - "native_severities": [1, ..., n_severities]
        - "distortion_strengths": [D_j(1), ..., D_j(n)]
    """
    severities = list(range(1, n_severities + 1))
    strengths: List[float] = []

    for s in severities:
        vals: List[float] = []
        for x in images:
            x = x.to(torch.float32)
            y = apply_degradation(
                image=x,
                paper=paper,
                term=term,
                severity=s,
                mode="original",   # IMPORTANT: native scale of the backend
            )
            d = distortion_metric(x, y, kind=metric)
            if not np.isinf(d) and not np.isnan(d):
                vals.append(d)

        if vals:
            strengths.append(float(np.mean(vals)))
        else:
            print(f"⚠️  All values invalid for severity {s}, term '{term}'")
            strengths.append(float("nan"))


    return {
        "paper": paper,
        "term": term,
        "metric": metric,
        "native_severities": severities,
        "distortion_strengths": strengths,
    }


# =====================================================================
#  Polynomial fitting (EXPERIMENTAL; not used in main harmonization)
# =====================================================================

def fit_polynomial(
    native_severities: List[int],
    strengths: List[float],
    degree: int = 2,
    force_endpoints: bool = False,
) -> Dict[str, Any]:
    """
    EXPERIMENTAL: Fit a polynomial p(s) mapping native severity index -> distortion strength.

    - This is *not* required for the main canonical severity axis.
    - It can be used for analysis/visualization only.

    If force_endpoints=True and degree==2, we fit a constrained quadratic
    that exactly passes through the first and last points:

        p(s) = line_through_endpoints(s) + b * (s - s1) * (s - sn)

    Otherwise we fall back to standard np.polyfit.

    Returns dict with:
        - "coeffs": [a_deg, ..., a0]  (np.poly1d convention)
        - "fit_values": p(x) at the input x
    """
    x = np.array(native_severities, dtype=np.float32)
    y = np.array(strengths, dtype=np.float32)

    if len(x) < 2:
        raise ValueError("Need at least two severity levels to fit a polynomial.")

    # ---------------------------------------------------------
    # Constrained quadratic: pass through first & last points
    # ---------------------------------------------------------
    if force_endpoints and degree == 2 and len(x) >= 3:
        # Endpoints
        s1 = float(x[0])
        sn = float(x[-1])
        d1 = float(y[0])
        dn = float(y[-1])

        # Line through endpoints: l(s) = d1 + m*(s - s1)
        m = (dn - d1) / (sn - s1)

        # Interior points only
        x_int = x[1:-1]
        y_int = y[1:-1]

        # Basis term that vanishes at s1 and sn
        # p(s) = l(s) + b * (s - s1)(s - sn)
        X = (x_int - s1) * (x_int - sn)               # shape (n_int,)
        y_res = y_int - (d1 + m * (x_int - s1))       # residual vs. straight line

        if len(X) > 0 and np.any(X != 0):
            # Least-squares solution for scalar b
            b = float((X @ y_res) / (X @ X))
        else:
            b = 0.0

        # Expand to p(s) = a2 s^2 + a1 s + a0
        # p(s) = d1 + m*(s - s1) + b*(s - s1)*(s - sn)
        #      = b s^2 + (m - b*(s1 + sn)) s + (d1 - m*s1 + b*s1*sn)
        a2 = b
        a1 = m - b * (s1 + sn)
        a0 = d1 - m * s1 + b * s1 * sn

        coeffs = [a2, a1, a0]
        poly = np.poly1d(coeffs)
        y_fit = poly(x)

        return {
            "coeffs": coeffs,
            "fit_values": y_fit.tolist(),
        }

    # ---------------------------------------------------------
    # Fallback: standard unconstrained polynomial fit
    # ---------------------------------------------------------
    coeffs = np.polyfit(x, y, deg=degree)
    poly = np.poly1d(coeffs)
    y_fit = poly(x)

    return {
        "coeffs": coeffs.tolist(),
        "fit_values": y_fit.tolist(),
    }


def invert_polynomial(coeffs, target_D, valid_range=(1.0, 5.0)) -> float | None:
    """
    EXPERIMENTAL helper:
    Given polynomial coefficients and a target distortion D, find a real root s
    in a given severity range, such that p(s) = D.

    Used only for analysis / what-if experiments, not in main harmonization.
    """
    poly = np.poly1d(coeffs)
    roots = np.roots(poly - target_D)
    lo, hi = valid_range
    roots = [
        r.real for r in roots
        if np.isreal(r) and lo <= r.real <= hi
    ]
    if len(roots) == 0:
        return None
    return roots[0]


# =====================================================================
#  Canonical distortion strengths (main method)
# =====================================================================

def derive_canonical_strengths(
    strengths: List[float],
    n_levels: int = 5,
) -> List[float]:
    """
    Main method: create canonical distortion-strength levels for this distortion.

    We simply place n_levels points linearly between the observed min and max
    distortion strengths. This defines a metric-based canonical axis:

        D_min ---- D_1 ---- ... ---- D_n ---- D_max

    where all steps represent equal increments in distortion metric space.
    """
    if len(strengths) == 0:
        raise ValueError("strengths must be non-empty")

    s_min = float(min(strengths))
    s_max = float(max(strengths))

    if s_min == s_max:
        # degenerate case: all strengths equal
        return [s_min] * n_levels

    canon = np.linspace(s_min, s_max, n_levels)
    return [float(v) for v in canon]


def calibrate_distortion(
    images: List[torch.Tensor],
    paper: str,
    term: str,
    metric: str = "1-ssim",
    n_canonical_levels: int = 5,

    # EXPERIMENTAL options (polynomial fit, optional)
    compute_poly: bool = False,
    poly_degree: int = 2,
    poly_force_endpoints: bool = False,
) -> Dict[str, Any]:
    """
    Main calibration entry point.

    1) Measures native distortion strengths D_j(s) for the given backend.
    2) Derives a metric-based canonical severity axis with n_canonical_levels.
    3) (Optional) Adds experimental polynomial fit info for analysis.

    Returns e.g.:

      {
        "paper": "Agnolucci_WACV_2024",
        "term": "gaublur",
        "metric": "1-ssim",
        "native_severities": [1, 2, 3, 4, 5],
        "distortion_strengths": [...],

        # always present:
        "canonical_strengths": [D1, D2, D3, D4, D5],

        # only if compute_poly=True:
        "coeffs": [a2, a1, a0],
        "fit_values": [...],
      }
    """
    # 1) Measure native curve
    curve = measure_severity_curve(
        images=images,
        paper=paper,
        term=term,
        metric=metric,
        n_severities=5,
    )

    result: Dict[str, Any] = {**curve}

    # 2) Derive canonical strengths (main method)
    canonical_strengths = derive_canonical_strengths(
        strengths=curve["distortion_strengths"],
        n_levels=n_canonical_levels,
    )
    result["canonical_strengths"] = canonical_strengths

    # 3) Optional: polynomial fit for experiments
    if compute_poly:
        poly_info = fit_polynomial(
            native_severities=curve["native_severities"],
            strengths=curve["distortion_strengths"],
            degree=poly_degree,
            force_endpoints=poly_force_endpoints,
        )
        result.update(poly_info)

    return result
