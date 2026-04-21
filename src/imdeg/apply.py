# src/imdeg/apply.py
from typing import Any, Dict, Optional, Tuple, List
import torch

from .taxonomy import map_paper_term
from .backends.arniqa import distortion_range
from .registry import resolve_backend, BACKEND_CAPS
from .severity import native_for_canonical_level
from .severity import invert_poly_to_native


def _apply_at_native_severity(
    image: torch.Tensor,
    backend_fn: Any,
    paper: str,
    term: str,
    native_severity: int,
) -> torch.Tensor:
    """Apply a backend using its native severity semantics."""
    caps = BACKEND_CAPS.get((paper, term), None)

    if caps is None:
        raise KeyError(f"No BACKEND_CAPS entry for {(paper, term)}")

    s = max(1, min(5, int(round(native_severity))))

    if caps["mode"] == "param":
        if term not in distortion_range:
            raise KeyError(
                f"term '{term}' requires parameter mode but no distortion_range entry found."
            )
        param = distortion_range[term][s - 1]
        return backend_fn(image, param)

    if caps["mode"] == "severity":
        return backend_fn(image, severity=s)

    raise ValueError(f"Unsupported backend mode {caps['mode']!r} for {(paper, term)}")

def apply_degradation(
    image: torch.Tensor,
    paper: str,
    term: str,
    severity: float = 3.0,
    mode: str = "original",  # "original" or "canonical"
    calibration: Optional[Dict[str, Any]] = None,
    canonical_strengths: Optional[list[float]] = None,
) -> torch.Tensor:
    """
    Apply a degradation defined in the unified taxonomy.

    Args
    ----
    image:
        (C,H,W) tensor in [0,1].
    paper:
        Key of the paper mapping, e.g. "Agnolucci_WACV_2024" or "Hendrycks_ICLR_2018".
    term:
        Original corruption/distortion name in that paper, e.g. "gaublur", "gaussian_noise".
    severity:
        - if mode == "original": native severity index (1..5)
        - if mode == "canonical": canonical severity index (1..5)
    mode:
        "original" - use the paper's native severity schedule.
        "canonical" - map canonical severity to canonical strength to native severity.
    calibration:
        Dict returned by `calibrate_distortion(...)`.
    canonical_strengths:
        List [D1..D5] defining canonical distortion strengths for this distortion.
        If None and calibration contains "canonical_strengths", that will be used.
    """

    image_clone = image.clone()
    backend_fn = resolve_backend(paper, term)

    if mode == "original":
        return _apply_at_native_severity(
            image=image_clone,
            backend_fn=backend_fn,
            paper=paper,
            term=term,
            native_severity=int(round(severity)),
        )

    if mode == "canonical":
        if calibration is None:
            raise ValueError("canonical mode requires 'calibration' dict.")

        if canonical_strengths is None:
            canonical_strengths = calibration.get("canonical_strengths", None)
        if canonical_strengths is None:
            raise ValueError(
                "canonical mode requires 'canonical_strengths', "
                "either passed explicitly or stored in calibration['canonical_strengths']."
            )

        k = max(1, min(5, int(round(severity))))
        target_strength = canonical_strengths[k - 1]
        native_s = native_for_canonical_level(calibration, target_strength)
        return _apply_at_native_severity(
            image=image_clone,
            backend_fn=backend_fn,
            paper=paper,
            term=term,
            native_severity=int(round(native_s)),
        )

    raise ValueError(f"Unknown mode {mode!r}")




def degrade_image_with_annotation(
    image: torch.Tensor,
    paper: str,
    term: str,
    severity: int = 1,
    mode: str = "original",
    dataset_tags: Optional[list] = None,
    extra_params: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Apply degradation and return image plus a rich annotation dict
    consistent with the taxonomy schema.
    """

    image_clone = image.clone()       # Prevents changes to original image

    degraded = apply_degradation(
        image=image_clone,
        paper=paper,
        term=term,
        severity=severity,
        mode=mode,
    )

    info = map_paper_term(paper, term)
    g = info["group"]

    annot: Dict[str, Any] = {
        "group_id": g["id"],
        "source": info.get("source", g["source"]),
        "effect": info.get("effect", g["effect"]),
        "subtype": None,
        "name": term,
        "co_groups": [],
        "severity": float(severity),
        "params": {
            "paper": paper,
            "term": term,
            "mode": mode,
            "backend_param": distortion_range.get(term, [None])[severity - 1]
            if term in distortion_range else None,
        },
        "sequence_len": 1,
        "dataset_tags": dataset_tags or [],
        "notes": "",
    }

    if extra_params:
        annot["params"].update(extra_params)

    return degraded, annot


import numpy as np
import torch
from typing import Dict, Any, Optional

def apply_degradation_extrapolated(
    image: torch.Tensor,
    paper: str,
    term: str,
    severity: float = 6.0,
    calibration: Optional[Dict[str, Any]] = None,
    poly_degree: int = 2,
    search_extend: float = 2.0,  # Extend range for extrapolation
    max_native_allowed: int = 10  # Optional safety cap
) -> torch.Tensor:
    """
    Apply degradation using extrapolated canonical severity (e.g., s = 6, 7, ...),
    based on inversion of polynomial fit from degradation strength to native severity.

    Args:
        image: Input image (C,H,W) in [0,1].
        paper: Backend key, e.g., 'Hendrycks_ICLR_2019'.
        term: Degradation name, e.g., 'gaussian_blur'.
        severity: Canonical severity index (can exceed 5).
        calibration: Dict with canonical/native strengths and polynomial coeffs.
        poly_degree: Degree of the polynomial fit used for inversion.
        search_extend: Multiplier for extrapolation range search (e.g., 2x native max).
        max_native_allowed: Clamp native severity to this value to prevent invalid backend calls.

    Returns:
        Degraded image as tensor.
    """

    image_clone = image.clone()       # Prevents changes to original image

    if calibration is None:
        raise ValueError("Calibration dict is required.")
    if "native_severities" not in calibration or "distortion_strengths" not in calibration:
        raise ValueError("Calibration must contain 'native_severities' and 'distortion_strengths'.")

    # Fit polynomial if not already present
    if "coeffs" not in calibration:
        native = np.array(calibration["native_severities"], dtype=np.float32)
        distort = np.array(calibration["distortion_strengths"], dtype=np.float32)
        coeffs = np.polyfit(native, distort, deg=poly_degree)
        calibration["coeffs"] = coeffs.tolist()
    else:
        coeffs = np.array(calibration["coeffs"], dtype=np.float32)

    # Get canonical degradation strengths
    canon = calibration.get("canonical_strengths", None)
    if canon is None:
        raise ValueError("Calibration must contain 'canonical_strengths'.")

    k = int(round(severity))
    K = len(canon)

    # Step 1: Determine target canonical degradation strength
    if k <= K:
        target_strength = canon[k - 1]
    else:
        delta = canon[-1] - canon[-2]
        target_strength = canon[-1] + (k - K) * delta

    # Step 2: Invert polynomial to get native severity
    s_min = float(min(calibration["native_severities"]))
    s_max = float(max(calibration["native_severities"])) * search_extend
    grid = np.linspace(s_min, s_max, 1000)

    poly = np.poly1d(coeffs)
    d_vals = poly(grid)
    idx = np.argmin(np.abs(d_vals - target_strength))
    native_severity = float(grid[idx])

    # Safety clamp
    native_severity = min(native_severity, max_native_allowed)

    # Step 3: Apply degradation using native severity (cascaded if needed)
    from .apply import apply_degradation  # Adjust to match your actual import
    backend_fn = apply_degradation  # Should resolve the correct backend internally

    print(native_severity)
    if native_severity <= 5:
        return backend_fn(image_clone, paper=paper, term=term, severity=int(round(native_severity)))
    else:
        # Split into two applications: severity 5 + residual
        int_part = int(np.floor(native_severity))
        frac_part = native_severity - int_part
        img_mid = backend_fn(image_clone, paper=paper, term=term, severity=5)
        final_severity = round(frac_part * 5)  # scale fraction to [0,5] scale
        final_severity = max(1, min(5, final_severity))  # keep valid
        return backend_fn(img_mid, paper=paper, term=term, severity=final_severity)





