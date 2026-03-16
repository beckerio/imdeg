# imgdeg/calibration_plot.py

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


def plot_calibration_curve(
    calib: Dict[str, Any],
    ax: Optional[plt.Axes] = None,
    show_points: bool = True,
    show_fit: bool = True,
    show_canonical: bool = True,
    show_secondary_axis: bool = True,
    title: Optional[str] = None,
    metric: Optional[str] = None,
    paper: Optional[str] = None,
    term: Optional[str] = None,

    # ----- COLOR OPTIONS -----
    point_color: Optional[str] = None,
    fit_color: Optional[str] = None,
    canonical_color: Optional[str] = None,
    palette: Optional[Sequence[str]] = None,
) -> plt.Axes:
    """
    Visualize a single calibration result as severity -> distortion-strength curve.

    Expects a dict produced by calibrate_distortion(...), with keys:
        - "paper", "term", "metric"
        - "native_severities"      : list[int]
        - "distortion_strengths"   : list[float]
        - "coeffs"                 : list[float]   (optional, polynomial fit)
        - "fit_values"             : list[float]   (optional)
        - "canonical_strengths"    : list[float]   (optional)

    Features:
        - markers for the measured native severities (s vs D_s),
        - optional polynomial fit curve,
        - optional horizontal lines for canonical strengths,
        - optional secondary y-axis labelling canonical severity indices (1..K),
        - optional color customization via individual colors or palette.
    """
    # ----------------------------------------------
    # Set up axis and figure
    # ----------------------------------------------
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    s = np.array(calib["native_severities"], dtype=float)
    d = np.array(calib["distortion_strengths"], dtype=float)

    if metric is None:
        metric = calib.get("metric", "distortion")
    if paper is None:
        paper = calib.get("paper", "")
    if term is None:
        term = calib.get("term", "")

    # ----------------------------------------------
    # Resolve color palette override (if provided)
    # palette[i]: 0 = points, 1 = fit curve, 2 = canonical
    # ----------------------------------------------
    if palette is not None:
        if len(palette) > 0:
            point_color = palette[0]
        if len(palette) > 1:
            fit_color = palette[1]
        if len(palette) > 2:
            canonical_color = palette[2]

    # ----------------------------------------------
    # Native measured points
    # ----------------------------------------------
    if show_points:
        ax.plot(
            s,
            d,
            "o--",
            label="Measured",
            color=point_color,
        )

    # ----------------------------------------------
    # Polynomial fit (if present)
    # ----------------------------------------------
    coeffs = calib.get("coeffs", None)
    if show_fit and coeffs is not None:
        poly = np.poly1d(coeffs)
        s_fine = np.linspace(s.min(), s.max(), 200)
        d_fine = poly(s_fine)
        ax.plot(
            s_fine,
            d_fine,
            "-",
            label="Polynomial Fit",
            color=fit_color,
        )

    # ----------------------------------------------
    # Canonical strength levels (horizontal lines)
    # ----------------------------------------------
    canon = calib.get("canonical_strengths", None)
    if show_canonical and canon is not None:
        for c in canon:
            ax.axhline(
                y=c,
                linestyle="--",
                linewidth=0.8,
                color=canonical_color,
            )
        # Fake handle for legend
        ax.plot(
            [],
            [],
            "--",
            linewidth=0.8,
            color=canonical_color,
            label="Canonical Levels",
        )

    # ----------------------------------------------
    # Axes formatting: primary axis = metric
    # ----------------------------------------------
    ax.set_xlabel("Native Severity Level ")
    ax.set_ylabel(f"Degradation Strength ({metric})")

    # enforce integer severity ticks (1..5 in most backends)
    s_min = int(np.floor(s.min()))
    s_max = int(np.ceil(s.max()))
    ax.set_xticks(list(range(s_min, s_max + 1)))
    ax.set_xlim(s_min - 0.1, s_max + 0.1)

    if title is None:
        title = f"{term} ({paper})"
    ax.set_title(title)

    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(loc="best", fontsize="small")

    # ----------------------------------------------
    # Secondary y-axis: canonical severity indices (1..K)
    # aligned with metric-space canonical strengths
    # ----------------------------------------------
    if show_secondary_axis and canon is not None and len(canon) > 0:
        ax2 = ax.twinx()

        # sync the metric scale between primary and secondary axes
        ax2.set_ylim(ax.get_ylim())

        # y-positions (metric values) of canonical levels:
        canon_y = np.array(canon, dtype=float)
        canon_levels = np.arange(1, len(canon) + 1, dtype=int)

        ax2.set_yticks(canon_y)
        ax2.set_yticklabels([str(k) for k in canon_levels])
        color_ax2 = canonical_color# "#69b3a2"
        ax2.tick_params(axis="y", labelcolor=color_ax2)
        ax2.set_ylabel("Canonical Severity Level", color=color_ax2)


    fig.tight_layout()
    return ax

def plot_calibration_with_extrapolation_fixed_intervals(
    calib: Dict[str, Any],
    ax: Optional[plt.Axes] = None,
    show_points: bool = True,
    show_fit: bool = True,
    show_canonical: bool = True,
    show_secondary_axis: bool = True,
    show_extrapolated: bool = True,
    K_extra: int = 5,
    fit_degree: int = 2,
    title: Optional[str] = None,
    metric: Optional[str] = None,
    paper: Optional[str] = None,
    term: Optional[str] = None,
    point_color: Optional[str] = None,
    fit_color: Optional[str] = None,
    canonical_color: Optional[str] = None,
    extrapolated_color: Optional[str] = "red",
    palette: Optional[Sequence[str]] = None,
) -> plt.Axes:
    import numpy as np
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    s = np.array(calib["native_severities"], dtype=float)
    d = np.array(calib["distortion_strengths"], dtype=float)
    canon = calib.get("canonical_strengths", None)

    if metric is None:
        metric = calib.get("metric", "distortion")
    if paper is None:
        paper = calib.get("paper", "")
    if term is None:
        term = calib.get("term", "")

    if palette is not None:
        if len(palette) > 0: point_color = palette[0]
        if len(palette) > 1: fit_color = palette[1]
        if len(palette) > 2: canonical_color = palette[2]

    # Determine if distortion increases or decreases with severity
    is_increasing = d[-1] > d[0]

    # Polynomial fit
    coeffs = calib.get("coeffs", None)
    if coeffs is None:
        coeffs = np.polyfit(s, d, deg=fit_degree)
    poly = np.poly1d(coeffs)

    # Plot native points
    if show_points:
        ax.plot(s, d, "o--", label="Measured", color=point_color)

    # Plot fit curve
    if show_fit:
        s_fine = np.linspace(s.min(), s.max(), 200)
        d_fine = poly(s_fine)
        ax.plot(s_fine, d_fine, "-", label="Polynomial Fit", color=fit_color)

    # Canonical lines
    if show_canonical and canon is not None:
        canon_sorted = sorted(canon, reverse=not is_increasing)
        for c in canon_sorted:
            ax.axhline(y=c, linestyle="--", linewidth=0.8, color=canonical_color)
        ax.plot([], [], "--", linewidth=0.8, color=canonical_color, label="Canonical Levels")

    # Extrapolation
    extrap_s, extrap_d, extended_canon = [], [], []
    if show_extrapolated and canon is not None:
        canon_sorted = sorted(canon, reverse=not is_increasing)
        delta = canon_sorted[-1] - canon_sorted[-2]
        for m in range(1, K_extra + 1):
            # Linear extrapolation along distortion axis
            ext_val = canon_sorted[-1] + m * delta if is_increasing else canon_sorted[-1] - m * abs(delta)

            # Invert polynomial to get native severity
            roots = (poly - ext_val).roots
            roots = [r.real for r in roots if np.isreal(r) and r.real > s.max()]
            if roots:
                extrap_s_val = roots[0]
            else:
                # Fallback
                slope = (s[-1] - s[-2]) / (d[-1] - d[-2])
                extrap_s_val = s[-1] + slope * (ext_val - d[-1])

            extrap_s.append(extrap_s_val)
            extrap_d.append(ext_val)
            extended_canon.append(ext_val)

        # Plot extrapolated levels
        ax.plot(extrap_s, extrap_d, "s", color=extrapolated_color,
                label="Extrapolated Canonical", markersize=6)
        for d_ext in extended_canon:
            ax.axhline(y=d_ext, linestyle="--", linewidth=0.8, color=extrapolated_color)

    # Axis ranges
    all_s = np.concatenate([s, extrap_s]) if extrap_s else s
    all_d = np.concatenate([d, extrap_d]) if extrap_d else d
    ax.set_xlim(np.floor(all_s.min()) - 0.5, np.ceil(all_s.max()) + 0.5)
    ax.set_ylim(all_d.min() - 0.1 * np.abs(all_d).max(), all_d.max() + 0.2 * np.abs(all_d).max())

    ax.set_xlabel("Native Severity Level")
    ax.set_ylabel(f"Degradation Strength ({metric})")
    ax.set_title(title or f"{term} ({paper})")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(loc="best", fontsize="small")

    x_ticks = np.arange(0, max(all_s.astype(int)) + 2)
    ax.set_xticks(x_ticks)
    for idx, label in enumerate(ax.get_xticklabels()):
        label.set_color("black" if idx <= 5 else "lightgrey")

    # Secondary y-axis for canonical severity
    if show_secondary_axis and canon is not None:
        ax2 = ax.twinx()
        full_canon = canon_sorted + extended_canon
        canon_labels = [str(i + 1) for i in range(len(full_canon))]

        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(full_canon)
        ax2.set_yticklabels(canon_labels)

        for idx, label in enumerate(ax2.get_yticklabels()):
            label.set_color(canonical_color if idx < len(canon_sorted) else extrapolated_color)

        ax2.set_ylabel("Canonical Severity Level")
        ax2.tick_params(axis="y")

    fig.tight_layout()
    return ax




def plot_multi_calibration_curves(
    calibrations: Sequence[Dict[str, Any]],
    ax: Optional[plt.Axes] = None,
    show_fit: bool = True,
    use_labels_from: str = "term",  # "term" | "paper+term"
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot several calibration curves in a single figure, to compare different
    distortions/backends on the same metric/scale.

    Each element of `calibrations` should be a dict produced by calibrate_distortion.

    Arguments
    ---------
    calibrations:
        A sequence of calibration-result dicts.
    ax:
        Optional matplotlib Axes to draw on. If None, a new figure is created.
    show_fit:
        If True, draw the polynomial fit for each curve (if available);
        otherwise draw only measured points.
    use_labels_from:
        - "term": legend entry is just the term, e.g., "gaublur".
        - "paper+term": legend entry is "<paper>: <term>".
    title:
        Optional custom title; if None, derived from metric name.

    Returns
    -------
    ax:
        The matplotlib Axes instance.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 3.0))
    else:
        fig = ax.figure

    if not calibrations:
        raise ValueError("calibrations must be a non-empty sequence.")

    metric = calibrations[0].get("metric", "distortion")

    for calib in calibrations:
        s = np.array(calib["native_severities"], dtype=float)
        d = np.array(calib["distortion_strengths"], dtype=float)
        paper = calib.get("paper", "")
        term = calib.get("term", "")

        if use_labels_from == "paper+term":
            label = f"{paper}: {term}"
        else:
            label = term or paper

        # Measured points
        ax.plot(
            s,
            d,
            "o",
            markersize=3,
            label=label,
        )

        # Optional polynomial fit
        if show_fit and "coeffs" in calib:
            poly = np.poly1d(calib["coeffs"])
            s_fine = np.linspace(s.min(), s.max(), 200)
            d_fine = poly(s_fine)
            ax.plot(
                s_fine,
                d_fine,
                "-",
            )

    ax.set_xlabel("Native Severity Level $s$")
    ax.set_ylabel(f"Distortion Strength ({metric})")

    # integer ticks on native severity axis
    s_min = int(np.floor(s.min()))
    s_max = int(np.ceil(s.max()))
    ax.set_xticks(list(range(s_min, s_max + 1)))
    ax.set_xlim(s_min - 0.1, s_max + 0.1)

    if title is None:
        title = f"Severity–Distortion Curves ({metric})"
    ax.set_title(title)

    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(loc="best", fontsize="x-small")

    fig.tight_layout()
    return ax


def map_native_to_canonical_indices(calib: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a calibration record (from calibrate_distortion),
    compute the discrete mapping:
        native severity s_i  ->  canonical index k_i in {1..K}
    by nearest canonical_strength in distortion-strength space.
    """
    native_s = np.array(calib["native_severities"], dtype=int)
    native_d = np.array(calib["distortion_strengths"], dtype=float)
    canon_d = np.array(calib["canonical_strengths"], dtype=float)

    canon_indices: List[int] = []
    for d in native_d:
        k = int(np.argmin(np.abs(canon_d - d)))  # 0-based
        canon_indices.append(k + 1)              # → 1..K

    return {
        "native_severities": native_s.tolist(),
        "canonical_indices": canon_indices,
    }


def plot_severity_mapping(
    calib: Dict[str, Any],
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    color: Optional[str] = None,
) -> plt.Axes:
    """
    Plot native severity index s vs. canonical severity index \\hat{s}(s).
    Shows the identity line y=x as a reference.

    Useful to see how a given backend's native severity schedule
    aligns with the canonical five-level axis.
    """
    mapping = map_native_to_canonical_indices(calib)
    s_native = np.array(mapping["native_severities"], dtype=int)
    s_canon = np.array(mapping["canonical_indices"], dtype=int)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Identity line
    s_min = int(min(s_native.min(), s_canon.min()))
    s_max = int(max(s_native.max(), s_canon.max()))
    ax.plot(
        [s_min, s_max],
        [s_min, s_max],
        "--",
        linewidth=0.8,
        color="gray",
        label="Identity $y=x$",
    )

    # Mapping points + connecting line
    ax.plot(
        s_native,
        s_canon,
        "o-",
        label="Native $\\to$ canonical",
        color=color,
    )

    ax.set_xlabel("Native severity index $s$")
    ax.set_ylabel("Canonical severity index $\\hat{s}(s)$")

    ax.set_xticks(range(s_min, s_max + 1))
    ax.set_yticks(range(s_min, s_max + 1))
    ax.set_xlim(s_min - 0.1, s_max + 0.1)
    ax.set_ylim(s_min - 0.1, s_max + 0.1)

    if title is None:
        paper = calib.get("paper", "")
        term = calib.get("term", "")
        title = f"{paper}: {term}"
    ax.set_title(title)

    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
    ax.legend(loc="best", fontsize="small")

    fig.tight_layout()
    return ax
