"""
Helpers to export calibration results to a unified severity.json file.

We assume you already ran `calibrate_distortion(...)` for various
(paper, term) pairs and collected the results in a dictionary.
"""

from __future__ import annotations

import json
from typing import Dict, Tuple, Any, List


def export_severity_json(
    calibrations: Dict[Tuple[str, str], Dict[str, Any]],
    out_path: str,
    metric: str = "1-ssim",
    canonical_levels: List[int] | None = None,
) -> None:
    """
    Export a dictionary of calibration results to severity.json.

    Args
    ----
    calibrations:
        Mapping from (paper, term) -> calibration dict, where each dict
        is the result of `calibrate_distortion(...)`, e.g.:

        {
            ("Agnolucci_WACV_2024", "gaublur"): {
                "paper": "Agnolucci_WACV_2024",
                "term": "gaublur",
                "metric": "1-ssim",
                "native_severities": [1,2,3,4,5],
                "distortion_strengths": [...],
                "coeffs": [...],
                "fit_values": [...],
                # optional:
                # "canonical_strengths": [...]
            },
            ...
        }

    out_path:
        Path to write the JSON file (e.g. severity.json).

    metric:
        Name of the metric used for calibration (for metadata only).

    canonical_levels:
        Canonical severity indices; default [1,2,3,4,5].
    """
    if canonical_levels is None:
        canonical_levels = [1, 2, 3, 4, 5]

    root = {
        "metric": metric,
        "canonical_levels": canonical_levels,
        "papers": {},  # paper -> term -> calib
    }

    for (paper, term), info in calibrations.items():
        if paper not in root["papers"]:
            root["papers"][paper] = {}

        # Pick only the relevant keys to make the JSON clean
        entry = {
            "native_severities": info.get("native_severities", []),
            "distortion_strengths": info.get("distortion_strengths", []),
            "coeffs": info.get("coeffs", []),
            "fit_values": info.get("fit_values", []),
        }
        if "canonical_strengths" in info:
            entry["canonical_strengths"] = info["canonical_strengths"]

        root["papers"][paper][term] = entry

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(root, f, indent=2)

    print(f"[imgdeg] severity.json written to: {out_path}")
