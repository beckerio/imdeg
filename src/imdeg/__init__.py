"""
imgdeg – Imaging degradation library

This package provides:
- a taxonomy of imaging degradations (taxonomy.py)
- paper-specific mappings (Liu, Hendrycks, ARNIQA, …)
- backend registries to actual distortion functions (registry.py, backends/)
- convenience functions to apply degradations (apply.py)
"""

from .taxonomy import taxonomy, map_paper_term, list_paper_types
from .registry import resolve_backend
from .apply import apply_degradation, degrade_image_with_annotation
#from .calibration import invert_poly_to_native, native_for_canonical_level

__all__ = [
    "taxonomy",
    "map_paper_term",
    "list_paper_types",
    "resolve_backend",
    "apply_degradation",
    "degrade_image_with_annotation",
    #"invert_poly_to_native",
    #"native_for_canonical_level"
]
