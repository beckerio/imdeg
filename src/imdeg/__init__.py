"""Public package interface for :mod:`imdeg`."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .taxonomy import list_paper_types, map_paper_term, taxonomy

try:
    __version__ = version("imdeg")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "__version__",
    "taxonomy",
    "map_paper_term",
    "list_paper_types",
    "resolve_backend",
    "apply_degradation",
    "degrade_image_with_annotation",
]


def resolve_backend(*args: Any, **kwargs: Any) -> Any:
    from .registry import resolve_backend as _resolve_backend

    return _resolve_backend(*args, **kwargs)


def apply_degradation(*args: Any, **kwargs: Any) -> Any:
    from .apply import apply_degradation as _apply_degradation

    return _apply_degradation(*args, **kwargs)


def degrade_image_with_annotation(*args: Any, **kwargs: Any) -> Any:
    from .apply import degrade_image_with_annotation as _degrade_image_with_annotation

    return _degrade_image_with_annotation(*args, **kwargs)
