# src/imdeg/registry.py
"""
Backend registry for imaging degradations.

This module connects:
- the canonical taxonomy (papers / terms),
- with the actual implementation functions in backends/.

Use resolve_backend(paper, term) to get a callable f(image, severity, **kwargs).
"""
from typing import Callable, Dict, Tuple, Any, List

# ARNIQA backend (your code)
from .backends.arniqa import distortion_functions as ARNIQA_FUNCTIONS

# Hendrycks ImageNet-C backend (your implementation of the original code)
# Expected to expose a dict {name: function}, e.g. corruption_dict = {...}

from .backends.hendrycks_adapter import HENDRYCKS_FUNCTIONS
from .backends.Liu_IJCV_2024.Liu_IJCV_2024_wrapper import LIU_FUNCTIONS

# -------------------------------------------------------------------
# Core registry: paper -> { term_name -> function }
# -------------------------------------------------------------------
CORRUPTION_REGISTRY: Dict[str, Dict[str, Callable]] = {
    # Hendrycks et al. (ImageNet-C)
    "Hendrycks_ICLR_2019": HENDRYCKS_FUNCTIONS,

    # Agnolucci et al. (ARNIQA distortions)
    "Agnolucci_WACV_2024": ARNIQA_FUNCTIONS,

    # Liu et al. IJCV 2024
    "Liu_IJCV_2024": LIU_FUNCTIONS,
    # "COCO_C": COCO_C_FUNCTIONS,
}


# -------------------------------------------------------------------
#   Optional aliases:
#   (paper, taxonomy_term) -> backend_key
#
# Use this if taxonomy "term" != actual function key.
# For Hendrycks, the names usually match out of the box.
# For ARNIQA, you might want to map more descriptive taxonomy names
# to the short backend keys (gaussian_blur -> gaublur, etc.).
# -------------------------------------------------------------------

BACKEND_ALIASES: Dict[Tuple[str, str], str] = {
    # Examples for ARNIQA:
    # If your taxonomy uses 'gaussian_blur', but backend uses 'gaublur':
    ("Agnolucci_WACV_2024", "gaussian_blur"): "gaublur",
    ("Agnolucci_WACV_2024", "lens_blur"): "lensblur",
    ("Agnolucci_WACV_2024", "motion_blur"): "motionblur",

    ("Agnolucci_WACV_2024", "white_noise"): "whitenoise",
    ("Agnolucci_WACV_2024", "white_noise_in_color_component"): "whitenoiseCC",
    ("Agnolucci_WACV_2024", "impulse_noise"): "impulsenoise",
    ("Agnolucci_WACV_2024", "multiplicative_gaussian_noise"): "multnoise",

    ("Agnolucci_WACV_2024", "brightness_increase"): "brighten",
    ("Agnolucci_WACV_2024", "brightness_decrease"): "darken",
    ("Agnolucci_WACV_2024", "mean_shift"): "meanshift",

    ("Agnolucci_WACV_2024", "color_diffusion"): "colordiff",
    ("Agnolucci_WACV_2024", "color_shift"): "colorshift",
    ("Agnolucci_WACV_2024", "color_saturation_1"): "colorsat1",
    ("Agnolucci_WACV_2024", "color_saturation_2"): "colorsat2",

    ("Agnolucci_WACV_2024", "jpeg_compression"): "jpeg",
    ("Agnolucci_WACV_2024", "jpeg2000_compression"): "jpeg2000",

    ("Agnolucci_WACV_2024", "jitter"): "jitter",
    ("Agnolucci_WACV_2024", "non_eccentricity_patch"): "noneccpatch",
    ("Agnolucci_WACV_2024", "pixelate"): "pixelate",
    ("Agnolucci_WACV_2024", "quantization"): "quantization",
    ("Agnolucci_WACV_2024", "color_block"): "colorblock",

    ("Agnolucci_WACV_2024", "high_sharpen"): "highsharpen",
    ("Agnolucci_WACV_2024", "linear_contrast_change"): "lincontrchange",
    ("Agnolucci_WACV_2024", "nonlinear_contrast_change"): "nonlincontrchange",

    # ===== Noise (G1) =====
    ("Hendrycks_ICLR_2019", "gaussian_noise"): "gaussian_noise",
    ("Hendrycks_ICLR_2019", "shot_noise"): "shot_noise",
    ("Hendrycks_ICLR_2019", "impulse_noise"): "impulse_noise",
    ("Hendrycks_ICLR_2019", "speckle_noise"): "speckle_noise",

    # ===== Blur (G2) =====
    ("Hendrycks_ICLR_2019", "defocus_blur"): "defocus_blur",
    ("Hendrycks_ICLR_2019", "glass_blur"): "glass_blur",
    ("Hendrycks_ICLR_2019", "motion_blur"): "motion_blur",
    ("Hendrycks_ICLR_2019", "zoom_blur"): "zoom_blur",
    ("Hendrycks_ICLR_2019", "gaussian_blur"): "gaussian_blur",

    # ===== Weather (G8) =====
    ("Hendrycks_ICLR_2019", "snow"): "snow",
    ("Hendrycks_ICLR_2019", "frost"): "frost",
    ("Hendrycks_ICLR_2019", "fog"): "fog",
    ("Hendrycks_ICLR_2019", "spatter"): "spatter",

    # ===== Brightness / Exposure (G6) =====
    ("Hendrycks_ICLR_2019", "brightness"): "brightness",

    # ===== Sharpness / Contrast / Texture (G10) =====
    ("Hendrycks_ICLR_2019", "contrast"): "contrast",
    ("Hendrycks_ICLR_2019", "saturate"): "saturate",

    # ===== Geometry / Spatial (G7) =====
    ("Hendrycks_ICLR_2019", "elastic_transform"): "elastic_transform",

    # ===== Resolution / Sampling (G3) =====
    ("Hendrycks_ICLR_2019", "pixelate"): "pixelate",

    # ===== Compression (G4) =====
    ("Hendrycks_ICLR_2019", "jpeg_compression"): "jpeg_compression",

    ##################################################################
    #
    # Camera damage
    ("Liu_IJCV_2024", "fog"): "fog",
    ("Liu_IJCV_2024", "lens_obstruction"): "lens_obstruction",
    ("Liu_IJCV_2024", "focus_motor_damage"): "focus_motor_damage",
    ("Liu_IJCV_2024", "ccd_sensor_damage"): "ccd_sensor_damage",
    ("Liu_IJCV_2024", "cmos_sensor_damage"): "cmos_sensor_damage",

    # ISP damage
    ("Liu_IJCV_2024", "insufficient_blc"): "insufficient_blc",
    ("Liu_IJCV_2024", "excessive_blc"): "excessive_blc",
    ("Liu_IJCV_2024", "lens_shading_damage"): "lens_shading_damage",
    ("Liu_IJCV_2024", "awb_damage"): "awb_damage",
    ("Liu_IJCV_2024", "bad_pixel_correction"): "bad_pixel_correction",
    ("Liu_IJCV_2024", "cfa_interpolation_damage"): "cfa_interpolation_damage",
    ("Liu_IJCV_2024", "gamma_damage"): "gamma_damage",
    ("Liu_IJCV_2024", "color_space_conversion_damage"): "color_space_conversion_damage",

    # Board-level damage
    ("Liu_IJCV_2024", "memory_exceptions"): "memory_exceptions",
    ("Liu_IJCV_2024", "transfer_harness_exceptions"): "transfer_harness_exceptions",
    ("Liu_IJCV_2024", "sensor_broken"): "sensor_broken",
}


# -------------------------------------------------------------------
# Backend capabilities:
# (paper, term) -> {"mode": "param" | "severity"}
#
# "param"    : backend expects (image, param)
# "severity" : backend expects (image, severity=1..5)
#
# NOTE:
#  - We provide entries for both backend keys and alias/taxonomy keys
#    where needed, so that apply_degradation can be called either with
#    backend names (e.g. "gaublur") or taxonomy-style names
#    (e.g. "gaussian_blur").
# -------------------------------------------------------------------

BACKEND_CAPS: Dict[Tuple[str, str], Dict[str, Any]] = {}

# ----------------- ARNIQA: param-based -----------------
_arn_param_terms = [
    # backend keys
    "gaublur",
    "lensblur",
    "motionblur",
    "whitenoise",
    "whitenoiseCC",
    "impulsenoise",
    "multnoise",
    "brighten",
    "darken",
    "meanshift",
    "colordiff",
    "colorshift",
    "colorsat1",
    "colorsat2",
    "jpeg2000",
    "jpeg",
    "jitter",
    "noneccpatch",
    "pixelate",
    "quantization",
    "colorblock",
    "highsharpen",
    "lincontrchange",
    "nonlincontrchange",
]

for t in _arn_param_terms:
    BACKEND_CAPS[("Agnolucci_WACV_2024", t)] = {"mode": "param"}

# taxonomy/alias names for ARNIQA (same mode)
_arn_alias_terms = [
    "gaussian_blur",
    "lens_blur",
    "motion_blur",
    "white_noise",
    "white_noise_in_color_component",
    "impulse_noise",
    "multiplicative_gaussian_noise",
    "brightness_increase",
    "brightness_decrease",
    "mean_shift",
    "color_diffusion",
    "color_shift",
    "color_saturation_1",
    "color_saturation_2",
    "jpeg_compression",
    "jpeg2000_compression",
    "jitter",  # same name as backend
    "non_eccentricity_patch",
    "pixelate",       # same name as backend
    "quantization",   # same name as backend
    "color_block",
    "high_sharpen",
    "linear_contrast_change",
    "nonlinear_contrast_change",
]

for t in _arn_alias_terms:
    BACKEND_CAPS[("Agnolucci_WACV_2024", t)] = {"mode": "param"}


# ----------------- Hendrycks ImageNet-C: severity-based -----------------
_hend_severity_terms = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
    "speckle_noise",
    "gaussian_blur",
    "spatter",
    "saturate",
]

for t in _hend_severity_terms:
    BACKEND_CAPS[("Hendrycks_ICLR_2019", t)] = {"mode": "severity"}


# ----------------- Liu IJCV 2024: severity-based -----------------
#
_liu_severity_terms = [
    # Camera damage
    "fog",
    "lens_obstruction",
    "focus_motor_damage",
    "ccd_sensor_damage",
    "cmos_sensor_damage",

    # ISP damage
    "insufficient_blc",
    "excessive_blc",
    "lens_shading_damage",
    "awb_damage",
    "bad_pixel_correction",
    "cfa_interpolation_damage",
    "gamma_damage",
    "color_space_conversion_damage",

    # Board-level damage
    "memory_exceptions",
    "transfer_harness_exceptions",
    "sensor_broken",
]

for t in _liu_severity_terms:
    BACKEND_CAPS[("Liu_IJCV_2024", t)] = {"mode": "severity"}


# -------------------------------------------------------------------
# Resolver: given (paper, term), return the backend function
# -------------------------------------------------------------------

def resolve_backend(paper: str, term: str) -> Callable:
    """
    Resolve the implementation function for a given (paper, term).

    This first looks up the paper-level registry, and then:
    - tries the term directly as a key,
    - if that fails, tries the BACKEND_ALIASES[(paper, term)] mapping.

    Raises KeyError if nothing matches.
    """
    if paper not in CORRUPTION_REGISTRY:
        raise KeyError(f"Unknown paper {paper!r} in CORRUPTION_REGISTRY.")

    backend = CORRUPTION_REGISTRY[paper]

    # Direct hit?  term is original term
    if term in backend:
        return backend[term]

    # If in the future you use more descriptive taxonomy names,
    # you can add a BACKEND_ALIASES dict here.
    alias_key = BACKEND_ALIASES.get((paper, term))
    if alias_key is not None and alias_key in backend:
        return backend[alias_key]

    raise KeyError(
        f"No backend function found for paper={paper!r}, term={term!r}. "
        f"Consider adding an entry to BACKEND_ALIASES."
    )


# -------------------------------------------------------------------
# Registry introspection helpers (for users)
# -------------------------------------------------------------------

def list_available_papers() -> List[str]:
    """
    Return a sorted list of available paper keys, e.g.
    ["Agnolucci_WACV_2024", "Hendrycks_ICLR_2019"].
    """
    return sorted(CORRUPTION_REGISTRY.keys())


def list_paper_terms(
    paper: str,
) -> List[Dict[str, Any]]:
    """
    List all backend functions for a given paper, grouped with their aliases.

    Returns one entry *per backend function*:

        {
          "backend_key": <original backend name, e.g. "gaublur">,
          "aliases": ["gaussian_blur", ...],   # taxonomy / new names
          "mode": "param" | "severity" | None
        }

    So:
      - backend_key ≈ original implementation / paper name
      - aliases     ≈ taxonomy/unified names that map to that key
    """
    if paper not in CORRUPTION_REGISTRY:
        raise KeyError(f"Paper {paper!r} not in CORRUPTION_REGISTRY.")

    backend = CORRUPTION_REGISTRY[paper]

    # Build: backend_key -> [alias1, alias2, ...]
    aliases_by_backend: Dict[str, List[str]] = {k: [] for k in backend.keys()}

    for (p, alias_term), backend_key in BACKEND_ALIASES.items():
        if p != paper:
            continue
        if backend_key in backend:
            aliases_by_backend.setdefault(backend_key, []).append(alias_term)

    out: List[Dict[str, Any]] = []
    for backend_key in sorted(backend.keys()):
        mode = BACKEND_CAPS.get((paper, backend_key), {}).get("mode")
        alias_list = sorted(aliases_by_backend.get(backend_key, []))
        out.append(
            {
                "backend_key": backend_key,   # original name / implementation key
                "aliases": alias_list,        # new / taxonomy names
                "mode": mode,
            }
        )
    return out



def print_registry_summary(verbose: bool = False) -> None:
    """
    Pretty-print a summary of available papers and methods.

    Example (non-verbose):

      [imgdeg] Available papers and methods:

        - Agnolucci_WACV_2024:
            24 backend functions
            24 taxonomy/alias names
            * 24 functions with mode=param

        - Hendrycks_ICLR_2018:
            19 backend functions
            19 taxonomy/alias names
            * 19 functions with mode=severity

    If verbose=True, also prints each backend function with its aliases.
    """
    print("[imgdeg] Available papers and methods:\n")
    for paper in list_available_papers():
        backend = CORRUPTION_REGISTRY[paper]
        terms_info = list_paper_terms(paper)

        n_backend = len(backend)
        n_aliases = sum(len(info["aliases"]) for info in terms_info)

        print(f"  - {paper}:")
        print(f"      {n_backend} backend functions")
        print(f"      {n_aliases} taxonomy/alias names")

        # Aggregate by mode
        by_mode: Dict[str, int] = {}
        for info in terms_info:
            mode = info.get("mode") or "unknown"
            by_mode[mode] = by_mode.get(mode, 0) + 1

        for mode, count in by_mode.items():
            print(f"      * {count} functions with mode={mode}")

        if verbose:
            print("      Functions:")
            for info in terms_info:
                backend_key = info["backend_key"]
                mode = info.get("mode") or "unknown"
                aliases = info["aliases"]
                if aliases:
                    alias_str = ", ".join(aliases)
                else:
                    alias_str = "-"
                print(
                    f"        - {backend_key} "
                    f"(mode={mode}, aliases: {alias_str})"
                )
        print("")

def parse_cli_degradations(degradations):
    """Parse and categorize degradation names."""
    if degradations is None:
        return None
    
    parsed_degradations = []
    papers_clear_name = ["hendrycks_", "arniqa_", "liu_"]

    for idx, paper in enumerate(CORRUPTION_REGISTRY.keys()):
        paper_clear_name = papers_clear_name[idx]
        for degradation in degradations[:]:                         # [:] creates a shallow copy, to allow .remove() in for loop
            if degradation.startswith(paper_clear_name) or degradation in CORRUPTION_REGISTRY[paper]:
                parsed_degradations.append({
                    "name": degradation.removeprefix(paper_clear_name),
                    "paper": paper 
                })
                degradations.remove(degradation)

    for degradation in degradations:
        raise ValueError(f"Degradation does not exist: {degradation}")
    
    return parsed_degradations