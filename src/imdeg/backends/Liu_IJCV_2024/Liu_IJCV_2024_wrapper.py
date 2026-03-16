"""
Liu IJCV 2024 backend wrappers for real-world image corruption functions.

This module provides the integration layer between the imdeg API and the
real-world corruption benchmark introduced in:

    Liu, Jiawei and Wang, Zhijie and Ma, Lei and Fang, Chunrong and Bai, Tongtong
    and Zhang, Xufan and Liu, Jia and Chen, Zhenyu.
    "Benchmarking Object Detection Robustness against Real-World Corruptions."
    International Journal of Computer Vision, 132(10):4398-4416, 2024.
    DOI: 10.1007/s11263-024-02096-6
    https://doi.org/10.1007/s11263-024-02096-6

Project website:
    https://sites.google.com/view/real-worldbenchmark/real-world-benchmarking

Reference repository:
    https://github.com/Jackie-Bai888/image-noise-pattern

Licensing note:
    The original upstream implementation files are not distributed with this
    project because the licensing status of the reference repository is unclear.

Backend design:
    - user-supplied upstream files may be placed under:
          external/liu_image_noise_pattern/
    - selected upstream functions are loaded dynamically at runtime
    - small mechanical compatibility fixes may be applied during loading
      (for example: removing demo code, removing problematic imports,
      injecting local helper utilities)
    - degradations that require more than mechanical compatibility fixes are
      implemented locally as adapted compatibility versions

This file therefore contains:
    1. wrappers around user-supplied external implementations, and
    2. calls to maintained local adapted implementations where needed

The goal is to provide a stable torch-based corruption backend without
redistributing the original upstream source code.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Optional, Literal

import numpy as np
import torch
from typing import Dict
from functools import lru_cache
import importlib.util
import types
import re
from imdeg.backends.Liu_IJCV_2024.Liu_IJCV_2024_adapted import lens_shading_error_adapted, color_space_error_adapted, memory_corruption_adapted, sensor_broken_adapted
from imdeg.severity import map_severity_to_range
from imdeg.utils.image_conversion import tensor_to_bgr_uint8,bgr_uint8_to_tensor, RGB2Bayer_RG
import importlib.util
import pathlib
from typing import Callable

def _strip_demo_block(source: str) -> str:
    """
    Remove common bottom-of-file demo/test blocks from third-party scripts.

    This is intentionally conservative:
    - remove anything under `if __name__ == "__main__":`
    - remove blocks starting with markers like '#test', '# test', "''' test"
    - remove trailing code that starts with `src = cv2.imread(...)`
      or `cv2.imshow(...)` if present
    """

    # 1) safest case: remove __main__ block
    source = re.sub(
        r'(?ms)^if\s+__name__\s*==\s*[\'"]__main__[\'"]\s*:\s*\n.*$',
        '',
        source,
    )

    # 2) remove obvious test/demo marker sections
    demo_markers = [
        r'(?ms)^#\s*test\s*$.*$',
        r"(?ms)^'''\s*$.*$",
        r'(?ms)^"""\s*$.*$',
    ]
    for pattern in demo_markers:
        m = re.search(pattern, source)
        if m:
            source = source[:m.start()]
            break

    # 3) remove common trailing OpenCV demo lines
    lines = source.splitlines()
    cut_idx = None
    demo_prefixes = (
        "src = cv2.imread(",
        "image = cv2.imread(",
        "img = cv2.imread(",
        "cv2.imshow(",
        "cv2.waitKey(",
        "cv2.destroyAllWindows(",
    )

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(demo_prefixes):
            cut_idx = i
            break

    if cut_idx is not None:
        lines = lines[:cut_idx]

    return "\n".join(lines) + "\n"


def _apply_small_source_fixes(source: str) -> str:
    """
    Optional tiny compatibility fixes.

    Keep this for mechanical edits only:
    - add missing numpy import if needed
    - remove exact known bad lines
    - normalize obvious script-only code

    Do NOT rewrite algorithms here.
    """

    # Example: if code uses np. but forgot to import numpy
    if "np." in source and "import numpy as np" not in source:
        source = "import numpy as np\n" + source

    return source


@lru_cache(maxsize=None)
def _load_patched_module(module_filename: str):
    """
    Load a third-party Python file from external/... after applying
    tiny preprocessing fixes, then return it as an in-memory module.
    """
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    module_path = repo_root / "external" / "liu_image_noise_pattern" / module_filename

    if not module_path.exists():
        raise RuntimeError(
            f"Cannot find external module file: {module_path}\n"
            "Please place the original third-party file in:\n"
            "  external/liu_image_noise_pattern/\n"
        )

    source = module_path.read_text(encoding="utf-8")
    source = _strip_demo_block(source)
    source = _apply_small_source_fixes(source)
    source = _apply_file_specific_fixes(module_filename, source)

    module_name = f"_liu_patched_{module_path.stem}"
    module = types.ModuleType(module_name)
    module.__file__ = str(module_path)

    # Inject compatibility helpers here
    module.__dict__["RGB2Bayer_RG"] = RGB2Bayer_RG

    exec(compile(source, str(module_path), "exec"), module.__dict__)
    return module

def _import_liu_function(module_filename: str, func_name: str) -> Callable:
    module = _load_patched_module(module_filename)

    try:
        fn = getattr(module, func_name)
    except AttributeError:
        raise RuntimeError(
            f"Function {func_name!r} not found in external file {module_filename!r}."
        )

    if not callable(fn):
        raise RuntimeError(
            f"Object {func_name!r} in {module_filename!r} exists but is not callable."
        )

    return fn

def _import_liu_function_(module_filename: str, func_name: str) -> Callable:
    """
    Load a single function (e.g. fog) from a Liu IJCV 2024 module file
    under external/liu_image_noise_pattern/.

    module_filename: e.g. "F.py"
    func_name:       e.g. "fog"
    """
    # repo_root/imgdeg/backends/liu_backend.py  -> repo_root
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    module_path = repo_root / "external" / "liu_image_noise_pattern" / module_filename

    if not module_path.exists():
        raise RuntimeError(
            f"Cannot find Liu module file {module_path}.\n"
            "Place the original Liu IJCV 2024 .py files in:\n"
            "  external/liu_image_noise_pattern/\n"
            "and retry."
        )

    module_name = module_filename.rsplit(".", 1)[0]  # "F.py" -> "F"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    try:
        return getattr(module, func_name)
    except AttributeError:
        raise RuntimeError(
            f"Function {func_name!r} not found in {module_path}. "
            "Check that the Liu file defines this function."
        )



# Small helper to keep signature consistent
def _wrap_tensor(
    img: torch.Tensor,
    fn,
    *args,
    **kwargs,
) -> torch.Tensor:
    """
    Convert tensor -> BGR uint8 -> apply fn -> tensor.
    """
    device = img.device
    bgr = tensor_to_bgr_uint8(img)
    out = fn(bgr, *args, **kwargs)
    # some Liu ops (e.g. fog) return float 0..1
    if out.dtype != np.uint8:
        out = (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)
    return bgr_uint8_to_tensor(out, device=device)

def _apply_file_specific_fixes(module_filename: str, source: str) -> str:
    if module_filename in {
        "CCD-D_CMOS-D.py",
        "I-BLC_E-BLC.py",
        "BPC-D.py",
        "LSC-D.py",
    }:
        source = re.sub(
            r'^\s*from\s+RGB2RAW\s+import\s+RGB2Bayer_RG\s*(#.*)?$',
            '',
            source,
            flags=re.MULTILINE,
        )

    return source

# ------------------------------------------------------------
# Camera damage (Fog, Lens Obstruction, Focus Motor, CCD/CMOS)
# ------------------------------------------------------------

def fog(image: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """
    Fog (F) – Liu et al. IJCV 2024
    """
    liu_fog = _import_liu_function("F.py", "fog")
    pot = map_severity_to_range(severity, lo=0.05, hi=0.25)
    return _wrap_tensor(image, liu_fog, pot=pot)


def lens_obstruction(image: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """
    Lens Obstruction (LO) – simulate drops on lens.
    """
    liu_generate_rain_drop = _import_liu_function("LO.py", "generate_rain_drop")
    count = int(round(map_severity_to_range(severity, lo=50, hi=400)))
    return _wrap_tensor(image, liu_generate_rain_drop, count=count)


def focus_motor_damage(image: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """
    Focus Motor Damage (FM-D) – out-of-focus blur via lose_focus.
    """
    # -------------------------------------------------------------------
    # FM-D
    # https://github.com/Jackie-Bai888/image-noise-pattern/blob/main/FM-D.py
    # -------------------------------------------------------------------
    liu_lose_focus = _import_liu_function("FM-D.py", "lose_focus")
    level = int(round(map_severity_to_range(severity, lo=2, hi=15)))
    level = max(1, level)
    return _wrap_tensor(image, liu_lose_focus, level=level)


def ccd_sensor_damage(image: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """
    CCD Sensor Damage (CCD-D) – white dots + vertical streaks.
    """
    # -------------------------------------------------------------------
    # CCD-D_CMOS-D
    # https://github.com/Jackie-Bai888/image-noise-pattern/blob/main/CCD-D_CMOS-D.py
    # -
    sensor_damage = _import_liu_function("CCD-D_CMOS-D.py", "sensor_damage")
    bgr = tensor_to_bgr_uint8(image)
    h, w, _ = bgr.shape
    num = int(round(map_severity_to_range(severity, lo=5, hi=40)))
    out = sensor_damage(bgr, h=h, w=w, num=num, sensor='ccd')
    return bgr_uint8_to_tensor(out, device=image.device)


def cmos_sensor_damage(image: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """
    CMOS Sensor Damage (CMOS-D) – dark lines + dots.
    """
    # -------------------------------------------------------------------
    # CCD-D_CMOS-D
    # https://github.com/Jackie-Bai888/image-noise-pattern/blob/main/CCD-D_CMOS-D.py
    # -------------------------------------------------------------------
    sensor_damage = _import_liu_function("CCD-D_CMOS-D.py", "sensor_damage")
    bgr = tensor_to_bgr_uint8(image)
    h, w, _ = bgr.shape
    num = int(round(map_severity_to_range(severity, lo=5, hi=40)))
    out = sensor_damage(bgr, h=h, w=w, num=num, sensor='cmos')
    return bgr_uint8_to_tensor(out, device=image.device)


# ------------------------------------------------------------
# ISP damage
# ------------------------------------------------------------

def insufficient_blc(image: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """
    I-BLC – insufficient black level correction.
    """
    # -------------------------------------------------------------------
    # I-BLC_E-BLC
    # https://github.com/Jackie-Bai888/image-noise-pattern/blob/main/I-BLC_E-BLC.py
    # -------------------------------------------------------------------
    black_level_correct_less = _import_liu_function("I-BLC_E-BLC.py", "black_level_correct_less")
    color_num = int(round(map_severity_to_range(severity, lo=64, hi=255)))
    return _wrap_tensor(image, black_level_correct_less, color_num=color_num)


def excessive_blc(image: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """
    E-BLC – excessive black level correction.
    """
    # -------------------------------------------------------------------
    # I-BLC_E-BLC
    # https://github.com/Jackie-Bai888/image-noise-pattern/blob/main/I-BLC_E-BLC.py
    # -------------------------------------------------------------------
    black_level_correct_more = _import_liu_function("I-BLC_E-BLC.py", "black_level_correct_more")
    color_num = int(round(map_severity_to_range(severity, lo=64, hi=255)))
    return _wrap_tensor(image, black_level_correct_more, color_num=color_num)


def lens_shading_damage(image: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """
    LSC-D – lens shading correction damage.
    """
    #
    alpha = map_severity_to_range(severity, lo=0.1, hi=0.8)
    return _wrap_tensor(image, lens_shading_error_adapted, alpha=alpha)


def awb_damage(image: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """
    AWB-D – automatic white balance damage.
    """
    # -------------------------------------------------------------------
    # AWB-D
    # https://github.com/Jackie-Bai888/image-noise-pattern/blob/main/AWB-D.py
    # -------------------------------------------------------------------
    white_balance_error = _import_liu_function("AWB-D.py", "white_balance_error")
    precent = map_severity_to_range(severity, lo=0.05, hi=0.25)
    return _wrap_tensor(image, white_balance_error, precent=precent)


def bad_pixel_correction(image: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """
    BPC-D – bad pixel correction damage.
    """
    # -------------------------------------------------------------------
    # BPC-D
    # https://github.com/Jackie-Bai888/image-noise-pattern/blob/main/BPC-D.py
    # -------------------------------------------------------------------
    ori_bad_pixel_error = _import_liu_function("BPC-D.py", "ori_bad_pixel_error")
    num_precent = map_severity_to_range(severity, lo=0.01, hi=0.6)
    return _wrap_tensor(image, ori_bad_pixel_error, num_precent=num_precent)


def cfa_interpolation_damage(image: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """
    CFAI-D – CFA interpolation damage.
    """
    # -------------------------------------------------------------------
    # CFAI-D
    # https://github.com/Jackie-Bai888/image-noise-pattern/blob/main/CFAI-D.py
    # -------------------------------------------------------------------
    cfa_interpolation_error = _import_liu_function("CFAI-D.py", "cfa_interpolation_error")
    num_precent = map_severity_to_range(severity, lo=0.05, hi=0.6)
    return _wrap_tensor(image, cfa_interpolation_error, num_precent=num_precent)


def gamma_damage(image: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """
    GC-D – gamma correction damage.
    """
    # -------------------------------------------------------------------
    # GC-d
    # https://github.com/Jackie-Bai888/image-noise-pattern/blob/main/GC-d.py
    # -------------------------------------------------------------------
    gamma_correction = _import_liu_function("GC-d.py", "gamma_correction")
    gamma_par = map_severity_to_range(severity, lo=0.5, hi=3.0)
    return _wrap_tensor(image, gamma_correction, gamma_par=gamma_par)


def color_space_conversion_damage(image: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """
    CSC-D – color space conversion damage.
    """
    num_precent = map_severity_to_range(severity, lo=0.05, hi=0.6)
    return _wrap_tensor(image, color_space_error_adapted, num_precent=num_precent)


# ------------------------------------------------------------
# Board-level damage
# ------------------------------------------------------------

def memory_exceptions(image: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """
    ME – memory exceptions. We use Liu's memory_corruption (no GT needed).
    """
    # could scale strength with severity via custom logic if needed
    return _wrap_tensor(image, memory_corruption_adapted, severity=severity)


def transfer_harness_exceptions(image: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """
    THE – transfer harness exceptions. Use line_damage_error2 (bit errors).
    """
    # -------------------------------------------------------------------
    #
    # https://github.com/Jackie-Bai888/image-noise-pattern/blob/main/THE.py
    # -------------------------------------------------------------------
    line_damage_error2 = _import_liu_function("THE.py", "line_damage_error2")
    num_precent = map_severity_to_range(severity, lo=0.05, hi=0.5)
    return _wrap_tensor(image, line_damage_error2, num_precent=num_precent)

def sensor_broken(image: torch.Tensor, severity: int = 3) -> torch.Tensor:
    """
    sensorBroken – adapted version with severity-controlled parameters.
    """

    spot_count = int(round(map_severity_to_range(severity, lo=3, hi=15)))
    spot_alpha = map_severity_to_range(severity, lo=0.2, hi=0.8)
    deviation_strength = map_severity_to_range(severity, lo=0.2, hi=1.0)
    leak_length = int(round(map_severity_to_range(severity, lo=10, hi=40)))

    # Optional staged activation
    spot = True
    deviation = severity >= 2
    leak = severity >= 3

    return _wrap_tensor(
        image,
        sensor_broken_adapted,
        spot_count=spot_count,
        spot_alpha=spot_alpha,
        deviation_strength=deviation_strength,
        leak_length=leak_length,
        spot=spot,
        deviation=deviation,
        leak=leak,
    )

# def sensor_broken(image: torch.Tensor, severity: int = 3) -> torch.Tensor:
#     """
#     sensorBroken – aggregated real sensor damage (spot + deviation + leak).
#     Currently severity does not change internal flags, but can be extended.
#     """
#     # -------------------------------------------------------------------
#     # sensorBroken
#     # https://github.com/Jackie-Bai888/image-noise-pattern/blob/main/sensorBroken.py
#     # -------------------------------------------------------------------
#     sensor_broken = _import_liu_function("sensorBroken.py", "sensor_broken")
#     return _wrap_tensor(image, sensor_broken, spot=True, deviation=True, leak=True)


# ------------------------------------------------------------
# Registry: backend_key -> function
# ------------------------------------------------------------
LIU_FUNCTIONS: Dict[str, callable] = {
    # Camera damage
    "fog": fog,
    "lens_obstruction": lens_obstruction,
    "focus_motor_damage": focus_motor_damage,
    "ccd_sensor_damage": ccd_sensor_damage,
    "cmos_sensor_damage": cmos_sensor_damage,

    # ISP damage
    "insufficient_blc": insufficient_blc,
    "excessive_blc": excessive_blc,
    "lens_shading_damage": lens_shading_damage,
    "awb_damage": awb_damage,
    "bad_pixel_correction": bad_pixel_correction,
    "cfa_interpolation_damage": cfa_interpolation_damage,
    "gamma_damage": gamma_damage,
    "color_space_conversion_damage": color_space_conversion_damage,

    # Board-level damage
    "memory_exceptions": memory_exceptions,
    "transfer_harness_exceptions": transfer_harness_exceptions,
    "sensor_broken": sensor_broken,
}

liu_full_names= {
    # Noise
    "fog": "Fog",
    "lens_obstruction": "Lens Obstruction",
    "focus_motor_damage": "Focus Motor Damage",
    "ccd_sensor_damage": "Ccd Sensor Damage",
    "cmos_sensor_damage": "Cmos Sensor Damage",

    # ISP damage
    "insufficient_blc": "Insufficient Blc",
    "excessive_blc": "Excessive Blc",
    "lens_shading_damage": "Lens Shading Damage",
    "awb_damage": "Awb Damage",
    "bad_pixel_correction": "Bad Pixel Correction",
    "cfa_interpolation_damage": "Cfa interpolation damage",
    "gamma_damage": "Gamma Damage",
    "color_space_conversion_damage": "Color Space Conversion Damage",

    # Board-level damage
    "memory_exceptions": "Memory Exceptions",
    "transfer_harness_exceptions": "Transfer Harness Exceptions",
    "sensor_broken": "Sensor Broken",
}














