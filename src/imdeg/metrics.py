# src/imgdeg/metrics.py

import math
from typing import Literal, Optional, Tuple
import torch
import torch.nn.functional as F

try:
    import piq  # optional for SSIM, LPIPS, etc.
    HAS_PIQ = True
except ImportError:
    HAS_PIQ = False

def _to_float01(t: torch.Tensor, strict: bool = False) -> torch.Tensor:
    """
    Ensure float32 in [0,1]. If incoming floats appear to be 0..255, rescale.
    Tolerates tiny overshoots (±5e-3) by clamping. If strict=True, raise on out-of-range.
    """
    if not t.is_floating_point():
        t = t.to(torch.float32)
    with torch.no_grad():
        tmax = float(t.max())
        tmin = float(t.min())

    # Heuristic rescale for 0..255 floats
    if tmax > 1.5:
        t = t / 255.0
        with torch.no_grad():
            tmax = float(t.max())
            tmin = float(t.min())

    eps = 5e-3  # tolerate small numeric overshoots
    if strict and (tmin < -eps or tmax > 1.0 + eps):
        raise ValueError(f"Input tensor outside [0,1]: min={tmin}, max={tmax}")

    # Clamp tiny overshoots; hard clip anything beyond tolerance
    t = t.clamp(0.0, 1.0).to(torch.float32)
    return t


def psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0, cap_max: Optional[float] = None) -> float:
    """
    Compute PSNR in dB, optionally capped.

    Args:
        x, y: tensors (C,H,W) or (N,C,H,W), float in [0,1]
        max_val: max intensity (default 1.0)
        cap_max: if set, clips the result to at most this value (e.g., 50.0)

    Returns:
        PSNR in decibels (float), capped if specified
    """
    # Constant for numerical stability
    EPS = 1e-8
    if x.dim() == 3: x = x.unsqueeze(0)
    if y.dim() == 3: y = y.unsqueeze(0)

    # Ensure same spatial dimensions
    if x.shape[2:] != y.shape[2:]:
        # Resize y to match x dimensions
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)

    # Ensure values in [0,1] due to floating point rounding
    #x = torch.clamp(x, 0.0, 1.0)
    #y = torch.clamp(y, 0.0, 1.0)
    x = _to_float01(x)
    y = _to_float01(y)

    if HAS_PIQ:
        psnr_val = piq.psnr(x, y, data_range=1.0, reduction='mean').item()
    else:
        mse = F.mse_loss(x, y).item()
        psnr_val = float("inf") if mse == 0 else 10.0 * math.log10(max_val ** 2 / (mse+EPS))

    if cap_max is not None:
        return min(psnr_val, cap_max)
    return psnr_val



def mse(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute Mean Squared Error.
    """
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")
    if x.dim() not in (3, 4):
        raise ValueError("Expected (C,H,W) or (N,C,H,W)")
    if not x.is_floating_point() or not y.is_floating_point():
        raise TypeError("Expected float tensors")

    if x.dim() == 3: x = x.unsqueeze(0)
    if y.dim() == 3: y = y.unsqueeze(0)

    # Ensure same spatial dimensions
    if x.shape[2:] != y.shape[2:]:
        # Resize y to match x dimensions
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
    # Ensure values in [0,1] due to floating point rounding
    x = torch.clamp(x, 0.0, 1.0)
    y = torch.clamp(y, 0.0, 1.0)

    return F.mse_loss(x, y).item()


_LPIPS_MODEL = None


def lpips(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute Learned Perceptual Image Patch Similarity (LPIPS).
    Requires piq library.

    Args:
        x, y: tensors (C,H,W) or (N,C,H,W), float in [0,1]

    Returns:
        LPIPS distance (lower = more similar)
    """
    global _LPIPS_MODEL

    if not HAS_PIQ:
        raise RuntimeError("LPIPS requires the piq library.")

    if x.dim() == 3: x = x.unsqueeze(0)
    if y.dim() == 3: y = y.unsqueeze(0)

    if x.shape[2:] != y.shape[2:]:
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)

    x = _to_float01(x)
    y = _to_float01(y)

    # Initialize LPIPS model once and move to correct device
    if _LPIPS_MODEL is None:
        _LPIPS_MODEL = piq.LPIPS(replace_pooling=True, reduction='mean')

    # Move model to same device as input
    device = x.device
    _LPIPS_MODEL = _LPIPS_MODEL.to(device)

    return _LPIPS_MODEL(x, y).item()


def ssim_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Returns 1 - SSIM(x,y). Uses piq.SSIM if available, else L1-blurred fallback.
    """
    if x.dim() == 3: x = x.unsqueeze(0)
    if y.dim() == 3: y = y.unsqueeze(0)

    # Ensure same spatial dimensions
    if x.shape[2:] != y.shape[2:]:
        # Resize y to match x dimensions
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)

    # Ensure same device and dtype
    if x.device != y.device:
        y = y.to(x.device)
    if x.dtype != y.dtype:
        # Use float32 for consistency with most piq setups
        x = x.to(torch.float32)
        y = y.to(torch.float32)
    # Ensure values in [0,1] due to floating point rounding
    x = torch.clamp(x, 0.0, 1.0)
    y = torch.clamp(y, 0.0, 1.0)

    if HAS_PIQ:
        ssim_val = piq.ssim(x, y, data_range=1.0).item()
        return 1.0 - ssim_val
    else:
        x_blur = F.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
        y_blur = F.avg_pool2d(y, kernel_size=7, stride=1, padding=3)
        return F.l1_loss(x_blur, y_blur).item()


def distortion_metric(
    x: torch.Tensor,
    y: torch.Tensor,
    kind: Literal["psnr_drop", "1-ssim", "psnr", "mse", "lpips"] = "psnr_drop",
    cap_max_psnr: Optional[float] = 50.0,
) -> float:
    """
    Unified distortion metric: higher = worse distortion.

    Args:
        x, y: input tensors in [0,1]
        kind: metric to compute
        cap_max_psnr: used for 'psnr_drop', defines what PSNR is treated as "perfect"

    Returns:
        Scalar distortion score
    """
    if kind == "psnr_drop":
        # PSNR drop from perfect (defined by cap)
        psnr_val = psnr(x, y, max_val=1.0)
        return 1.0 / max(psnr_val, 1e-6)

    elif kind == "1-ssim":
        return ssim_distance(x, y)

    elif kind == "psnr":
        #return psnr(x, y, max_val=1.0, cap_max=cap_max_psnr)
        return psnr(x, y, max_val=1.0)

    elif kind == "mse":
        return mse(x, y)

    elif kind == "lpips":
        if not HAS_PIQ:
            raise RuntimeError("LPIPS requires the piq library.")
        return lpips(x, y)

    else:
        raise ValueError(f"Unknown distortion metric kind: {kind}")
