"""
Torch-compatible wrappers for ImageNet-C / common corruption functions.

This module provides the integration layer between the imdeg API and common
corruption implementations derived from ImageNet-C style benchmarks.

Original benchmark paper:
    Hendrycks, Dan and Dietterich, Thomas.
    "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations."
    ICLR, 2019.
    https://openreview.net/forum?id=HJz6tiCqYm

Reference implementation lineage:
    - original ImageNet-C / Hendrycks robustness repository:
      https://github.com/hendrycks/robustness/tree/master/ImageNet-C/imagenet_c
    - size-generalized implementation adapted from:
      https://github.com/asharakeh/probdet/blob/master/src/probabilistic_inference/image_corruptions.py
    - additional reference implementation for resize-compatible corruptions:
      https://github.com/bethgelab/imagecorruptions

Related papers:
    Harakeh, Ali and Waslander, Steven L.
    "Estimating and Evaluating Regression Predictive Uncertainty in Deep Object Detectors."
    ICLR, 2021.
    https://openreview.net/forum?id=YLewtnvKgR7

    Michaelis, Claudio and Mitzkus, Benjamin and Geirhos, Robert and Rusak, Evgenia and
    Bringmann, Oliver and Ecker, Alexander S. and Bethge, Matthias and Brendel, Wieland.
    "Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming."
    NeurIPS ML4AD, 2019.
    https://arxiv.org/abs/1907.07484

License:
    Apache License 2.0

Modifications in this project:
    - integrated into the imdeg backend interface
    - adapted for tensor-based input/output handling
    - used size-compatible variants for arbitrary image resolutions
    - reorganized for backend registry integration

Backend design:
    - corruption functions operate on PIL images or NumPy arrays
    - this wrapper converts between torch tensors and the expected input/output
      representation
    - the goal is to expose a stable tensor-based backend API within imdeg

This module mainly contains wrapper logic; the corruption implementations
themselves are defined in the corresponding Hendrycks-style backend module.
"""

import torch
from ..utils.image_conversion import tensor_to_numpy_uint8, numpy_uint8_to_tensor, tensor_to_pil, pil_or_np_to_tensor
from .hendrycks import corruption_dict
from PIL import Image
from typing import Callable, Dict

def wrap_hendrycks_function(fn: Callable) -> Callable:
    """
       Wrap any ImageNet-C corruption to accept torch tensors
       and return torch tensors.
       """
    def wrapped(image: torch.Tensor, severity: int = 1, **kwargs) -> torch.Tensor:
        pil = tensor_to_pil(image)
        out_np_or_pil = fn(pil, severity=severity, **kwargs)
        out_t = pil_or_np_to_tensor(out_np_or_pil)
        return out_t
    return wrapped

# Build the backend mapping
HENDRYCKS_FUNCTIONS = {
    name: wrap_hendrycks_function(f)
    for name, f in corruption_dict.items()
}



