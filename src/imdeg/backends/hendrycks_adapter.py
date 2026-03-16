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



