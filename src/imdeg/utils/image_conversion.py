import torch
import numpy as np
from PIL import Image
from typing import Optional, Literal
import cv2

def tensor_to_numpy_uint8(x: torch.Tensor) -> np.ndarray:
    """
    x: torch tensor in [0,1] or [0,255], shape CxHxW or HxWxC
    Return: uint8 numpy array HxWxC (RGB)
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()

        if x.ndim == 3 and x.shape[0] in [1,3]:
            # convert CHW -> HWC
            x = x.permute(1,2,0)

        x = x.numpy()

    # normalize to [0,255] if needed
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255)
        if x.max() <= 1.0:
            x = x * 255
        x = x.astype(np.uint8)

    return x

def numpy_uint8_to_tensor(x: np.ndarray) -> torch.Tensor:
    """
    Convert HxWxC uint8 numpy array back to torch float tensor in [0,1].
    """
    if x.dtype != np.uint8:
        raise ValueError("Expected uint8 numpy array.")

    t = torch.from_numpy(x).float() / 255.0

    if t.ndim == 3:
        t = t.permute(2,0,1)  # HWC -> CHW

    return t

def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim == 4 and image.shape[0] == 1:
        image = image.squeeze(0)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    img_np = image.detach().cpu().clamp(0, 1).numpy()
    img_uint8 = (img_np * 255.0).round().astype(np.uint8)
    return Image.fromarray(img_uint8)

def pil_or_np_to_tensor(img) -> torch.Tensor:
    if isinstance(img, Image.Image):
        img = np.array(img)
    img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = img[:, :, None]
    img = torch.from_numpy(img).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
    return img


def tensor_to_bgr_uint8(img: torch.Tensor) -> np.ndarray:
    """
    Convert a torch tensor (3,H,W) RGB image to an OpenCV uint8 BGR image (H,W,3).
    Accepts [0,1] or [0,255] float or byte tensors.
    """
    if img.ndim != 3:
        raise ValueError(f"[liu_backend] Expected 3D tensor (C,H,W), got shape {img.shape}")
    if img.size(0) != 3:
        raise ValueError(f"[liu_backend] Expected RGB tensor with 3 channels, got {img.size(0)}")

    x = img.detach().cpu().float()
    # assume [0,1] if small
    if x.max() <= 1.01:
        x = x * 255.0

    x = x.clamp(0, 255).byte()  # uint8
    # C,H,W -> H,W,C
    x_hw_c = x.permute(1, 2, 0).numpy()
    # RGB -> BGR
    bgr = x_hw_c[:, :, ::-1].copy()
    return bgr


def bgr_uint8_to_tensor(img: np.ndarray, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convert OpenCV uint8 BGR image (H,W,3) to a torch RGB float tensor (3,H,W) in [0,1].
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"[liu_backend] Expected BGR image of shape (H,W,3), got {img.shape}")

    rgb = img[:, :, ::-1].astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1)  # C,H,W
    if device is not None:
        t = t.to(device)
    return t.clamp(0.0, 1.0)

def pil_to_np_rgb(img_pil):
    # PIL Image (RGB) -> NumPy uint8 HxWx3 (RGB)
    return np.array(img_pil)

def np_rgb_to_pil(arr):
    # NumPy HxWx3 (RGB) -> PIL Image
    return Image.fromarray(arr)

def rgb_to_bgr(arr_rgb):
    # to use cv2.imshow/write and require BGR
    return cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)

def RGB2Bayer_RG(img: np.ndarray) -> np.ndarray:
    """
    Convert BGR uint8 image to single-channel Bayer mosaic in RGGB layout.

    Input:
        img: H x W x 3, uint8, assumed OpenCV/BGR order

    Output:
        H x W, uint8 Bayer image
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape {img.shape}")

    h, w, _ = img.shape
    out = np.empty((h, w), dtype=img.dtype)

    # OpenCV uses BGR
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    # RGGB pattern:
    # R G
    # G B
    out[0::2, 0::2] = r[0::2, 0::2]
    out[0::2, 1::2] = g[0::2, 1::2]
    out[1::2, 0::2] = g[1::2, 0::2]
    out[1::2, 1::2] = b[1::2, 1::2]

    return out



