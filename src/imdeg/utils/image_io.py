from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union
import os
import torch
from PIL import Image
from torchvision import transforms


def load_images_from_folder(
    folder: Union[str, Path],
    pattern: str = "*",
) -> Tuple[List[torch.Tensor], List[Path]]:
    """
    Load all images from a folder as tensors in [0,1].

    Supported extensions: .png, .jpg, .jpeg, .bmp, .tif, .tiff (case-insensitive).

    Returns:
        images: list of (C,H,W) float32 tensors in [0,1]
        paths:  list of Path objects with same length
    """
    folder = Path(folder)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    all_paths = sorted(folder.glob(pattern))
    img_paths = [p for p in all_paths if p.suffix.lower() in exts]

    tfs = transforms.ToTensor()
    images: List[torch.Tensor] = []

    for p in img_paths:
        img = Image.open(p).convert("RGB")
        images.append(tfs(img))

    return images, img_paths

def load_imgpaths_from_folder(
    folder: Union[str, Path],
    pattern: str = "*",
) -> List[Path]:
    """
    Load all images path from a folder .

    Supported extensions: .png, .jpg, .jpeg, .bmp, .tif, .tiff (case-insensitive).

    Returns:
        paths:  list of Path objects with same length
    """
    folder = Path(folder)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    all_paths = sorted(folder.glob(pattern))
    img_paths = [p for p in all_paths if p.suffix.lower() in exts]

    return  img_paths


def save_tensor_as_image(
    image: torch.Tensor,
    path: Union[str, Path],
) -> None:
    """
    Save a (C,H,W) image tensor in [0,1] as a PNG.

    - Automatically creates parent directories.
    - Converts CHW -> HWC, scales to 0..255, and writes using Pillow.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    img = image.detach().cpu().clamp(0, 1)

    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.permute(1, 2, 0)  # CHW -> HWC

    img = (img * 255.0).round().byte().numpy()
    im = Image.fromarray(img)
    im.save(path)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
