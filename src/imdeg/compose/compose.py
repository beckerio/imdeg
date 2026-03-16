"""
Dataset composition utilities for imgdeg.

Implements:
- compose_round_robin
- compose_full_factorial

Both functions accept either:
- a list of image paths
- or a directory containing images
"""

import os
import itertools
from typing import List, Tuple, Optional, Callable, Sequence, Dict

import torch
from PIL import Image
from torchvision import transforms

from imdeg.apply import apply_degradation
from imdeg.utils.image_io import load_images_from_folder, ensure_dir, load_imgpaths_from_folder
from imdeg.utils.image_conversion import tensor_to_numpy_uint8


# ------------------------------------------------------------
# Utility: load images
# ------------------------------------------------------------

def _load_images(input_src: str | List[str]) -> List[str]:
    """Return a list of image file paths."""
    if isinstance(input_src, list):
        return input_src
    else:
        return load_imgpaths_from_folder(input_src)



# ------------------------------------------------------------
# Round-robin dataset generation (severity-first structure)
# ------------------------------------------------------------

def compose_round_robin(
    input_src: str | List[str],
    out_dir: str,
    paper: str,
    terms: List[str],
    severities: Sequence[int] = (1, 2, 3, 4, 5),
    output_mode: str = "subdirs",   # "subdirs" or "filename"
):
    """
    round-robin over *terms* (corruption types),
    with severity-first directory structure:

        out_dir/<severity>/<term>/<image_name>

    - Deterministic mapping:
        term_idx = image_index % len(terms)
        term     = terms[term_idx]

    - For all severity levels, **each image keeps its assigned term**.

    Args
    ----
    input_src:
        Folder path or list of image paths.
    out_dir:
        Output directory where degraded images will be written.
    paper:
        Taxonomy paper key, e.g. "Hendrycks_ICLR_2018" or "Agnolucci_WACV_2024".
    terms:
        List of corruption names (backend or taxonomy terms) to cycle over.
    severities:
        Sequence of severity indices (1..5).
    output_mode:
        - "subdirs": out_dir/<sev>/<term>/<fname>
        - "filename": out_dir/<sev>/<fname>_<term>.jpg
    """

    ensure_dir(out_dir)

    # IMPORTANT: sorted for deterministic mapping: image_index -> term
    img_paths = sorted(_load_images(input_src))

    to_tensor = transforms.ToTensor()
    num_terms = len(terms)

    if num_terms == 0:
        raise ValueError("compose_round_robin_terms: 'terms' list must not be empty.")

    print(f"[imgdeg] Terms (round-robin): {terms}")
    print(f"[imgdeg] Severities: {list(severities)}")
    print(f"[imgdeg] Mapping: img[i] -> terms[i % {num_terms}]")

    # === MAIN LOOP ===
    for sev in severities:
        print(f"[imgdeg] Generating severity {sev}...")

        for idx, img_path in enumerate(img_paths):

            img = Image.open(img_path).convert("RGB")
            img_t = to_tensor(img)

            # Round-robin: deterministic term assignment
            term_idx = idx % num_terms
            term = terms[term_idx]

            degraded = apply_degradation(
                image=img_t,
                paper=paper,
                term=term,
                severity=sev,
                mode="original"
            )

            arr = tensor_to_numpy_uint8(degraded)
            fname = os.path.basename(img_path)

            if output_mode == "subdirs":
                # severity-first structure: <out>/<sev>/<term>/<image>
                save_dir = os.path.join(out_dir, str(sev), term)
                ensure_dir(save_dir)
                out_path = os.path.join(save_dir, fname)

            else:  # filename mode
                fname_noext, ext = os.path.splitext(fname)
                out_path = os.path.join(out_dir, str(sev),
                                        f"{fname_noext}_{term}{ext}")
                ensure_dir(os.path.dirname(out_path))

            Image.fromarray(arr).save(out_path)

    print(f"[imgdeg] Round-robin (terms, severity-first) dataset written to: {out_dir}")



## ------------------------------------------------------------
# Full factorial dataset generation
# ------------------------------------------------------------

def compose_full_factorial(
    input_src: str | List[str],
    out_dir: str,
    paper: str,
    terms: List[str],
    chain_lengths: Sequence[int] = (1,2),
    severities: Sequence[int] = (3,),   # default severity 3 like Michaelis
    output_mode: str = "subdirs"
):
    """
    Apply L-step sequential chains of corruptions for all L in chain_lengths.

    Example:
        chain_lengths = [1,2]
        terms = ["gaussian_noise","motion_blur"]

    Chains generated:
        L=1:  [gaussian_noise], [motion_blur]
        L=2:  [gaussian_noise -> gaussian_noise],
              [gaussian_noise -> motion_blur],
              [motion_blur -> gaussian_noise],
              [motion_blur -> motion_blur]
    """

    ensure_dir(out_dir)
    img_paths = _load_images(input_src)

    # All possible corruption chains for each L
    chains = []
    for L in chain_lengths:
        for combo in itertools.product(terms, repeat=L):
            chains.append(combo)

    to_tensor = transforms.ToTensor()

    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        img_t = to_tensor(img)

        for chain in chains:
            for sev in severities:

                x = img_t.clone()
                # apply sequentially
                for term in chain:
                    x = apply_degradation(
                        image=x,
                        paper=paper,
                        term=term,
                        severity=sev,
                        mode="original"
                    )

                arr = tensor_to_numpy_uint8(x)

                chain_tag = "-".join(chain)

                if output_mode == "subdirs":
                    save_dir = os.path.join(out_dir, chain_tag, str(sev))
                    ensure_dir(save_dir)
                    fname = os.path.basename(img_path)
                    out_path = os.path.join(save_dir, fname)
                else:
                    fname, ext = os.path.splitext(os.path.basename(img_path))
                    ensure_dir(out_dir)
                    out_path = os.path.join(
                        out_dir,
                        f"{fname}_{chain_tag}_{sev}{ext}"
                    )

                Image.fromarray(arr).save(out_path)

    print(f"[imgdeg] Full-factorial dataset written to: {out_dir}")


# ------------------------------------------------------------
# Cartesian dataset generation (severity-first structure)
# ------------------------------------------------------------

def compose_cartesian(
    input_src: str | List[str],
    out_dir: str,
    paper: str,
    terms: List[str],
    severities: Sequence[int] = (1, 2, 3, 4, 5),
    output_mode: str = "subdirs",   # "subdirs" or "filename"
):
    """
    Cartesian product over images, *terms* (corruption types), and severities,
    with severity-first directory structure:

        out_dir/<severity>/<term>/<image_name>

    This corresponds to the ImageNet-C / multi-version protocol:
    every image appears once for every (term, severity) pair.

    Args
    ----
    input_src:
        Folder path or list of image paths.
    out_dir:
        Output directory where degraded images will be written.
    paper:
        Taxonomy paper key, e.g. "Hendrycks_ICLR_2018" or "Agnolucci_WACV_2024".
    terms:
        List of corruption names (backend or taxonomy terms).
    severities:
        Sequence of severity indices (1..5).
    output_mode:
        - "subdirs": out_dir/<sev>/<term>/<fname>
        - "filename": out_dir/<sev>/<fname>_<term>.jpg
    """

    ensure_dir(out_dir)

    # Sorted for deterministic order, although Cartesian does not depend on index
    img_paths = sorted(_load_images(input_src))

    to_tensor = transforms.ToTensor()
    num_terms = len(terms)

    if num_terms == 0:
        raise ValueError("compose_cartesian: 'terms' list must not be empty.")

    print(f"[imgdeg] Terms (cartesian): {terms}")
    print(f"[imgdeg] Severities: {list(severities)}")
    print(f"[imgdeg] Cartesian protocol: all (image, term, severity) combinations.")

    # === MAIN LOOP ===
    for sev in severities:
        print(f"[imgdeg] Generating severity {sev} (cartesian)...")

        for term in terms:
            print(f"[imgdeg]  - applying term '{term}'...")

            for img_path in img_paths:
                img = Image.open(img_path).convert("RGB")
                img_t = to_tensor(img)

                degraded = apply_degradation(
                    image=img_t,
                    paper=paper,
                    term=term,
                    severity=sev,
                    mode="original"
                )

                arr = tensor_to_numpy_uint8(degraded)
                fname = os.path.basename(img_path)

                if output_mode == "subdirs":
                    # severity-first structure: <out>/<sev>/<term>/<image>
                    save_dir = os.path.join(out_dir, str(sev), term)
                    ensure_dir(save_dir)
                    out_path = os.path.join(save_dir, fname)

                else:  # filename mode
                    fname_noext, ext = os.path.splitext(fname)
                    save_dir = os.path.join(out_dir, str(sev))
                    ensure_dir(save_dir)
                    out_path = os.path.join(save_dir,
                                             f"{fname_noext}_{term}{ext}")

                Image.fromarray(arr).save(out_path)

    print(f"[imgdeg] Cartesian (ImageNet-C style) dataset written to: {out_dir}")