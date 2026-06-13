# imdeg

`imdeg` is a Python library for organizing and applying image degradations used in robustness evaluation.
It combines a causal taxonomy, paper-specific mappings, and backend adapters so degradations from different benchmarks can be applied and analyzed through one API.

## Features

- Unified taxonomy across synthetic corruptions, IQA distortions, and real-world camera or ISP failures
- Paper-aware registry for benchmark-native degradation names
- Severity calibration helpers for comparing native schedules without flattening them into a fake universal scale
- Torch-friendly application APIs for image degradation pipelines

## Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/beckerio/imdeg.git
cd imdeg
pip install -e .
```

Some backends depend on system libraries in addition to Python packages:

- `Wand` requires ImageMagick to be available on the system
- OpenCV-based backends require a working `opencv-python` installation

## Quick Start

```python
import torch
from imdeg import apply_degradation, list_paper_types, map_paper_term

image = torch.rand(3, 224, 224)

info = map_paper_term("Hendrycks_ICLR_2019", "gaussian_noise")
degraded = apply_degradation(
    image=image,
    paper="Hendrycks_ICLR_2019",
    term="gaussian_noise",
    severity=3,
)

print(info["group"]["id"])
print(list_paper_types("Hendrycks_ICLR_2019")[:3])
print(degraded.shape)
```

## Canonical Severity Example

Canonical mode requires a calibration record that contains measured native
distortion strengths and derived canonical strengths. A runnable example is
provided in [scripts/canonical_quick_start.py](/home/ste82041/python_src/imdeg/scripts/canonical_quick_start.py).

```python
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

from imdeg import apply_degradation
from imdeg.calibration import calibrate_distortion

x = to_tensor(Image.open("example.jpg").convert("RGB"))
calibration = calibrate_distortion(
  images=[x],
  paper="Agnolucci_WACV_2024",
  term="gaublur",
)
canonical_strengths = calibration["canonical_strengths"]

y = apply_degradation(
  image=x,
  paper="Agnolucci_WACV_2024",
  term="gaublur",
  severity=3,
  mode="canonical",
  calibration=calibration,
  canonical_strengths=canonical_strengths,
)
to_pil_image(y).save("gaublur_canonical_k3.png")
```

## Project Layout

- `src/imdeg/taxonomy.py`: canonical taxonomy and paper mappings
- `src/imdeg/registry.py`: backend registry and alias resolution
- `src/imdeg/apply.py`: high-level degradation application helpers
- `src/imdeg/severity.py`: severity calibration and mapping tools
- `src/imdeg/backends/`: maintained backend adapters and included assets
- `src/imdeg/external/`: optional user-supplied external implementations

## External Liu Backend Files

The Liu IJCV 2024 integration is special: the original upstream implementation is
not redistributed with `imdeg`.

Users must place the required upstream Liu files under:

```text
src/imdeg/external/liu_image_noise_pattern/
```

This is required because the upstream Liu repository does not provide clear
redistribution license information, so `imdeg` only ships wrapper and
compatibility code for that backend.

If the external Liu files are not installed:

- Liu degradations such as `Liu_IJCV_2024` terms will fail at runtime
- Hendrycks and ARNIQA-based functionality continues to work

See [src/imdeg/external/liu_image_noise_pattern/README.md](/home/ste82041/python_src/imdeg/src/imdeg/external/liu_image_noise_pattern/README.md) for setup details and the upstream reference.

## Licensing and Third-Party Code

This project is released under the Apache License 2.0.

Some integrations refer to upstream implementations with unclear redistribution terms. Those sources are not republished as first-class bundled code; instead, the package provides wrapper and compatibility logic where appropriate. See `THIRD_PARTY_NOTICES.md` for attribution and license details.

## Citation

If you use this package in academic work, please consider to cite the following paper:
```bibtex
@misc{becker2026causallygroundedtaxonomyimage,
      title={A Causally Grounded Taxonomy for Image Degradation Robustness Evaluation}, 
      author={Stefan Becker and Simon Weiss and Wolfgang Hübner and Michael Arens},
      year={2026},
      eprint={2605.15906},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.15906}, 
}
```
