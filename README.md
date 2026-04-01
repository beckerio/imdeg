# imdeg

`imdeg` is a Python library for organizing and applying image degradations used in robustness evaluation.
It combines a causal taxonomy, paper-specific mappings, and backend adapters so degradations from different benchmarks can be applied and analyzed through one API.

## Features

- Unified taxonomy across synthetic corruptions, IQA distortions, and real-world camera or ISP failures
- Paper-aware registry for benchmark-native degradation names
- Severity calibration helpers for comparing native schedules without flattening them into a fake universal scale
- Torch-friendly application APIs for image degradation pipelines

## Installation

Install the package in editable mode during development:

```bash
pip install -e .[dev]
```

For visualization helpers:

```bash
pip install -e .[viz]
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

## Project Layout

- `src/imdeg/taxonomy.py`: canonical taxonomy and paper mappings
- `src/imdeg/registry.py`: backend registry and alias resolution
- `src/imdeg/apply.py`: high-level degradation application helpers
- `src/imdeg/severity.py`: severity calibration and mapping tools
- `src/imdeg/backends/`: maintained backend adapters and included assets
- `src/imdeg/external/`: optional user-supplied external implementations

## Licensing and Third-Party Code

This project is released under the Apache License 2.0.

Some integrations refer to upstream implementations with unclear redistribution terms. Those sources are not republished as first-class bundled code; instead, the package provides wrapper and compatibility logic where appropriate. See `THIRD_PARTY_NOTICES.md` for attribution and license details.

## Citation

If you use this package in academic work, cite the associated paper when its publication details are finalized:

```bibtex
@article{Becker_arxiv_2026,
  title={A Causally Grounded Taxonomy for Image Degradation Robustness Evaluation},
  author={Becker, Stefan and Weiss, Simon and H{\"u}bner, Wolfgang and Arens, Michael},
  journal={arXiv preprint arXiv:XXXX.XXXX},
  year={2025}
}
```

