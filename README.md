# imdeg

**imdeg** is a Python library for working with **image degradations** in the context of robustness evaluation.  
It provides a unified representation of degradations originating from different research traditions and supports their consistent application and analysis.

The library is built around a **causally grounded taxonomy** of image degradations and a lightweight **severity measurement layer**. Its primary goal is to make existing degradation definitions easier to organize, interpret, and compare—without modifying their original implementations.

## Scope

imdeg integrates degradations from three commonly used sources:

- **Synthetic corruption benchmarks**  
  (e.g. ImageNet-C / ImageNet-P, COCO-C-style pipelines)
- **Perceptual image quality assessment (IQA) distortions**  
  (e.g. KADID-10k / ARNIQA-style degradations)
- **Real-camera, ISP, and system-level failure models**  
  (e.g. Liu et al., IJCV 2024)

These degradation families are treated as complementary rather than interchangeable and are preserved in their original form.

The library is intended to support:
- Robustness evaluation for **classification and detection**
- Cross-benchmark analysis with **explicit severity interpretation**
- Construction of degraded datasets (e.g. **COCO-Degradation**)
- Analysis of degradations with respect to **causal origin** and **perceptual effect**

---

## Taxonomy

imdeg uses a compact taxonomy that organizes degradations along two orthogonal axes:

- **Causal source (cause)**  
  - Environment (E)  
  - Sensor / Optics (S)  
  - ISP / Rendering / Codec (R)  
  - System / Transfer (T)

- **Perceptual effect (effect)**  
  - Noise (N)  
  - Blur (B)  
  - Weather / Medium (WX)  
  - Compression / Quantization (CP)  
  - Color / White Balance (CL)  
  - Illumination / Exposure (IL)  
  - Geometry / Spatial (GD)  
  - Resolution / Sampling (RZ)  
  - Occlusion / Obstruction (OC)  
  - Texture / Sharpness / Contrast (TX)  
  - Temporal / Video (TV)

The combination of these axes yields **12 canonical degradation groups (G1–G12)**.  
Each degradation function is assigned a dominant cause and effect for organizational purposes, acknowledging that real-world degradations may span multiple stages.

---

## Severity Measurement and Harmonization

Different benchmarks define severity levels using backend-specific parameters that are not directly comparable.  
imdeg addresses this through a **severity measurement layer**:

- Native severity schedules are **preserved**
- For each degradation and severity level, distortion strength can be **measured** using:
  - PSNR
  - SSIM / (1-SSIM)
  - LPIPS
  

These measurements:
- Make severity semantics explicit
- Enable cross-degradation and cross-benchmark comparison
- Do not impose a universal severity scale

### Optional Canonical Severity Mapping

For analysis purposes, measured distortion strengths can be mapped to a shared, coarse severity axis:

- Polynomial fits relate native severity to metric space
- Canonical levels are defined in metric space, not parameter space
- This mapping is intended for **interpretation**, not dataset generation

### Optional Severity Extrapolation

imdeg also supports controlled extrapolation beyond native severity ranges:

- Canonical levels may be extended linearly in metric space
- Native parameters are inferred via calibrated inversion
- This is useful for probing detector failure regimes, but is optional

---

## Library Structure

The codebase is organized to separate definitions, mappings, and application logic:




- 🧱 **Modular architecture**:
  - `taxonomy.py` – canonical taxonomy & paper mappings,
  - `backends/` – backend wrappers and maintained corruption implementations.
            Implementations with permissive licenses (e.g. Apache-2.0)
            are included directly here.
  - `external/` – optional user-supplied implementations for dependencies with
            unclear or restrictive licensing (e.g. liu_image_noisepattern https://github.com/Jackie-Bai888/image-noise-pattern)
  - `registry.py` – connects taxonomy terms to backend functions,
  - `severity.py` – polynomial calibration tools,
  - `apply.py` – high-level API to apply degradations,
  - `datasets/` – dataset-specific integrations (COCO, ImageNet, etc.).

---

### Python

TODO

```python
# provide python examples


```

## Installation

Clone the repository and install in editable mode:
```
git clone https://github.com/yourname/imdeg.git
cd imdeg
pip install -e .
```

## Citation

If you use our code or the imagecorruptions package, please consider citing:
```
@article{Becker_arxiv_2026,
  title={A Causally Grounded Taxonomy for Image Degradation Robustness Evaluation},
  author={Becker, Stefan and Weiss, Simon and H{\"u}bner, Wolfgang and Arens, Michael},
  journal={arXiv preprint arXiv:XXXX.XXXX},
  year={2025}
}
```

### Acknowledgements

- **Hendrycks and Dietterich**: [Code](https://github.com/hendrycks/robustness), [Paper](https://arxiv.org/abs/1903.12261)
- **Michaelis et al.**: [Code](https://github.com/bethgelab/robust-detection-benchmark), [Paper](https://arxiv.org/abs/1907.07484)
- **Liu et al.**: [Code](https://sites.google.com/view/real-worldbenchmark/real-world-benchmarkin), [Paper](https://link.springer.com/article/10.1007/s11263-024-02096-6)
- **Oksuz et al.**: [Code](https://github.com/asharakeh/probdet), [Paper](https://arxiv.org/abs/2101.05036)
- **Agnolucci et al.**: [Code](https://github.com/miccunifi/ARNIQA), [Paper](https://arxiv.org/abs/2310.14918)
- add ..
 

## License

This project is released under the Apache License 2.0.

It builds on and interoperates with several prior open-source benchmarks and
datasets. See `THIRD_PARTY_NOTICES.md` for detailed attributions and licensing
information.


