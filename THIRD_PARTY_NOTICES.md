# Third-Party Notices

This project, **imgdeg**, incorporates or re-implements functionality inspired by
prior open-source projects and published benchmarks. The following attributions
and license notices apply.

---

## Hendrycks & Dietterich – ImageNet-C / ImageNet-P

- Repository: https://github.com/hendrycks/robustness
- Paper: Hendrycks and Dietterich, *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations*, ICLR 2019
- License: Apache License, Version 2.0

Portions of the degradation definitions and severity schedules are based on the
publicly released ImageNet-C/ImageNet-P reference implementations.
All modifications and integrations are provided under the Apache-2.0 license.

---

## Michaelis et al. – Robust Detection Benchmark (COCO-C)

- Repository: https://github.com/bethgelab/robust-detection-benchmark
- Paper: Michaelis et al., *Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming*, NeurIPS Workshops 2019
- License: MIT License

Concepts and degradation definitions are used for compatibility with COCO-C-style
benchmarks. No original code is redistributed verbatim.

---

## Öksüz et al. – Probabilistic Object Detection

- Repository: https://github.com/asharakeh/probdet
- Paper: Öksüz et al., *Towards Building Self-Aware Object Detectors via Reliable Uncertainty Quantification and Calibration*, CVPR 2023
- License: Apache License, Version 2.0

The round-robin dataset construction protocol is re-implemented in a backend-agnostic
manner.

---

## Agnolucci et al. – ARNIQA / IQA-Based Degradations

- Repository: https://github.com/miccunifi/ARNIQA
- Paper: Agnolucci et al., *ARNIQA: Learning Distortion Manifold for Image Quality Assessment*, WACV 2024
- License: Apache License, Version 2.0

IQA-motivated degradation operators and severity concepts are re-implemented for
taxonomy-aligned evaluation.

---

## Liu et al. – Real-World Camera Degradation Benchmark

- Project page: https://sites.google.com/view/real-worldbenchmark/real-world-benchmarking
- Paper: Liu et al., *Benchmarking Object Detection Robustness against Real-World Corruptions*, IJCV 2024
- Dataset License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Code License: Not specified at the time of writing

The **imdeg** library does not include unlicensed source code from this repository.
Degradation functions associated with this work are implemented independently
based on descriptions in the published paper. Dataset usage follows the terms
of the CC BY 4.0 license and requires appropriate attribution.

---

## Additional Notes

- All third-party names, trademarks, and datasets remain the property of their
  respective owners.
- This project is not affiliated with or endorsed by the above authors unless
  explicitly stated.
