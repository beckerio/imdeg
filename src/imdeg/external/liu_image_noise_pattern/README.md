# Liu External Backend Files

This directory is reserved for user-supplied files from the Liu IJCV 2024
reference implementation:

- Paper: "Benchmarking Object Detection Robustness against Real-World Corruptions"
- Project page: `https://sites.google.com/view/real-worldbenchmark/real-world-benchmarking`
- Reference repository: `https://github.com/Jackie-Bai888/image-noise-pattern`

## Why these files are not bundled

`imdeg` does not redistribute the original Liu source files because the upstream
repository does not provide clear redistribution license information.

For that reason:

- `imdeg` includes wrapper and compatibility code for the Liu backend
- the original Liu files must be added by the user
- those upstream files should not be committed into another public repository
  unless their licensing status is clarified

## Expected location

Place the upstream `.py` files directly in:

```text
src/imdeg/external/liu_image_noise_pattern/
```

The wrapper code loads files from this folder at runtime.

## If the folder is missing or incomplete

Only the Liu backend is affected. Other package functionality, including
Hendrycks and ARNIQA-related degradations, can still be used.
