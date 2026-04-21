from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from imdeg import apply_degradation
from imdeg.calibration import calibrate_distortion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demonstrate canonical-severity degradation with imdeg."
    )
    parser.add_argument("image", type=Path, help="Input RGB image to degrade.")
    parser.add_argument(
        "--calibration-image",
        dest="calibration_images",
        action="append",
        type=Path,
        help=(
            "Reference image used to derive canonical strengths. "
            "Pass this option multiple times to use several images. "
            "If omitted, the input image is reused for calibration."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gaublur_canonical_k3.png"),
        help="Output path for the degraded image.",
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=3,
        choices=range(1, 6),
        help="Canonical severity level to apply.",
    )
    return parser.parse_args()


def load_rgb_tensor(path: Path):
    return to_tensor(Image.open(path).convert("RGB"))


def main() -> None:
    args = parse_args()

    image = load_rgb_tensor(args.image)
    calibration_paths = args.calibration_images or [args.image]
    calibration_images = [load_rgb_tensor(path) for path in calibration_paths]

    calibration = calibrate_distortion(
        images=calibration_images,
        paper="Agnolucci_WACV_2024",
        term="gaublur",
        metric="1-ssim",
    )
    canonical_strengths = calibration["canonical_strengths"]

    degraded = apply_degradation(
        image=image,
        paper="Agnolucci_WACV_2024",
        term="gaublur",
        severity=args.severity,
        mode="canonical",
        calibration=calibration,
        canonical_strengths=canonical_strengths,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(degraded).save(args.output)

    print(f"Saved degraded image to {args.output}")
    print(f"Canonical strengths: {canonical_strengths}")


if __name__ == "__main__":
    main()