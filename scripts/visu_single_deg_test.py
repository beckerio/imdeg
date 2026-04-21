import sys
from argparse import ArgumentParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlretrieve

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import cv2
import os
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_tensor, to_pil_image
#from imgdeg import taxonomy, apply_degradation
import matplotlib.pyplot as plt
import numpy as np

#from vca_utils.filehandling import load_files_folder, img_formats, load_img_from_folder, file_to_img

from imdeg import (
    taxonomy,
    map_paper_term,
    list_paper_types,
    apply_degradation,
    degrade_image_with_annotation,
)


#from imgdeg.apply import apply_degradation
from imdeg.metrics import distortion_metric
#from imgdeg.calibration import invert_poly_to_native, native_for_canonical_level

def to_imshow_array(img: torch.Tensor):
    """
    Convert a torch image tensor to a numpy array suitable for plt.imshow.
    Accepts:

      - (3,H,W) or (1,H,W) or (H,W,3) or (H,W) or (1,3,H,W)

    Returns:

      - (H,W,3) float32 in [0,1] or (H,W) for grayscale

    """
    if torch.is_tensor(img):
        x = img
        # remove batch if present
        if x.ndim == 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        # CHW -> HWC
        if x.ndim == 3 and x.shape[0] in (1, 3):
            x = x.permute(1, 2, 0).contiguous()
        # (H,W,1) -> (H,W)
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        # ensure float in [0,1]
        if x.dtype.is_floating_point:
            x = x.clamp(0, 1)
        else:
            # uint8 -> float [0,1]
            x = x.to(torch.float32) / 255.0
        return x.detach().cpu().numpy()
    else:
        # numpy path (assume already HWC or HW)
        return img

def safe_name(s: str) -> str:
    # keep only alnum, dash, underscore; replace others with underscore
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(s))


DEFAULT_SAMPLE_IMAGE_URLS = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png",
    "http://images.cocodataset.org/val2017/000000039769.jpg",
)
DEFAULT_COCO_IMAGE_URL = DEFAULT_SAMPLE_IMAGE_URLS[0]
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "samples"


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Visualize imdeg degradations on a single input image.")
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Optional local image path. If omitted, a default COCO sample image is downloaded.",
    )
    parser.add_argument(
        "--image-url",
        default=DEFAULT_COCO_IMAGE_URL,
        help=(
            "Remote image URL used when --image is not provided. "
            "Defaults to a Hugging Face-hosted sample image."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory used to cache downloaded sample images.",
    )
    return parser.parse_args()


def resolve_input_image(image_path: Path | None, image_url: str, cache_dir: Path) -> Path:
    if image_path is not None:
        if not image_path.exists():
            raise FileNotFoundError(f"Image path does not exist: {image_path}")
        return image_path

    cache_dir.mkdir(parents=True, exist_ok=True)

    if image_url == DEFAULT_COCO_IMAGE_URL:
        candidate_urls = DEFAULT_SAMPLE_IMAGE_URLS
    else:
        candidate_urls = (image_url,)

    download_errors: list[str] = []
    for candidate_url in candidate_urls:
        target_name = Path(urlparse(candidate_url).path).name or "sample_image"
        target_path = cache_dir / target_name

        if target_path.exists():
            return target_path

        try:
            print(f"Downloading default sample image from {candidate_url}")
            urlretrieve(candidate_url, target_path)
            return target_path
        except (HTTPError, URLError, OSError) as exc:
            target_path.unlink(missing_ok=True)
            download_errors.append(f"{candidate_url}: {exc}")

    joined_errors = "\n".join(download_errors)
    raise RuntimeError(f"Unable to download a sample image. Tried:\n{joined_errors}")


if __name__ == '__main__':
    args = parse_args()
    PYTHON_VERSION = sys.version_info[0]
    print("***********************************************************************************************************")
    print(f"torch version {torch.__version__}")
    print(f"python version {sys.version_info[0]}")
    print("***********************************************************************************************************")

    minga = "/mnt/Data-2TB/"
    kalle = "/media/ste82041/DATA-2TB/"
    gs007 = "/mnt/15TB-NVME/ste82041/"

    data_folder = minga


    #save_dir_img = "/media/ste82041/DATA-2TB/paperforge/2026/SAD_DegMani/figure/single_deg_examples"
    #save_dir_img = "mnt/Data-512GB/paperforge/2026/TaxonomyDegradations/figure/cascaded_deg"
    save_dir_img = None
    #None


    paper_name = "Agnolucci_WACV_2024"
    #paper_name = "Liu_IJCV_2024"
    #paper_name = "Hendrycks_ICLR_2019"

    #Inspect  types in the taxonomy
    types_arn = list_paper_types(paper_name)
    #types_arn = list_paper_types("Liu_IJCV_2024")
    #types_arn = list_paper_types("Hendrycks_ICLR_2019")
    #for row in types_arn:
    #    print(row["term"], "->", row["group_id"], row["group_name"])
    #input()

    # 2) online: apply canonical severity 3
    input_folder = "../data"
    input_image_path = resolve_input_image(
        image_path=args.image,
        image_url=args.image_url,
        cache_dir=args.cache_dir,
    )
    img_cv_rgb = Image.open(input_image_path).convert("RGB")

    img_t = to_tensor(img_cv_rgb)

    severity_level = 5
    severity = [1, 2, 3, 4, 5]
    severity = [5]
    imgs_deg = []
    for level in severity:
        print(F" severity {level}")
        for index, row in enumerate(types_arn):
            print(index+1,row["term"], "->", row["group_id"], row["group_name"])

            #  ("Liu_IJCV_2024", "ccd_sensor_damage"): "ccd_sensor_damage",
            img_deg = apply_degradation(
                image=img_t.clone(),
                paper=paper_name,
                term=row["term"],
                severity=level,
                mode="original",
            )

            img_deg_vius = to_imshow_array(img_deg)
            img_u8_bgr = (np.clip(img_deg_vius, 0, 1) * 255).round().astype(np.uint8)
            img_u8_bgr = cv2.cvtColor(img_u8_bgr, cv2.COLOR_RGB2BGR)
            if save_dir_img is not None:
                paper_safe = safe_name(paper_name)
                term_safe = safe_name(row['term'])
                out_dir = os.path.join(save_dir_img, paper_safe, term_safe)
                # If you also want a folder per severity, uncomment:
                # out_dir = os.path.join(save_dir_img, paper_safe, term_safe, f"severity_{level}")
                os.makedirs(out_dir, exist_ok=True)

                save_img_name = os.path.join(
                    out_dir,
                    f"{index + 1:04d}_{paper_safe}_{term_safe}_degraded_img_{level}.jpg"
                )
                ok = cv2.imwrite(save_img_name, img_u8_bgr)
                if not ok:
                    print(f"Failed to save: {save_img_name}")
                else:
                    print(f"Saved: {save_img_name}")
            plt.figure()
            #plt.axis([0, width_org, 0, height_org])
            # plt.xlabel('$x$ / pixels')
            # plt.ylabel('$y$ / pixels')
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.gca().invert_yaxis()
            plt.imshow(img_deg_vius)
            plt.show()
            plt.close()

            imgs_deg.append(img_deg_vius)


            #plt.savefig(save_img_name, dpi=300)
        # plt.savefig("F:\\Develop\\Py_src\\out\\duke_test\\"+counter_str + ".png", dpi=300)
        # cv2.imwrite(f"{opt.out_dir}/frame_" + counter_str + ".jpg", frame)

