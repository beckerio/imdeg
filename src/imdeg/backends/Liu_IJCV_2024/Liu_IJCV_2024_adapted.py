"""
@Article{Liu_IJCV_2024,
  author   = {Liu, Jiawei and Wang, Zhijie and Ma, Lei and Fang, Chunrong and Bai, Tongtong and Zhang, Xufan and Liu, Jia and Chen, Zhenyu},
  journal  = {International Journal of Computer Vision},
  title    = {Benchmarking Object Detection Robustness against Real-World Corruptions},
  year     = {2024},
  issn     = {1573-1405},
  month    = {Oct},
  number   = {10},
  pages    = {4398-4416},
  volume   = {132},
  abstract = {With the rapid recent development, deep learning based object detection techniques have been applied to various real-world software systems, especially in safety-critical applications like autonomous driving. However, few studies are conducted to systematically investigate the robustness of state-of-the-art object detection techniques against real-world image corruptions and yet few benchmarks of object detection methods in terms of robustness are publicly available. To bridge this gap, we initiate to create a public benchmark of COCO-C and BDD100K-C, composed of sixteen real-world corruptions according to the real damages in camera sensors and image pipeline. Based on that, we further perform a systematic empirical study and evaluation of twelve representative object detectors covering three different categories of architectures (i.e., two-stage, one-stage, transformer architectures) to identify the current challenges and explore future opportunities. Our key findings include (1) the proposed real-world corruptions pose a threat to object detectors, especially for the corruptions involving colour changes, (2) a detector with a high mAP may still be vulnerable to real-world corruptions, (3) if there are potential cross-scenarios applications, the one-stage detectors are recommended, (4) when object detection architectures suffer from real-world corruptions, the effectiveness of existing robustness enhancement methods is limited, and (5) two-stage and one-stage object detection architectures are more likely to miss detect objects compared with transformer-based methods against the proposed corruptions. Our results highlight the need for designing robust object detection methods against real-world corruption and the need for more effective robustness enhancement methods for existing object detectors.},
  day      = {01},
  doi      = {10.1007/s11263-024-02096-6},
  url      = {https://doi.org/10.1007/s11263-024-02096-6},
}
website: https://sites.google.com/view/real-worldbenchmark/real-world-benchmarking
Code: https://github.com/Jackie-Bai888/image-noise-pattern
"""

import cv2
import random
import numpy as np
import math
from skimage import exposure
from imdeg.utils.image_conversion import RGB2Bayer_RG


########################################################################################################################
def lens_shading_error_adapted(img, alpha=0.5):
    original_shape = img.shape[:2]
    img = RGB2Bayer_RG(img)

    row, col = img.shape
    if row % 2 != 0:
        img = img[:row - 1, :]
        row -= 1
    if col % 2 != 0:
        img = img[:, :col - 1]
        col -= 1

    size = math.sqrt((row // 2) ** 2 + (col // 2) ** 2)
    center = (row // 2, col // 2)

    # FIXED: r should be proportional to size, controlled by alpha
    # Higher alpha → smaller r → more darkening
    r = size * (1.0 - alpha)  # alpha in [0.1, 0.8]

    # Now a will be negative (darkening effect)
    a = 1 / (r - size)
    b = 0 - size * a

    # Vectorized version
    y_coords, x_coords = np.ogrid[:row, :col]
    distances = np.sqrt((y_coords - center[0]) ** 2 + (x_coords - center[1]) ** 2)

    shading = np.ones((row, col))
    mask = distances > r
    shading[mask] = a * distances[mask] + b
    shading = np.clip(shading, 0, 1)  # Prevent negative values

    img = np.round(img * shading).astype(img.dtype)

    dst = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2BGR)

    if dst.shape[:2] != original_shape:
        dst = cv2.resize(dst, (original_shape[1], original_shape[0]),
                         interpolation=cv2.INTER_LINEAR)

    return dst

def color_space_error_adapted(img, num_precent=0.6):
    y_cb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(y_cb_img)

    Y = Y.astype(np.int16)
    Cr = Cr.astype(np.int16)
    Cb = Cb.astype(np.int16)

    height, width, _ = img.shape
    num = int(height * width * num_precent)

    # Mask for random pixel
    mask = np.zeros(height * width, dtype=bool)
    pixel_position = random.sample(range(height * width), num)
    mask[pixel_position] = True
    mask = mask.reshape(height, width)

    #
    Y[mask] -= 16
    Cr[mask] -= 128
    Cb[mask] -= 128

    Y = np.clip(Y, 0, 255).astype(np.uint8)
    Cr = np.clip(Cr, 0, 255).astype(np.uint8)
    Cb = np.clip(Cb, 0, 255).astype(np.uint8)

    dst = cv2.merge([Y, Cr, Cb])
    output_img = cv2.cvtColor(dst, cv2.COLOR_YCrCb2BGR)
    return output_img


def memory_corruption_adapted(img, severity: int = 3):
    """
    Apply memory corruption with severity-based threshold.

    Args:
        severity: 1 (minimal) to 5 (maximum corruption)
    """
    # Map severity to threshold: higher severity = lower threshold = more corruption
    threshold_map = {
        1: 16,  # Low threshold = many pixels pass = minimal corruption
        2: 32,  # Light corruption
        3: 64,  # Moderate corruption
        4: 128,  # Heavy corruption
        5: 200  # High threshold = few pixels pass = severe corruption
    }
    threshold = threshold_map.get(severity, 32)

    b, g, r = cv2.split(img)
    row, col, chs = img.shape

    damageB = np.random.randint(0, 256, img.shape[0:2], dtype=np.uint8)
    damageG = np.random.randint(0, 256, img.shape[0:2], dtype=np.uint8)
    damageR = np.random.randint(0, 256, img.shape[0:2], dtype=np.uint8)

    tB, damageB = cv2.threshold(damageB, threshold, 255, cv2.THRESH_BINARY)
    tG, damageG = cv2.threshold(damageG, threshold, 255, cv2.THRESH_BINARY)
    tR, damageR = cv2.threshold(damageR, threshold, 255, cv2.THRESH_BINARY)

    b = cv2.bitwise_and(b, damageB)
    g = cv2.bitwise_and(g, damageG)
    r = cv2.bitwise_and(r, damageR)

    dst = cv2.merge([b, g, r])
    return dst

def sensor_broken_adapted(
    img: np.ndarray,
    spot_count: int = 10,
    spot_alpha: float = 0.5,
    deviation_strength: float = 0.6,
    leak_length: int = 40,
    spot: bool = True,
    deviation: bool = True,
    leak: bool = True,
) -> np.ndarray:
    des = img.copy()

    h, w = img.shape[:2]
    x = int(random.uniform(0, h))
    y = int(random.uniform(100, w) - 100) if w > 100 else w // 2

    if spot:
        img_spot = np.zeros(img.shape, np.uint8)
        for _ in range(spot_count):
            center = (
                int(random.uniform(0, w)),
                int(random.uniform(0, h)),
            )
            radius = int(random.uniform(1, 7))
            cv2.circle(img_spot, center, radius, (0, 0, 255), -1)

        des = cv2.addWeighted(des, 1.0, img_spot, spot_alpha, 0)

    if deviation:
        img_deviation = np.zeros(img.shape, np.uint8)
        axis = int(random.uniform(10, 20))

        for i in range(50, -1, -1):
            col = y - i + axis
            if 0 <= col < w:
                value = max(0, int((255 - i * 5 - 5) * deviation_strength))
                img_deviation[:, col, 0] = value
                img_deviation[:, col, 2] = value

        for j in range(50, 0, -1):
            col = y + j + axis
            if 0 <= col < w:
                value = max(0, int((255 - j * 5 - 5) * deviation_strength))
                img_deviation[:, col, 0] = value
                img_deviation[:, col, 2] = value

        des = cv2.addWeighted(des, 1.0, img_deviation, 0.6, 0)

    if leak:
        axis = int(random.uniform(10, 30))
        des[max(0, x - 1):min(h, x + 1), :, :] = 0

        for i in range(leak_length):
            col = y - i + axis
            if 0 <= col < w:
                end_row = int(x + 16 - (i / 10) ** 2) + int(random.uniform(0, max(1, 20 - i // 2)))
                end_row = max(x, min(h, end_row))
                des[x:end_row, col, :] = 0

        for j in range(leak_length):
            col = y + j + axis
            if 0 <= col < w:
                end_row = int(x + 16 - (j / 10) ** 2) + int(random.uniform(0, max(1, 20 - j // 2)))
                end_row = max(x, min(h, end_row))
                des[x:end_row, col, :] = 0

    return des


