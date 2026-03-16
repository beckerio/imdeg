"""
Adapted from:

https://github.com/miccunifi/ARNIQA/blob/main/utils/utils_distortions.py

@InProceedings{Agnolucci_WACV_2024,
  author    = {Agnolucci, Lorenzo and Galteri, Leonardo and Bertini, Marco and Del Bimbo, Alberto},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  title     = {ARNIQA: Learning Distortion Manifold for Image Quality Assessment},
  year      = {2024},
  pages     = {189--198},
}

License: Apache License 2.0
https://github.com/miccunifi/ARNIQA/tree/main?tab=Apache-2.0-1-ov-file#readme
"""
import math
import numpy as np
from typing import Union, Tuple, List, Callable, Dict
import torch
from torch.nn import functional as F
import scipy

import random
import kornia
import io
from torchvision.io.image import decode_jpeg, encode_jpeg
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

distortion_groups = {
    "blur": ["gaublur", "lensblur", "motionblur"],
    "color_distortion": ["colordiff", "colorshift", "colorsat1", "colorsat2"],
    "jpeg": ["jpeg2000", "jpeg"],
    "noise": ["whitenoise", "whitenoiseCC", "impulsenoise", "multnoise"],
    "brightness_change": ["brighten", "darken", "meanshift"],
    "spatial_distortion": ["jitter", "noneccpatch", "pixelate", "quantization", "colorblock"],
    "sharpness_contrast": ["highsharpen", "lincontrchange", "nonlincontrchange"],
}

full_names = {
    "brighten": "Brighten",
    "darken": "Darken",
    "meanshift": "Mean Shift",

    "gaublur": "Gaussian Blur",
    "lensblur": "Lens Blur",
    "motionblur": "Motion Blur",

    "colordiff": "Color Diffusion",
    "colorshift": "Color Shift",
    "colorsat1": "Color Saturation 1",
    "colorsat2": "Color Saturation 2",

    "jpeg2000": "JPEG2000",
    "jpeg": "JPEG",

    "whitenoise": "White Noise",
    "whitenoiseCC": "White Noise CC",# "White noise cc in color component"
    "impulsenoise": "Impulse Noise",
    "multnoise": "Multiplicative Noise",

    "jitter": "Jitter",
    "noneccpatch": "Non-Eccentricity Patch",
    "pixelate": "Pixelate",
    "quantization": "Quantization",
    "colorblock": "Color Block",

    "highsharpen": "High Sharpen",
    "lincontrchange": "Linear Contrast Change",
    "nonlincontrchange": "Nonlinear Contrast Change",
}




distortion_groups_mapping = {
    "gaublur": "blur",
    "lensblur": "blur",
    "motionblur": "blur",
    "colordiff": "color_distortion",
    "colorshift": "color_distortion",
    "colorsat1": "color_distortion",
    "colorsat2": "color_distortion",
    "jpeg2000": "jpeg",
    "jpeg": "jpeg",
    "whitenoise": "noise",
    "whitenoiseCC": "noise",
    "impulsenoise": "noise",
    "multnoise": "noise",
    "brighten": "brightness_change",
    "darken": "brightness_change",
    "meanshift": "brightness_change",
    "jitter": "spatial_distortion",
    "noneccpatch": "spatial_distortion",
    "pixelate": "spatial_distortion",
    "quantization": "spatial_distortion",
    "colorblock": "spatial_distortion",
    "highsharpen": "sharpness_contrast",
    "lincontrchange": "sharpness_contrast",
    "nonlincontrchange": "sharpness_contrast",
}

distortion_range = {
    "gaublur": [0.1, 0.5, 1, 2, 5],
    "lensblur": [1, 2, 4, 6, 8],
    "motionblur": [1, 2, 4, 6, 10],
    "colordiff": [1, 3, 6, 8, 12],
    "colorshift": [1, 3, 6, 8, 12],
    "colorsat1": [0.4, 0.2, 0.1, 0, -0.4],
    "colorsat2": [1, 2, 3, 6, 9],
    "jpeg2000": [16, 32, 45, 120, 170],
    "jpeg": [43, 36, 24, 7, 4],
    "whitenoise": [0.001, 0.002, 0.003, 0.005, 0.01],
    "whitenoiseCC": [0.0001, 0.0005, 0.001, 0.002, 0.003],
    "impulsenoise": [0.001, 0.005, 0.01, 0.02, 0.03],
    "multnoise": [0.001, 0.005, 0.01, 0.02, 0.05],
    "brighten": [0.1, 0.2, 0.4, 0.7, 1.1],
    "darken": [0.05, 0.1, 0.2, 0.4, 0.8],
    "meanshift": [0, 0.08, -0.08, 0.15, -0.15],
    "jitter": [0.05, 0.1, 0.2, 0.5, 1],
    "noneccpatch": [20, 40, 60, 80, 100],
    "pixelate": [0.01, 0.05, 0.1, 0.2, 0.5],
    "quantization": [20, 16, 13, 10, 7],
    "colorblock": [2, 4, 6, 8, 10],
    "highsharpen": [1, 2, 3, 6, 12],
    "lincontrchange": [0., 0.15, -0.4, 0.3, -0.6],
    "nonlincontrchange": [0.4, 0.3, 0.2, 0.1, 0.05],
}


def get_distortions_composition(max_distortions: int = 7, num_levels: int = 5) -> (List[Callable], List[Union[int, float]]):
    """
    Image Degradation model proposed in the paper https://arxiv.org/abs/2310.14918. Returns a randomly assembled ordered
    sequence of distortion functions and their values.

    Args:
        max_distortions (int): maximum number of distortions to apply to the image
        num_levels (int): number of levels of distortion that can be applied to the image

    Returns:
        distort_functions (list): list of the distortion functions to apply to the image
        distort_values (list): list of the values of the distortion functions to apply to the image
    """
    if max_distortions > 7:
        print("⚠️  max_distortions is 7  ⚠️")
    max_distortions = min(7, max_distortions)

    if num_levels > 5:
        print("⚠️  num_levels is 5  ⚠️")
    num_levels = min(5, num_levels)

    MEAN = 0 # default
    #MEAN = (num_levels - 1) / 2
    STD = 2.5

    #print("number of dist groups ",len(list(distortion_groups.keys())))

    num_distortions = random.randint(1, max_distortions)
    groups = random.sample(list(distortion_groups.keys()), num_distortions)
    distortions = [random.choice(distortion_groups[group]) for group in groups]
    distort_functions = [distortion_functions[dist] for dist in distortions]

    probabilities = [1 / (STD * np.sqrt(2 * np.pi)) * np.exp(-((i - MEAN) ** 2) / (2 * STD ** 2))
                     for i in range(num_levels)]  # probabilities according to a gaussian distribution
    normalized_probabilities = [prob / sum(probabilities)
                                for prob in probabilities]  # normalize probabilities
    distort_values = [np.random.choice(distortion_range[dist][:num_levels], p=normalized_probabilities) for dist
                      in distortions]

    return distort_functions, distort_values

def distort_images(image: torch.Tensor, distort_functions: list = None, distort_values: list = None,
                   max_distortions: int = 4, num_levels: int = 5) -> torch.Tensor:
    """
    Distorts an image using the distortion composition obtained with the image degradation model proposed in the paper
    ARNIQA: Learning Distortion Manifold for Image Quality Assessment
    https://arxiv.org/abs/2310.14918.

    Args:
        image (Tensor): image to distort
        distort_functions (list): list of the distortion functions to apply to the image. If None, the functions are randomly chosen.
        distort_values (list): list of the values of the distortion functions to apply to the image. If None, the values are randomly chosen.
        max_distortions (int): maximum number of distortions to apply to the image
        num_levels (int): number of levels of distortion that can be applied to the image

    Returns:
        image (Tensor): distorted image
        distort_functions (list): list of the distortion functions applied to the image
        distort_values (list): list of the values of the distortion functions applied to the image
    """
    if distort_functions is None or distort_values is None:
        distort_functions, distort_values = get_distortions_composition(max_distortions, num_levels)

    for distortion, value in zip(distort_functions, distort_values):
        image = distortion(image, value)
        image = image.to(torch.float32)
        image = torch.clip(image, 0, 1)
    return image, distort_functions, distort_values


def sign(x: float) -> int:
    return 1 if x >= 0 else -1


def mapmm(x: torch.Tensor) -> torch.Tensor:
    minx = torch.min(x)
    maxx = torch.max(x)
    if minx < maxx:
        x = (x - minx) / (maxx - minx)
    return x


def fspecial(filter_type: str, p2: Union[int, Tuple[int, int]], p3: Union[int, float] = None) -> np.ndarray:
    if filter_type == 'gaussian':
        m, n = [(ss - 1.) / 2. for ss in p2]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * p3 * p3))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh

        return h

    elif filter_type == 'disk':
        rad = p2
        crad = math.ceil(rad - 0.5)

        x, y = np.ogrid[-rad: rad + 1, -rad: rad + 1]
        y = np.tile(y.transpose(), y.shape[1])
        x = np.tile(x, x.shape[0]).transpose()

        y = np.abs(y)
        x = np.abs(x)

        maxxy = np.maximum(x, y)
        minxy = np.minimum(x, y)

        r1 = (rad ** 2 - (maxxy + 0.5) ** 2)
        r2 = (rad ** 2 - (minxy - 0.5) ** 2)

        if (r1 > 0).all():
            warn_m1 = r1 ** 0.5
        else:
            warn_m1 = 0
        if (r2 > 0).all():
            warn_m2 = r2 ** 0.5
        else:
            warn_m2 = 0

        m1 = (rad ** 2 < (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2) * (minxy - 0.5) + (
                rad ** 2 >= (maxxy + 0.5) ** 2 + (minxy - 0.5) ** 2) * warn_m1
        m2 = (rad ** 2 > (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2) * (minxy + 0.5) + (
                rad ** 2 <= (maxxy - 0.5) ** 2 + (minxy + 0.5) ** 2) * warn_m2

        sgrid = (rad ** 2 * (0.5 * (np.arcsin(m2 / rad) - np.arcsin(m1 / rad)) +
                             0.25 * (np.sin(2 * np.arcsin(m2 / rad)) - np.sin(2 * np.arcsin(m1 / rad)))) - (
                         maxxy - 0.5) * (m2 - m1) + (m1 - minxy + 0.5)) * np.logical_or(
            np.logical_and((rad ** 2 < (maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2),
                           (rad ** 2 > (maxxy - 0.5) ** 2 + (minxy - 0.5) ** 2)),
            np.logical_and(np.logical_and(minxy == 0, maxxy - 0.5 < rad), maxxy + 0.5 >= rad))

        sgrid = sgrid + ((maxxy + 0.5) ** 2 + (minxy + 0.5) ** 2 < rad ** 2)
        sgrid[crad, crad] = np.minimum(math.pi * rad ** 2, math.pi / 2)
        if (crad > 0) and (rad > crad - 0.5) and (rad ** 2 < (crad - 0.5) ** 2 + 0.25):
            m1 = np.sqrt(rad ** 2 - (crad - 0.5) ** 2)
            m1n = m1 / rad
            sg0 = 2 * (rad ** 2 * (0.5 * np.arcsin(m1n) + 0.25 * np.sin(2 * np.arcsin(m1n))) - m1 * (crad - 0.5))
            sgrid[2 * crad, crad] = sg0
            sgrid[crad, 2 * crad] = sg0
            sgrid[crad, 0] = sg0
            sgrid[0, crad] = sg0
            sgrid[2 * crad, crad] = sgrid[2 * crad, crad] - sg0
            sgrid[crad, 2 * crad] = sgrid[crad, 2 * crad] - sg0
            sgrid[crad, 2] = sgrid[crad, 2] - sg0
            sgrid[2, crad] = sgrid[2, crad + 1] - sg0

        sgrid[crad, crad] = np.minimum(sgrid[crad, crad], 1)
        h = sgrid / np.sum(sgrid)
        return h
    elif filter_type == 'motion':

        eps = 2.2204e-16
        length = max(1, p2)
        half_len = (length - 1) / 2.
        phi = (p3 % 180) / 180 * math.pi

        cosphi = math.cos(phi)
        sinphi = math.sin(phi)
        xsign = sign(cosphi)
        linewdt = 1

        sx = int(half_len * cosphi + linewdt * xsign - length * eps)
        sy = int(half_len * sinphi + linewdt - length * eps)
        x, y = np.mgrid[0:sx + (1 * xsign):xsign, 0:sy + 1]
        x = x.transpose()
        y = y.transpose()

        dist2line = (y * cosphi - x * sinphi)
        rad = (x ** 2 + y ** 2) ** 0.5

        lastpix = np.where(np.logical_and((rad >= half_len), (abs(dist2line) <= linewdt)))
        x2lastpix = half_len - np.abs((x[lastpix] + dist2line[lastpix] * sinphi) / cosphi);

        dist2line[lastpix] = np.sqrt(dist2line[lastpix] ** 2 + x2lastpix ** 2)
        dist2line = linewdt + eps - np.abs(dist2line)
        dist2line[dist2line < 0] = 0

        h = np.rot90(dist2line, 2)
        tmp_h = np.zeros((h.shape[0] * 2 - 1, h.shape[1] * 2 - 1))
        tmp_h[0:h.shape[0], 0:h.shape[1]] = h
        tmp_h[(h.shape[0]) - 1:, h.shape[1] - 1:] = dist2line
        h = tmp_h

        h /= np.sum(h) + eps * length * length

        if cosphi > 0:
            h = np.flipud(h)

        return h

    else:
        raise NotImplementedError(f"Filter type {filter_type} not implemented")


def filter2D(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """PyTorch version of cv2.filter2D
    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    img = img.float()
    k1 = kernel.size(-2)
    k2 = kernel.size(-1)

    b, c, h, w = img.size()
    if k1 % 2 == 1 or k2 % 2 == 1:
        img = F.pad(img, (k2 // 2, k2 // 2, k1 // 2, k1 // 2), mode='replicate')
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k1, k2)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k1, k2).repeat(1, c, 1, 1).view(b * c, 1, k1, k2)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


def curves(xx: torch.Tensor, coef: float) -> torch.Tensor:
    if type(coef) == list:
        coef = [[0.3, 0.5, 0.7],
                [coef[0], 0.5, coef[1]]]
    else:
        coef = [[0.5], [coef]]

    x = np.array([0] + [p for p in coef[0]] + [1])
    y = np.array([0] + [p for p in coef[1]] + [1])

    cs = spline(x, y)

    yy = ppval(cs, xx)

    yy = torch.clamp(yy, 0, 1)

    return yy


def spline(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    dd = 1
    dx = np.diff(x)
    divdif = np.diff(y) / dx

    if n == 3:
        y[1:3] = divdif
        y[2] = np.diff(divdif.T).T / (x[2] - x[0])
        y[1] -= y[2] * dx[0]
        dlk = y[[2, 1, 0]].shape[0]
        l = x[[0, 2]].shape[0] - 1
        dl = np.prod(dd) * l
        k = np.fix(dlk / dl + 100 * 2.2204e-16)

        pp = (x[[0, 2]], y[[2, 1, 0]], l, int(k), dd)

    elif n > 3:
        b = np.zeros(n)
        b[1:n - 1] = 3 * (dx[1:n] * divdif[0:n - 2] + dx[0:n - 2] * divdif[1:n])

        x31 = x[2] - x[0]
        xn = x[n - 1] - x[n - 3]

        b[0] = ((dx[0] + 2 * x31) * dx[1] * divdif[0] + dx[0] ** 2 * divdif[1]) / x31
        b[n - 1] = (dx[n - 2] ** 2 * divdif[n - 3] + (2 * xn + dx[n - 2]) * dx[n - 3] * divdif[n - 2]) / xn;

        dxt = dx.T
        c = np.zeros((3, 5))
        c[0, :] = [x31] + list(dxt[0:n - 2]) + [0]
        c[1, :] = [dxt[1]] + list(2 * (dxt[1:n - 1] + dxt[0:n - 2])) + [dxt[n - 3]]
        c[2, :] = [0] + list(dxt[1:n - 1]) + [xn]

        c = scipy.sparse.dia_matrix((c, [-1, 0, 1]), shape=(5, 5))
        c = scipy.sparse.csc_matrix(c)
        ic = scipy.sparse.linalg.inv(c)
        s = b * ic

        n = x.shape[0]
        d = 1
        dxd = dx

        dzzdx = (divdif - s[0:n - 1]) / dxd
        dzdxdx = (s[1:n] - divdif) / dxd

        coefs = np.vstack(((dzdxdx - dzzdx) / dxd, 2 * dzzdx - dzdxdx, s[0:n - 1], y[0:n - 1])).T

        pp = (x, coefs, x.shape[0], x.shape[0], d)
    else:
        raise ValueError('x.shape[0] must be >= 3')

    return pp


def ppval(pp: np.ndarray, xx: torch.Tensor) -> torch.Tensor:
    lx = torch.numel(xx)
    xs = xx.reshape(1, lx)
    b, c, l, k, dd = pp
    b = torch.as_tensor(b, device=xx.device)
    ranges = b.clone()
    ranges[0] = -torch.inf
    ranges[-1] = torch.inf
    index = histc(xs, ranges)

    xs = xs - b[index]

    c = torch.as_tensor(c, device=xx.device)

    if len(c.shape) == 1:
        v = c[0]
        for i in range(1, k):
            v = xs * v + c[i]
    else:
        v = c[index, 0]

        for i in range(1, k - 1):
            v = xs * v + c[index, i]
    v = v.view(xx.shape)
    return v


def histc(x: torch.Tensor, binranges: torch.Tensor) -> torch.Tensor:
    indices = torch.bucketize(x, binranges)
    return torch.remainder(indices, len(binranges)) - 1


def imscatter(x: torch.Tensor, amount: float, iterations=1) -> torch.Tensor:
    y = x
    for i in range(iterations):
        shiftmap = torch.randn((2, x.shape[1], x.shape[2]), device=x.device) * amount

        sy = shiftmap[0, :, :]
        sx = shiftmap[1, :, :]

        m_sx = torch.ceil(torch.abs(torch.max(sx))).to(torch.int32)
        m_sy = torch.ceil(torch.abs(torch.max(sy))).to(torch.int32)

        y = F.pad(y, (m_sy, m_sy), mode='replicate')
        y = F.pad(y.transpose(2, 1), (m_sx, m_sx), mode='replicate').transpose(2, 1)

        sy = F.pad(sy, (m_sy, m_sy), mode='replicate')
        sy = F.pad(sy.transpose(1, 0), (m_sx, m_sx), mode='replicate').transpose(1, 0)
        sx = F.pad(sx, (m_sy, m_sy), mode='replicate')
        sx = F.pad(sx.transpose(1, 0), (m_sx, m_sx), mode='replicate').transpose(1, 0)

        xx, yy = torch.as_tensor(np.mgrid[0:y.shape[1], 0:y.shape[2]], device=x.device)

        z = torch.zeros_like(y)
        bx = (xx - sx)
        by = (yy - sy)
        for i in range(3):
            j = bilinear_interpolate_torch(y[i, ...], by, bx)
            z[i, :, :] = j

        y = z[:, m_sy:m_sy + x.shape[1], m_sx:m_sx + x.shape[2]]
    return y


def bilinear_interpolate_torch(im: torch.Tensor, x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    dtype_long = torch.LongTensor

    x0 = torch.floor(x).type(dtype_long).to(im.device)
    x1 = x0 + 1

    y0 = torch.floor(y).type(dtype_long).to(im.device)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    R1 = Ia * (x1 - x) / (x1 - x0 + eps) + Ic * (x - x0) / (x1 - x0 + eps)
    R2 = Ib * (x1 - x) / (x1 - x0 + eps) + Id * (x - x0) / (x1 - x0 + eps)
    P = R1 * (y1 - y) / (y1 - y0 + eps) + R2 * (y - y0) / (y1 - y0 + eps)
    return P


"""------------------------------------------------------------------------------------------------------------------"""
""" distortions """
"""------------------------------------------------------------------------------------------------------------------"""

def gaussian_blur(x: torch.Tensor, blur_sigma: int = 0.1) -> torch.Tensor:
    fs = 2 * math.ceil(2 * blur_sigma) + 1
    h = fspecial('gaussian', (fs, fs), blur_sigma)
    h = torch.from_numpy(h).float()

    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    y = filter2D(x, h.unsqueeze(0)).squeeze(0)
    return y


def lens_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    h = fspecial('disk', radius)
    h = torch.from_numpy(h).float()

    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    y = filter2D(x, h.unsqueeze(0)).squeeze(0)
    return y


def motion_blur(x: torch.Tensor, radius: int, angle: bool = None) -> torch.Tensor:
    if angle is None:
        angle = random.randint(0, 180)
    h = fspecial('motion', radius, angle)
    h = torch.from_numpy(h.copy()).float()

    if len(x.shape) == 3:
        x = x.unsqueeze(0)

    y = filter2D(x, h.unsqueeze(0)).squeeze(0)
    return y

def color_diffusion(x: torch.Tensor, amount: int) -> torch.Tensor:
    blur_sigma = 1.5 * amount + 2
    scaling = amount
    x = x[[2, 1, 0], ...]
    lab = kornia.color.rgb_to_lab(x)

    fs = 2 * math.ceil(2 * blur_sigma) + 1
    h = fspecial('gaussian', (fs, fs), blur_sigma)
    h = torch.from_numpy(h).float()

    if len(lab.shape) == 3:
        lab = lab.unsqueeze(0)

    diff_ab = filter2D(lab[:, 1:3, ...], h.unsqueeze(0))
    lab[:, 1:3, ...] = diff_ab * scaling

    y = torch.trunc(kornia.color.lab_to_rgb(lab) * 255.) / 255.
    y = y[:, [2, 1, 0]].squeeze(0)
    return y


def color_shift(x: torch.Tensor, amount: int) -> torch.Tensor:
    def perc(x, perc):
        xs = torch.sort(x)
        i = len(xs) * perc / 100.
        i = max(min(i, len(xs)), 1)
        v = xs[round(i - 1)]
        return v

    gray = kornia.color.rgb_to_grayscale(x)
    gradxy = kornia.filters.spatial_gradient(gray.unsqueeze(0), 'diff')
    e = torch.sum(gradxy ** 2, 2) ** 0.5

    fs = 2 * math.ceil(2 * 4) + 1
    h = fspecial('gaussian', (fs, fs), 4)
    h = torch.from_numpy(h).float()

    e = filter2D(e, h.unsqueeze(0))

    mine = torch.min(e)
    maxe = torch.max(e)

    if mine < maxe:
        e = (e - mine) / (maxe - mine)

    percdev = [1, 1]
    valuehi = perc(e, 100 - percdev[1])
    valuelo = 1 - perc(1 - e, 100 - percdev[0])

    e = torch.max(torch.min(e, valuehi), valuelo)

    channel = 1 # ❌ HARDCODED - always shifts green channel TODO
    g = x[channel, :, :]
    a = np.random.random((1, 2))
    amount_shift = np.round(a / (np.sum(a ** 2) ** 0.5) * amount)[0].astype(int)

    y = F.pad(g, (amount_shift[0], amount_shift[0]), mode='replicate')
    y = F.pad(y.transpose(1, 0), (amount_shift[1], amount_shift[1]), mode='replicate').transpose(1, 0)
    y = torch.roll(y, (amount_shift[0], amount_shift[1]), dims=(0, 1))

    if amount_shift[1] != 0:
        y = y[amount_shift[1]:-amount_shift[1], ...]
    if amount_shift[0] != 0:
        y = y[..., amount_shift[0]:-amount_shift[0]]

    yblend = y * e + x[channel, ...] * (1 - e)
    x[channel, ...] = yblend

    return x


def color_saturation1(x: torch.Tensor, factor: int) -> torch.Tensor:
    x = x[[2, 1, 0], ...]
    hsv = kornia.color.rgb_to_hsv(x)
    hsv[1, ...] *= factor
    y = kornia.color.hsv_to_rgb(hsv)
    return y[[2, 1, 0], ...]


def color_saturation2(x: torch.Tensor, factor: int) -> torch.Tensor:
    x = x[[2, 1, 0], ...]
    lab = kornia.color.rgb_to_lab(x)
    lab[1:3, ...] = lab[1:3, ...] * factor
    y = torch.trunc(kornia.color.lab_to_rgb(lab) * 255) / 255.
    return y[[2, 1, 0], ...]


def jpeg2000_old(x: torch.Tensor, ratio: int) -> torch.Tensor:
    ratio = int(ratio)
    compression_params = {
        'quality_mode': 'rates',
        'quality_layers': [ratio],  # Compression ratio
        'num_resolutions': 8,  # Number of wavelet decompositions
        'prog_order': 'LRCP',  # Progression order: Layer-Resolution-Component-Position
    }

    # Compress the image and save it using the JPEG2000 format
    x *= 255.
    x = x.byte().cpu().numpy()

    x = Image.fromarray(x.transpose(1, 2, 0), 'RGB')

    with io.BytesIO() as output:
        #x.save(output, format='JPEG2000', **compression_params)
        x.save(output, format='JPEG2000', kind='jp2', **compression_params)  # minimal save
        #x.save(output, format='JP2', **compression_params)
        compressed_data = output.getvalue()

    y = Image.open(io.BytesIO(compressed_data))
    y = transforms.ToTensor()(y)

    return y

def jpeg2000(x: torch.Tensor, ratio: int) -> torch.Tensor:

    # Validate inputs
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(f"Expected tensor shape (3,H,W), got {tuple(x.shape)}")
    if ratio is None or int(ratio) < 2:
        # JPEG2000 'rates' must be >1; 2–50 is typical
        ratio = 10

    C, H, W = x.shape
    # Convert to uint8 safely
    x_uint8 = (x.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    img = Image.fromarray(x_uint8.transpose(1, 2, 0), mode='RGB')

    # Choose a safe num_resolutions based on image size
    # OpenJPEG uses levels = num_resolutions - 1; levels cannot exceed floor(log2(min(H,W)))
    max_levels = max(0, int(math.floor(math.log2(min(H, W)))))
    num_resolutions = min(6, max_levels + 1)  # keep conservative

    compression_params = {
        'quality_mode': 'rates',
        'quality_layers': [float(ratio)],   # must be > 1
        'num_resolutions': num_resolutions,  # Number of wavelet decompositions
        'prog_order': 'LRCP',
        'irreversible': True,               # use lossy 9x7 wavelet for 'rates'
        'mct': True,                        # multicomponent transform for RGB
    }

    with io.BytesIO() as output:
        img.save(output, format='JPEG2000', **compression_params)
        compressed_data = output.getvalue()

    with Image.open(io.BytesIO(compressed_data)) as y_img:
        y_img = y_img.convert('RGB')

    y = transforms.ToTensor()(y_img)
    return y


def jpeg(x: torch.Tensor, quality: int) -> torch.Tensor:
    x *= 255.
    y = encode_jpeg(x.byte().cpu(), quality=quality)
    y = (decode_jpeg(y) / 255.).to(torch.float32)
    return y


def white_noise(x: torch.Tensor, var: float, clip: bool = True, rounds: bool = False) -> torch.Tensor:
    noise = torch.randn(*x.size(), dtype=x.dtype) * math.sqrt(var)

    y = x + noise

    if clip and rounds:
        y = torch.clip((y * 255.0).round(), 0, 255) / 255.
    elif clip:
        y = torch.clip(y, 0, 1)
    elif rounds:
        y = (y * 255.0).round() / 255.
    return y


def white_noise_cc(x: torch.Tensor, var: float, clip: bool = True, rounds: bool = False) -> torch.Tensor:
    noise = torch.randn(*x.size(), dtype=x.dtype) * math.sqrt(var)

    ycbcr = kornia.color.rgb_to_ycbcr(x)
    y = ycbcr + noise

    y = kornia.color.ycbcr_to_rgb(y)

    if clip and rounds:
        y = torch.clip((y * 255.0).round(), 0, 255) / 255.
    elif clip:
        y = torch.clip(y, 0, 1)
    elif rounds:
        y = (y * 255.0).round() / 255.

    return y


def impulse_noise(x: torch.Tensor, d: float, s_vs_p: float = 0.5) -> torch.Tensor:
    num_sp = int(d * x.shape[0] * x.shape[1] * x.shape[2])

    coords = np.concatenate((np.random.randint(0, x.shape[0], (num_sp, 1)),
                             np.random.randint(0, x.shape[1], (num_sp, 1)),
                             np.random.randint(0, x.shape[2], (num_sp, 1))), 1)
    
    num_salt = int(s_vs_p * num_sp)

    coords_salt = coords[:num_salt].transpose(1, 0)
    coords_pepper = coords[num_salt:].transpose(1, 0)

    x[tuple(coords_salt)] = 1
    x[tuple(coords_pepper)] = 0

    return x


def multiplicative_noise(x: torch.Tensor, var: float) -> torch.Tensor:
    noise = torch.randn(*x.size(), dtype=x.dtype) * math.sqrt(var)
    y = x + x * noise
    y = torch.clip(y, 0, 1)
    return y


def brighten(x: torch.Tensor, amount: float) -> torch.Tensor:
    x = x[[2, 1, 0]]
    lab = kornia.color.rgb_to_lab(x)

    l = lab[0, ...] / 100.
    l_ = curves(l, 0.5 + amount / 2)
    lab[0, ...] = l_ * 100.

    y = curves(x, 0.5 + amount / 2)

    j = torch.clamp(kornia.color.lab_to_rgb(lab), 0, 1)

    y = (2 * y + j) / 3

    return y[[2, 1, 0]]


def darken(x: torch.Tensor, amount: float, dolab: bool = False) -> torch.Tensor:
    x = x[[2, 1, 0], :, :]
    lab = kornia.color.rgb_to_lab(x)
    if dolab:
        l = lab[0, ...] / 100.
        l_ = curves(l, 0.5 + amount / 2)
        lab[0, ...] = l_ * 100.

    y = curves(x, 0.5 - amount / 2)

    if dolab:
        j = torch.clamp(kornia.color.lab_to_rgb(lab), 0, 1)
        y = (2 * y + j) / 3

    return y[[2, 1, 0]]


def mean_shift(x: torch.Tensor, amount: float) -> torch.Tensor:
    x = x[[2, 1, 0], :, :]

    y = torch.clamp(x + amount, 0, 1)
    return y[[2, 1, 0]]


def jitter(x: torch.Tensor, amount: float) -> torch.Tensor:
    y = imscatter(x, amount, 5)  # ❌ Fixed 5 iterations
    return y


def non_eccentricity_patch(x: torch.Tensor, pnum: int) -> torch.Tensor:
    y = x
    patch_size = [16, 16]
    radius = 16
    h_min = radius
    w_min = radius
    c, h, w = x.shape

    h_max = h - patch_size[0] - radius
    w_max = w - patch_size[1] - radius

    for i in range(pnum):
        w_start = round(random.random() * (w_max - w_min)) + w_min
        h_start = round(random.random() * (h_max - h_min)) + h_min
        patch = y[:, h_start:h_start + patch_size[0], w_start:w_start + patch_size[0]]

        rand_w_start = round((random.random() - 0.5) * radius + w_start)
        rand_h_start = round((random.random() - 0.5) * radius + h_start)
        y[:, rand_h_start:rand_h_start + patch_size[0], rand_w_start:rand_w_start + patch_size[0]] = patch

    return y


def pixelate(x: torch.Tensor, strength: float) -> torch.Tensor:
    z = 0.95 - strength ** 0.6
    c, h, w = x.shape

    ylo = kornia.geometry.transform.resize(x, (int(h * z), int(w * z)), 'nearest')
    y = kornia.geometry.transform.resize(ylo, (h, w), 'nearest')

    return y


def quantization(x: torch.Tensor, levels: int) -> torch.Tensor:
    image = kornia.color.rgb_to_grayscale(x) * 255
    image = image.cpu().numpy()
    num_classes = levels

    # minimum variance thresholding
    hist, bins = np.histogram(image, num_classes, [0, 255])

    return_thresholds = np.zeros(num_classes - 1)
    for i in range(num_classes - 1):
        return_thresholds[i] = bins[i + 1]

    # quantize image with thresholds
    bins = torch.tensor([0] + return_thresholds.tolist() + [256])
    bins = bins.type(torch.int)
    image = torch.bucketize(x.contiguous() * 255., bins).to(torch.float32)
    image = mapmm(image)
    return image


def color_block(x: torch.Tensor, pnum: int) -> torch.Tensor:
    patch_size = [32, 32]

    c, w, h = x.shape

    y = x

    h_max = h - patch_size[0]
    w_max = w - patch_size[1]

    for i in range(pnum):
        color = np.random.random(3)
        px = math.floor(random.random() * w_max)
        py = math.floor(random.random() * h_max)
        patch = torch.ones((3, patch_size[0], patch_size[1]))
        for j in range(3):
            patch[j, ...] *= color[j]
        y[:, px:px + patch_size[0], py:py + patch_size[1]] = patch

    return y


def high_sharpen(x: torch.Tensor, amount: int, radius: int = 3) -> torch.Tensor:
    x = x[[2, 1, 0], ...]
    lab = kornia.color.rgb_to_lab(x)
    l = lab[0:1, ...].unsqueeze(0)

    filt_radius = math.ceil(radius * 2)
    fs = 2 * filt_radius + 1
    h = fspecial('gaussian', (fs, fs), filt_radius)
    h = torch.from_numpy(h).float()

    sharp_filter = torch.zeros((fs, fs))
    sharp_filter[filt_radius, filt_radius] = 1
    sharp_filter = sharp_filter - h

    sharp_filter *= amount
    sharp_filter[filt_radius, filt_radius] += 1

    l = filter2D(l, sharp_filter.unsqueeze(0))

    lab[0, ...] = l

    if len(lab.shape) == 3:
        lab = lab.unsqueeze(0)

    y = kornia.color.lab_to_rgb(lab)
    y = y[:, [2, 1, 0]].squeeze(0)
    return y


def linear_contrast_change(x: torch.Tensor, amount: float) -> torch.Tensor:
    y = curves(x, [0.25 - amount / 4, 0.75 + amount / 4])
    return y


def non_linear_contrast_change(x: torch.Tensor, output_offset_value: float, output_central_value: float = 0.5,
                               input_offset_value: float = 0.5, input_central_value: float = 0.5) -> torch.Tensor:
    low_in = input_central_value - input_offset_value
    high_in = input_central_value + input_offset_value
    low_out = output_central_value - output_offset_value
    high_out = output_central_value + output_offset_value

    # Clip the input image to the specified input range
    x = np.clip(x, low_in, high_in)

    # Calculate the slope and intercept of the linear transformation
    slope = (high_out - low_out) / (high_in - low_in)
    intercept = low_out - slope * low_in

    # Apply the linear transformation to adjust the pixel values
    y = slope * x + intercept

    # Clip the adjusted image to the specified output range
    y = np.clip(y, low_out, high_out)

    return y

distortion_functions = {
    "gaublur": gaussian_blur,
    "lensblur": lens_blur,
    "motionblur": motion_blur,
    "colordiff": color_diffusion,
    "colorshift": color_shift,
    "colorsat1": color_saturation1,
    "colorsat2": color_saturation2,
    "jpeg2000": jpeg2000,
    "jpeg": jpeg,
    "whitenoise": white_noise,
    "whitenoiseCC": white_noise_cc,
    "impulsenoise": impulse_noise,
    "multnoise": multiplicative_noise,
    "brighten": brighten,
    "darken": darken,
    "meanshift": mean_shift,
    "jitter": jitter,
    "noneccpatch": non_eccentricity_patch,
    "pixelate": pixelate,
    "quantization": quantization,
    "colorblock": color_block,
    "highsharpen": high_sharpen,
    "lincontrchange": linear_contrast_change,
    "nonlincontrchange": non_linear_contrast_change,
}