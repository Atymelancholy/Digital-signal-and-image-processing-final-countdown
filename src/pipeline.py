"""Полный пайплайн лабораторной: очистка фона и выделение контуров."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src import author, library

# Узкие диапазоны: синий пластик очень насыщенный, дерево и пульт — нет.
BLUE_HSV = ((92, 170, 70), (115, 255, 255))
GREEN_HSV = ((65, 170, 70), (90, 255, 255))
MAX_SIDE = 1400
MIN_REGION_FRAC = 0.008


@dataclass
class PipelineResult:
    original: np.ndarray
    denoised: np.ndarray
    hsv: np.ndarray
    mask: np.ndarray
    cleaned: np.ndarray
    gray: np.ndarray
    contrasted: np.ndarray
    binary_otsu: np.ndarray
    sobel: np.ndarray
    prewitt: np.ndarray
    overlay: np.ndarray
    hist_original: np.ndarray
    hist_cleaned: np.ndarray


def get_backend(name: str) -> Any:
    if name == "author":
        return author
    if name == "library":
        return library
    raise ValueError("backend должен быть 'author' или 'library'")


def resize_max_side(rgb: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    from PIL import Image

    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return rgb
    scale = max_side / longest
    img = Image.fromarray(rgb)
    img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    return np.array(img)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Заливка внутренних дыр маски — так возвращаются зелёные цифры внутри карточки."""
    h, w = mask.shape[:2]
    flood = mask.copy()
    ffmask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ffmask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def keep_large_regions(mask: np.ndarray, min_frac: float = MIN_REGION_FRAC) -> np.ndarray:
    """Убирает мелкие остатки пульта/кабеля, оставляя только крупные карточки."""
    min_area = max(500, int(min_frac * mask.shape[0] * mask.shape[1]))
    _, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    keep = np.zeros_like(mask)
    for i in range(1, stats.shape[0]):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == i] = 255
    return keep


def process(rgb: np.ndarray, backend: str = "author", median_ksize: int = 5) -> PipelineResult:
    """
    Вход: RGB-изображение карточек.
    Выход: объект без фона, контуры Собеля/Превитта и наложение.
    """
    ops = get_backend(backend)
    rgb = resize_max_side(rgb)

    denoised = ops.median_filter(rgb, median_ksize)
    hsv = ops.rgb_to_hsv(denoised)
    mask_blue = ops.color_mask(hsv, *BLUE_HSV)
    mask_green = ops.color_mask(hsv, *GREEN_HSV)
    mask = np.where((mask_blue > 0) | (mask_green > 0), 255, 0).astype(np.uint8)
    mask = ops.max_filter(mask, 5)
    mask = ops.median_filter(mask, median_ksize)
    mask = keep_large_regions(mask)
    mask = fill_holes(mask)

    cleaned = rgb.copy()
    cleaned[mask == 0] = 0

    gray = ops.rgb_to_gray(cleaned)
    contrasted = ops.linear_contrast(gray)
    otsu_t = ops.otsu_threshold(gray)
    binary_otsu = ops.threshold_binarize(gray, otsu_t)

    _, _, sobel_mag = ops.sobel(cleaned)
    _, _, prewitt_mag = ops.prewitt(cleaned)

    overlay = cleaned.copy()
    overlay[sobel_mag > 40] = (255, 255, 0)

    return PipelineResult(
        original=rgb,
        denoised=denoised,
        hsv=hsv,
        mask=mask,
        cleaned=cleaned,
        gray=gray,
        contrasted=contrasted,
        binary_otsu=binary_otsu,
        sobel=sobel_mag,
        prewitt=prewitt_mag,
        overlay=overlay,
        hist_original=ops.histogram(rgb),
        hist_cleaned=ops.histogram(cleaned),
    )


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def load_rgb(path: str | Path) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("RGB"))


def save_rgb(path: str | Path, image: np.ndarray) -> None:
    from PIL import Image

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def list_input_images(folder: str | Path = "data/input") -> list[Path]:
    folder = Path(folder)
    files: list[Path] = []
    seen: set[bytes] = set()
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        digest = path.read_bytes()
        if digest in seen:
            continue
        seen.add(digest)
        files.append(path)
    return files
