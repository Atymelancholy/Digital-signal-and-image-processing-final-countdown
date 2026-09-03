"""Те же алгоритмы через OpenCV (вариант B лабораторной)."""

from __future__ import annotations

import cv2
import numpy as np


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def histogram(image: np.ndarray) -> np.ndarray:
    gray = image if image.ndim == 2 else rgb_to_gray(image)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    return hist.flatten().astype(np.int64)


def linear_contrast(
    image: np.ndarray,
    gmin: float = 0.0,
    gmax: float = 255.0,
) -> np.ndarray:
    src = image.astype(np.float64)
    fmin, fmax = float(src.min()), float(src.max())
    if fmax - fmin < 1e-9:
        return image.copy()
    dst = (src - fmin) / (fmax - fmin) * (gmax - gmin) + gmin
    return np.clip(dst, 0, 255).astype(np.uint8)


def gamma_correction(image: np.ndarray, c: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    table = np.array(
        [np.clip(c * ((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)


def threshold_binarize(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    gray = image if image.ndim == 2 else rgb_to_gray(image)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def otsu_threshold(image: np.ndarray) -> int:
    gray = image if image.ndim == 2 else rgb_to_gray(image)
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(t)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def color_mask(
    hsv: np.ndarray,
    lower: tuple[int, int, int],
    upper: tuple[int, int, int],
) -> np.ndarray:
    return cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))


def median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.medianBlur(image, ksize)


def max_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    return cv2.dilate(image, kernel)


def sobel(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = image if image.ndim == 2 else rgb_to_gray(image)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    peak = mag.max()
    mag_u8 = np.zeros_like(gray, dtype=np.uint8) if peak < 1e-9 else np.clip(mag / peak * 255.0, 0, 255).astype(np.uint8)
    return gx, gy, mag_u8


def prewitt(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gray = image if image.ndim == 2 else rgb_to_gray(image)
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    gx = cv2.filter2D(gray.astype(np.float64), -1, kernel_x)
    gy = cv2.filter2D(gray.astype(np.float64), -1, kernel_y)
    mag = cv2.magnitude(gx, gy)
    peak = mag.max()
    mag_u8 = np.zeros_like(gray, dtype=np.uint8) if peak < 1e-9 else np.clip(mag / peak * 255.0, 0, 255).astype(np.uint8)
    return gx, gy, mag_u8
