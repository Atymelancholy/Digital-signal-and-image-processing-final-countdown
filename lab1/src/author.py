"""
Авторские реализации алгоритмов лабораторной №1.
Только NumPy: без OpenCV, scikit-image и готовых фильтров.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    """Полутоновое изображение по формуле из методички: Y = 0.3R + 0.59G + 0.11B."""
    rgb = rgb.astype(np.float64)
    y = 0.3 * rgb[..., 0] + 0.59 * rgb[..., 1] + 0.11 * rgb[..., 2]
    return np.clip(y, 0, 255).astype(np.uint8)


def histogram(image: np.ndarray) -> np.ndarray:
    """Гистограмма яркости: 256 счётчиков для значений 0..255."""
    gray = image if image.ndim == 2 else rgb_to_gray(image)
    hist = np.zeros(256, dtype=np.int64)
    flat = gray.reshape(-1)
    np.add.at(hist, flat, 1)
    return hist


def linear_contrast(
    image: np.ndarray,
    gmin: float = 0.0,
    gmax: float = 255.0,
) -> np.ndarray:
    """Линейное контрастирование (формула 1.4 методички)."""
    src = image.astype(np.float64)
    fmin = float(src.min())
    fmax = float(src.max())
    if fmax - fmin < 1e-9:
        return image.copy()
    dst = (src - fmin) / (fmax - fmin) * (gmax - gmin) + gmin
    return np.clip(dst, 0, 255).astype(np.uint8)


def gamma_correction(image: np.ndarray, c: float = 1.0, gamma: float = 1.0) -> np.ndarray:
    """Гамма-коррекция: g = c * f^γ, яркость нормируется в [0, 1]."""
    src = image.astype(np.float64) / 255.0
    dst = c * np.power(src, gamma)
    return np.clip(dst * 255.0, 0, 255).astype(np.uint8)


def threshold_binarize(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """Пороговая бинаризация (препарирование, рис. 1.3 а)."""
    gray = image if image.ndim == 2 else rgb_to_gray(image)
    return np.where(gray >= threshold, 255, 0).astype(np.uint8)


def otsu_threshold(image: np.ndarray) -> int:
    """Автоматический порог Отсу по гистограмме яркости."""
    gray = image if image.ndim == 2 else rgb_to_gray(image)
    hist = histogram(gray).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128
    sum_total = np.dot(np.arange(256), hist)
    w0 = 0.0
    sum0 = 0.0
    best_sigma = -1.0
    best_t = 0
    for t in range(256):
        w0 += hist[t]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += t * hist[t]
        m0 = sum0 / w0
        m1 = (sum_total - sum0) / w1
        sigma = w0 * w1 * (m0 - m1) ** 2
        if sigma > best_sigma:
            best_sigma = sigma
            best_t = t
    return int(best_t)


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """RGB → HSV в шкале OpenCV: H ∈ [0, 180], S,V ∈ [0, 255]."""
    rgb_f = rgb.astype(np.float64) / 255.0
    r, g, b = rgb_f[..., 0], rgb_f[..., 1], rgb_f[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    v = maxc
    s = np.zeros_like(maxc)
    nonzero = maxc > 1e-12
    s[nonzero] = delta[nonzero] / maxc[nonzero]

    h = np.zeros_like(maxc)
    mask = delta > 1e-12
    r_max = mask & (maxc == r)
    g_max = mask & (maxc == g) & ~r_max
    b_max = mask & (maxc == b) & ~r_max & ~g_max
    h[r_max] = ((g[r_max] - b[r_max]) / delta[r_max]) % 6.0
    h[g_max] = ((b[g_max] - r[g_max]) / delta[g_max]) + 2.0
    h[b_max] = ((r[b_max] - g[b_max]) / delta[b_max]) + 4.0
    h = h * 30.0  # 60° / 2, чтобы H был в [0, 180]

    hsv = np.stack(
        [
            np.clip(h, 0, 180),
            np.clip(s * 255.0, 0, 255),
            np.clip(v * 255.0, 0, 255),
        ],
        axis=-1,
    )
    return hsv.astype(np.uint8)


def color_mask(
    hsv: np.ndarray,
    lower: tuple[int, int, int],
    upper: tuple[int, int, int],
) -> np.ndarray:
    """Бинаризация по диапазону HSV (аналог cv2.inRange)."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    inside = (
        (h >= lower[0])
        & (h <= upper[0])
        & (s >= lower[1])
        & (s <= upper[1])
        & (v >= lower[2])
        & (v <= upper[2])
    )
    return np.where(inside, 255, 0).astype(np.uint8)


def median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Медианный фильтр. Краевые пиксели дополняются зеркальным отражением."""
    if ksize % 2 == 0 or ksize < 1:
        raise ValueError("ksize должен быть нечётным положительным")
    pad = ksize // 2
    if image.ndim == 2:
        padded = np.pad(image, pad, mode="reflect")
        windows = sliding_window_view(padded, (ksize, ksize))
        return np.median(windows, axis=(-1, -2)).astype(np.uint8)

    channels = []
    for c in range(image.shape[2]):
        channels.append(median_filter(image[..., c], ksize))
    return np.stack(channels, axis=-1)


def max_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Max-фильтр из методички: расширяет светлые области (дыры в маске слегка зарастают)."""
    if ksize % 2 == 0 or ksize < 1:
        raise ValueError("ksize должен быть нечётным положительным")
    pad = ksize // 2
    if image.ndim == 2:
        padded = np.pad(image, pad, mode="reflect")
        windows = sliding_window_view(padded, (ksize, ksize))
        return np.max(windows, axis=(-1, -2)).astype(np.uint8)
    channels = [max_filter(image[..., c], ksize) for c in range(image.shape[2])]
    return np.stack(channels, axis=-1)


def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Линейная свёртка с зеркальным дополнением границы."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image.astype(np.float64), ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    out = np.zeros(image.shape, dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i : i + image.shape[0], j : j + image.shape[1]]
    return out


def sobel(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Оператор Собеля: градиенты Gx, Gy и величина sqrt(Gx² + Gy²)."""
    gray = image.astype(np.float64) if image.ndim == 2 else rgb_to_gray(image).astype(np.float64)
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    gx = convolve2d(gray, kernel_x)
    gy = convolve2d(gray, kernel_y)
    mag = np.sqrt(gx * gx + gy * gy)
    mag_u8 = _normalize_abs(mag)
    return gx, gy, mag_u8


def prewitt(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Оператор Превитта (ядра из методички)."""
    gray = image.astype(np.float64) if image.ndim == 2 else rgb_to_gray(image).astype(np.float64)
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    gx = convolve2d(gray, kernel_x)
    gy = convolve2d(gray, kernel_y)
    mag = np.sqrt(gx * gx + gy * gy)
    return gx, gy, _normalize_abs(mag)


def _normalize_abs(values: np.ndarray) -> np.ndarray:
    mag = np.abs(values)
    peak = mag.max()
    if peak < 1e-9:
        return np.zeros_like(values, dtype=np.uint8)
    return np.clip(mag / peak * 255.0, 0, 255).astype(np.uint8)
