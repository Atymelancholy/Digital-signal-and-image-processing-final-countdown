"""
Данные для ЛР4: разбор MNIST своими руками, толстые синтетические цифры,
перевод вырезанной фигуры в 28×28 как у MNIST. Без torchvision / keras.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

from src.lab3 import bilinear_sample

MNIST_DIR = Path(__file__).resolve().parents[1] / "data/mnist"
MNIST_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)
MNIST_MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)


def resize_gray(image: np.ndarray, nh: int, nw: int) -> np.ndarray:
    """Своё билинейное масштабирование серого кадра."""
    img = np.asarray(image, dtype=np.float64)
    h, w = img.shape[:2]
    if h == nh and w == nw:
        return img
    ys = np.clip((np.arange(nh) + 0.5) * h / nh - 0.5, 0, h - 1)
    xs = np.clip((np.arange(nw) + 0.5) * w / nw - 0.5, 0, w - 1)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return bilinear_sample(img, xx, yy, fill=0.0)


def shift_gray(image: np.ndarray, dy: float, dx: float) -> np.ndarray:
    h, w = image.shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    return bilinear_sample(image.astype(np.float64), xx - dx, yy - dy, fill=0.0)


def center_mass(image: np.ndarray) -> tuple[float, float]:
    img = np.clip(image, 0, None)
    total = float(img.sum())
    if total < 1e-8:
        h, w = img.shape
        return (h - 1) / 2.0, (w - 1) / 2.0
    ys, xs = np.indices(img.shape, dtype=np.float64)
    return float((ys * img).sum() / total), float((xs * img).sum() / total)


def fit_to_mnist(gray: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """
    Обрезка, вписывание в 20×20, холст 28×28, центр по центру масс —
    тот же канон, что у MNIST.
    """
    img = np.asarray(gray, dtype=np.float64)
    if img.ndim == 3:
        img = img.max(axis=2)
    if mask is not None:
        img = img.copy()
        img[~mask.astype(bool)] = 0.0
    peak = float(img.max())
    if peak > 1.5:
        img = img / 255.0
    elif peak > 1.0:
        img = img / peak
    img = np.clip(img, 0.0, 1.0)

    ys, xs = np.nonzero(img > 0.08)
    if len(xs) == 0:
        return np.zeros((28, 28), dtype=np.float32)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = img[y0:y1, x0:x1]
    ch, cw = crop.shape
    scale = 20.0 / max(ch, cw, 1)
    nh = max(1, int(round(ch * scale)))
    nw = max(1, int(round(cw * scale)))
    small = np.clip(resize_gray(crop, nh, nw), 0.0, 1.0)

    canvas = np.zeros((28, 28), dtype=np.float64)
    y_off = (28 - nh) // 2
    x_off = (28 - nw) // 2
    canvas[y_off : y_off + nh, x_off : x_off + nw] = small
    cy, cx = center_mass(canvas)
    canvas = shift_gray(canvas, 13.5 - cy, 13.5 - cx)
    return np.clip(canvas, 0.0, 1.0).astype(np.float32)


def rgb_to_mnist(rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if rgb.ndim == 3:
        gray = rgb.max(axis=2).astype(np.float64)
    else:
        gray = rgb.astype(np.float64)
    return fit_to_mnist(gray, mask)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "dsaip-lab4"})
    with urllib.request.urlopen(req, timeout=60) as resp, dest.open("wb") as out:
        out.write(resp.read())


def ensure_mnist(folder: Path = MNIST_DIR) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for name in MNIST_FILES:
        dest = folder / name
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        last_err: Exception | None = None
        for mirror in MNIST_MIRRORS:
            try:
                _download(mirror + name, dest)
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        if last_err is not None:
            raise RuntimeError(f"Не удалось скачать {name}: {last_err}") from last_err
    return folder


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        magic, n, rows, cols = struct.unpack(">IIII", fh.read(16))
        if magic != 2051:
            raise ValueError(f"Ожидался IDX images, magic={magic}")
        raw = fh.read()
    return np.frombuffer(raw, dtype=np.uint8).reshape(n, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as fh:
        magic, n = struct.unpack(">II", fh.read(8))
        if magic != 2049:
            raise ValueError(f"Ожидался IDX labels, magic={magic}")
        raw = fh.read()
    return np.frombuffer(raw, dtype=np.uint8).copy()


def load_mnist(folder: Path = MNIST_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    folder = ensure_mnist(folder)
    x_tr = _read_idx_images(folder / "train-images-idx3-ubyte.gz").astype(np.float32) / 255.0
    y_tr = _read_idx_labels(folder / "train-labels-idx1-ubyte.gz").astype(np.int32)
    x_te = _read_idx_images(folder / "t10k-images-idx3-ubyte.gz").astype(np.float32) / 255.0
    y_te = _read_idx_labels(folder / "t10k-labels-idx1-ubyte.gz").astype(np.int32)
    return x_tr, y_tr, x_te, y_te


def _paint_polyline(canvas: np.ndarray, pts: np.ndarray, radius: float) -> None:
    h, w = canvas.shape
    yy, xx = np.indices((h, w), dtype=np.float64)
    dmin = np.full((h, w), 1e9, dtype=np.float64)
    for (y0, x0), (y1, x1) in zip(pts[:-1], pts[1:]):
        vx = float(x1 - x0)
        vy = float(y1 - y0)
        length = vx * vx + vy * vy + 1e-9
        t = np.clip(((xx - x0) * vx + (yy - y0) * vy) / length, 0.0, 1.0)
        dist = np.hypot(xx - (x0 + t * vx), yy - (y0 + t * vy))
        dmin = np.minimum(dmin, dist)
    alpha = np.clip(radius + 0.85 - dmin, 0.0, 1.0)
    np.maximum(canvas, alpha, out=canvas)


def _ellipse(cy: float, cx: float, ry: float, rx: float, n: int = 28) -> np.ndarray:
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=True)
    return np.stack([cy + ry * np.sin(ang), cx + rx * np.cos(ang)], axis=1)


def _jitter(pts: np.ndarray, rng: np.random.Generator, scale: float = 0.35) -> np.ndarray:
    return pts + rng.normal(0.0, scale, pts.shape)


def _digit_strokes(digit: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Ломаные толстой пластиковой цифры в координатах 28×28."""
    if digit == 0:
        return [_ellipse(14.0, 14.0, 8.2, 5.6)]
    if digit == 1:
        serif = np.array([[7.5, 10.5], [5.8, 14.0], [22.2, 14.0]])
        return [serif]
    if digit == 2:
        return [
            np.array([[7.0, 8.0], [6.2, 18.5], [10.5, 20.5], [16.5, 8.0], [22.0, 7.5], [22.0, 20.5]]),
        ]
    if digit == 3:
        # Две открытые слева дуги, а не два прямоугольника.
        a = np.linspace(-2.15, 2.15, 32)
        return [np.stack([cy + 4.5 * np.sin(a), 12.5 + 5.5 * np.cos(a)], axis=1)
                for cy in (9.5, 18.5)]
    if digit == 4:
        return [
            np.array([[6.5, 16.5], [16.5, 7.5], [16.5, 21.0]]),
            np.array([[6.5, 17.5], [22.0, 17.5]]),
        ]
    if digit == 5:
        return [
            np.array([[6.5, 20.0], [6.5, 8.0], [13.0, 8.0], [13.2, 18.5], [21.5, 19.0], [22.0, 8.5]]),
        ]
    if digit == 6:
        return [
            np.array([[6.5, 17.5], [7.0, 8.5], [21.5, 8.5]]),
            _ellipse(16.8, 14.2, 5.0, 5.2),
        ]
    if digit == 7:
        return [
            np.array([[6.2, 7.2], [6.2, 21.0], [21.8, 14.2]]),
        ]
    if digit == 8:
        return [_ellipse(9.4, 14.0, 4.6, 5.0), _ellipse(18.2, 14.0, 5.0, 5.3)]
    if digit == 9:
        return [
            _ellipse(11.0, 14.0, 5.0, 5.2),
            np.array([[11.0, 19.2], [21.6, 18.8], [21.2, 8.5]]),
        ]
    return [_ellipse(14.0, 14.0, 7.0, 5.0)]


def make_synthetic_digit(digit: int, rng: np.random.Generator) -> np.ndarray:
    canvas = np.zeros((28, 28), dtype=np.float64)
    radius = float(rng.uniform(0.8, 1.8))
    for stroke in _digit_strokes(int(digit), rng):
        _paint_polyline(canvas, _jitter(stroke, rng, 0.15), radius)
    # Нормализация совпадает с подготовкой реальных карточек.
    canvas = fit_to_mnist(canvas)
    yy, xx = np.indices((28, 28), dtype=np.float64)
    angle = rng.uniform(-0.20, 0.20)
    scale_x, scale_y = rng.uniform(0.88, 1.12, 2)
    dx, dy = (xx - 13.5) / scale_x, (yy - 13.5) / scale_y
    xs = dx * np.cos(angle) + dy * np.sin(angle) + 13.5
    ys = -dx * np.sin(angle) + dy * np.cos(angle) + 13.5
    return np.clip(bilinear_sample(canvas, xs, ys), 0, 1).astype(np.float32)


def make_synthetic_set(n_per_class: int, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    images = []
    labels = []
    for d in range(10):
        for _ in range(n_per_class):
            images.append(make_synthetic_digit(d, rng))
            labels.append(d)
    return np.stack(images), np.asarray(labels, dtype=np.int32)


def augment_images(images: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Пакетный сдвиг с билинейной интерполяцией и нулевым фоном."""
    n, h, w = images.shape
    shifts = rng.uniform(-2, 2, (n, 2))
    shifts[rng.random(n) > 0.65] = 0
    yy, xx = np.indices((h, w), dtype=np.float32)
    x = xx[None] - shifts[:, 1, None, None]
    y = yy[None] - shifts[:, 0, None, None]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    wx, wy = x - x0, y - y0
    out = np.zeros_like(images, dtype=np.float32)
    for ox, oy, weight in [(0, 0, (1-wx)*(1-wy)), (1, 0, wx*(1-wy)),
                           (0, 1, (1-wx)*wy), (1, 1, wx*wy)]:
        xi, yi = x0 + ox, y0 + oy
        valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        out += (images[np.arange(n)[:, None, None], np.clip(yi, 0, h-1),
                       np.clip(xi, 0, w-1)] * weight * valid).astype(np.float32)
    return out


def flatten(images: np.ndarray) -> np.ndarray:
    return images.reshape(len(images), -1).astype(np.float32)
