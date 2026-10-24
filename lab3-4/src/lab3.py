"""
Лабораторная №3: поворот фигур в читаемый вид.
Своя реализация: ось инерции (моменты), поворот билинейной интерполяцией,
выбор 180° по профилю ширины. Без cv2.warpAffine / getRotationMatrix2D.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.lab2 import CLASS_RGB, Lab2Result, ObjectFeat, analyze


@dataclass
class UprightFigure:
    label: int
    cls: int
    angle_deg: float
    before: np.ndarray
    after: np.ndarray
    before_mask: np.ndarray
    after_mask: np.ndarray
    cx: float
    cy: float


@dataclass
class Lab3Result:
    lab2: Lab2Result
    figures: list[UprightFigure]
    method1: np.ndarray
    method1_color: np.ndarray


def principal_axis_angle(mask: np.ndarray) -> tuple[float, float, float]:
    """Угол главной оси инерции (рад) и центр масс. Формулы как в ЛР2, п. 2.2."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0.0, 0.0, 0.0
    cx = float(xs.mean())
    cy = float(ys.mean())
    xs_c = xs.astype(np.float64) - cx
    ys_c = ys.astype(np.float64) - cy
    mu20 = float(np.sum(xs_c * xs_c))
    mu02 = float(np.sum(ys_c * ys_c))
    mu11 = float(np.sum(xs_c * ys_c))
    theta = 0.5 * float(np.arctan2(2.0 * mu11, mu20 - mu02))
    return theta, cx, cy


def bilinear_sample(image: np.ndarray, x: np.ndarray, y: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Билинейная интерполяция: обратное отображение, без библиотечного warp."""
    h, w = image.shape[:2]
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = x - x0.astype(np.float64)
    wy = y - y0.astype(np.float64)
    ok = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)

    def grab(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            out = np.full(ix.shape, fill, dtype=np.float64)
            inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            out[inside] = image[iy[inside], ix[inside]]
            return out
        out = np.full(ix.shape + (image.shape[2],), fill, dtype=np.float64)
        inside = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        out[inside] = image[iy[inside], ix[inside]]
        return out

    w00 = (1.0 - wx) * (1.0 - wy)
    w10 = wx * (1.0 - wy)
    w01 = (1.0 - wx) * wy
    w11 = wx * wy
    if image.ndim == 2:
        acc = (
            grab(x0, y0) * w00
            + grab(x1, y0) * w10
            + grab(x0, y1) * w01
            + grab(x1, y1) * w11
        )
        acc[~ok] = fill
        return acc
    acc = (
        grab(x0, y0) * w00[..., None]
        + grab(x1, y0) * w10[..., None]
        + grab(x0, y1) * w01[..., None]
        + grab(x1, y1) * w11[..., None]
    )
    acc[~ok] = fill
    return acc


def rotate_around(
    image: np.ndarray,
    angle_deg: float,
    cx: float,
    cy: float,
    out_hw: tuple[int, int] | None = None,
    ncx: float | None = None,
    ncy: float | None = None,
    fill: float = 0.0,
) -> np.ndarray:
    """
    Поворот вокруг (cx, cy) на angle_deg (против часовой на экране с y вниз).
    Обратное отображение + билинейная интерполяция.
    """
    h, w = image.shape[:2]
    oh, ow = out_hw if out_hw is not None else (h, w)
    if ncx is None:
        ncx = cx
    if ncy is None:
        ncy = cy
    ang = np.deg2rad(angle_deg)
    cos_a = float(np.cos(ang))
    sin_a = float(np.sin(ang))
    yy, xx = np.indices((oh, ow), dtype=np.float64)
    dx = xx - ncx
    dy = yy - ncy
    xs = dx * cos_a + dy * sin_a + cx
    ys = -dx * sin_a + dy * cos_a + cy
    sampled = bilinear_sample(image, xs, ys, fill=fill)
    if image.dtype == np.bool_ or image.max() <= 1:
        return sampled > 0.5
    if np.issubdtype(image.dtype, np.integer):
        return np.clip(np.rint(sampled), 0, 255).astype(np.uint8)
    return sampled.astype(image.dtype)


def bbox(mask: np.ndarray, pad: int = 0) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape[:2]
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, h)
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, w)
    return y0, y1, x0, x1


def trim(image: np.ndarray, mask: np.ndarray, pad: int = 6) -> tuple[np.ndarray, np.ndarray]:
    box = bbox(mask, pad=pad)
    if box is None:
        return image, mask
    y0, y1, x0, x1 = box
    return image[y0:y1, x0:x1].copy(), mask[y0:y1, x0:x1].copy()


def row_widths(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    widths = np.zeros(h, dtype=np.float64)
    for y in range(h):
        xs = np.flatnonzero(mask[y])
        if xs.size:
            widths[y] = float(xs[-1] - xs[0] + 1)
    return widths


def wrap_signed(angle_deg: float) -> float:
    """Кратчайший угол в (−180, 180]. 358° → −2°."""
    a = (float(angle_deg) + 180.0) % 360.0 - 180.0
    if abs(a) < 0.5:
        return 0.0
    return a


def _rotate_mask(mask: np.ndarray, angle: float, cx: float, cy: float, diag: int) -> np.ndarray:
    ncx = ncy = (diag - 1) / 2.0
    rot = rotate_around(mask.astype(np.uint8), angle, cx, cy, (diag, diag), ncx, ncy)
    return rot > 0


def bar_align_delta(rot_mask: np.ndarray) -> float:
    """
    Если сверху широкая перекладина (семёрка), довернуть по её верхней кромке,
    чтобы она стала горизонтальной. Почти стоячая «7» больше не остаётся как есть.
    Единицу и петлю «8» не трогаем: узкая или почти ровная крыша.
    """
    box = bbox(rot_mask, pad=0)
    if box is None:
        return 0.0
    y0, y1, x0, x1 = box
    patch = rot_mask[y0:y1, x0:x1]
    h, w = patch.shape
    if h < 16 or w < 12:
        return 0.0

    top_x: list[int] = []
    top_y: list[int] = []
    for x in range(w):
        col = np.flatnonzero(patch[:, x])
        if col.size:
            top_x.append(x)
            top_y.append(int(col[0]))
    if len(top_x) < 12:
        return 0.0
    xs_all = np.asarray(top_x, dtype=np.float64)
    ys_all = np.asarray(top_y, dtype=np.float64)
    roof = float(ys_all.min())
    keep = ys_all <= roof + max(6.0, 0.14 * h)
    if int(keep.sum()) < 10:
        return 0.0
    xs = xs_all[keep]
    ys = ys_all[keep]
    span = float(xs.max() - xs.min() + 1.0)
    if span < 0.55 * w:
        return 0.0

    A = np.vstack([xs, np.ones_like(xs)]).T
    slope, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
    pred = slope * xs + np.mean(ys - slope * xs)
    rms = float(np.sqrt(np.mean((ys - pred) ** 2)))
    if rms > 0.07 * h:
        return 0.0

    deg = float(np.degrees(np.arctan(slope)))
    if abs(deg) < 2.0 or abs(deg) > 18.0:
        return 0.0
    return float(np.clip(-deg, -16.0, 16.0))


def upright_angle(mask: np.ndarray) -> float:
    """
    Угол, после которого фигура стоит вертикально и «головой» вверх:
    широкая часть (перекладина 7, петля 8) ближе к верху.
    Почти стоячие цифры доворачиваются по перекладине, а не остаются как есть.
    """
    theta, cx, cy = principal_axis_angle(mask)
    angle = 90.0 - np.degrees(theta)

    h, w = mask.shape
    diag = int(np.ceil(np.hypot(h, w))) + 4
    rot_m = _rotate_mask(mask, angle, cx, cy, diag)
    box = bbox(rot_m, pad=0)
    if box is None:
        return 0.0
    y0, y1, x0, x1 = box
    bh, bw = y1 - y0, x1 - x0
    if bw > bh * 1.08:
        angle += 90.0
        rot_m = _rotate_mask(mask, angle, cx, cy, diag)
        box = bbox(rot_m, pad=0)
        if box is None:
            return wrap_signed(angle)
        y0, y1, x0, x1 = box

    patch = rot_m[y0:y1, x0:x1]
    widths = row_widths(patch)
    if widths.max() >= 1:
        peak = int(np.argmax(widths))
        if peak > patch.shape[0] * 0.5:
            angle += 180.0
            rot_m = _rotate_mask(mask, angle, cx, cy, diag)

    angle += bar_align_delta(rot_m)
    return wrap_signed(angle)


def extract_rgb_patch(cleaned: np.ndarray, mask: np.ndarray, pad: int = 8) -> tuple[np.ndarray, np.ndarray, int, int]:
    box = bbox(mask, pad=pad)
    if box is None:
        return np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1), np.uint8), 0, 0
    y0, y1, x0, x1 = box
    local = mask[y0:y1, x0:x1]
    patch = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.uint8)
    patch[local] = cleaned[y0:y1, x0:x1][local]
    return patch, local.astype(np.uint8) * 255, x0, y0


def rotate_patch(patch: np.ndarray, mask: np.ndarray, angle_deg: float, cx: float, cy: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape[:2]
    diag = int(np.ceil(np.hypot(h, w))) + 8
    ncx = ncy = (diag - 1) / 2.0
    rot_rgb = rotate_around(patch, angle_deg, cx, cy, (diag, diag), ncx, ncy)
    rot_m = rotate_around(mask, angle_deg, cx, cy, (diag, diag), ncx, ncy)
    bin_m = rot_m > 0
    rot_rgb[~bin_m] = 0
    return trim(rot_rgb, bin_m, pad=8)


def square_pad(image: np.ndarray, side: int | None = None) -> np.ndarray:
    h, w = image.shape[:2]
    side = int(max(side or 0, h, w, 1))
    canvas = np.zeros((side, side, 3), dtype=np.uint8) if image.ndim == 3 else np.zeros((side, side), dtype=image.dtype)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    canvas[y0 : y0 + h, x0 : x0 + w] = image
    return canvas


def process_figures(lab2: Lab2Result) -> list[UprightFigure]:
    out: list[UprightFigure] = []
    for feat, cls in zip(lab2.objects, lab2.classes):
        patch, local, x0, y0 = extract_rgb_patch(lab2.cleaned, feat.mask, pad=4)
        _, cx_g, cy_g = principal_axis_angle(feat.mask)
        angle = upright_angle(feat.mask)
        cx = cx_g - x0
        cy = cy_g - y0
        after_rgb, after_m = rotate_patch(patch, local, angle, cx, cy)
        out.append(
            UprightFigure(
                label=feat.label,
                cls=int(cls) % 3,
                angle_deg=angle,
                before=patch,
                after=after_rgb,
                before_mask=local > 0,
                after_mask=after_m,
                cx=cx_g,
                cy=cy_g,
            )
        )
    return out


def compose_method1(lab2: Lab2Result, figures: list[UprightFigure], colored: bool) -> np.ndarray:
    """Способ 1: каждая цифра поворачивается на месте вокруг своего центра."""
    h, w = lab2.cleaned.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for feat, fig in zip(lab2.objects, figures):
        src = np.zeros((h, w, 3), dtype=np.uint8)
        if colored:
            src[feat.mask] = CLASS_RGB[fig.cls]
        else:
            src[feat.mask] = lab2.cleaned[feat.mask]
        rot = rotate_around(src, fig.angle_deg, fig.cx, fig.cy, (h, w), fig.cx, fig.cy)
        mrot = rotate_around(
            feat.mask.astype(np.uint8) * 255,
            fig.angle_deg,
            fig.cx,
            fig.cy,
            (h, w),
            fig.cx,
            fig.cy,
        )
        on = mrot > 127
        canvas[on] = rot[on]
    return canvas


def run_lab3(rgb: np.ndarray) -> Lab3Result:
    lab2 = analyze(rgb, backend="library")
    figures = process_figures(lab2)
    method1 = compose_method1(lab2, figures, colored=False)
    method1_color = compose_method1(lab2, figures, colored=True)
    return Lab3Result(lab2=lab2, figures=figures, method1=method1, method1_color=method1_color)
