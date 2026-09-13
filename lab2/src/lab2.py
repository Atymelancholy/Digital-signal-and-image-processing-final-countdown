"""
Лабораторная №2: разделение объектов, признаки и кластеризация.
Авторские формулы из методички (стр. 26–29): связные области, площадь,
периметр, центр масс, компактность, эксцентриситет; k-means по признакам.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.pipeline import process

CLASS_RGB = {
    0: (220, 35, 35),    # класс 1 — красный
    1: (35, 80, 230),    # класс 2 — синий
    2: (35, 200, 55),    # класс 3 — зелёный
}
TILE_CONTOUR = (0, 220, 255)


@dataclass
class ObjectFeat:
    label: int
    area: int
    perimeter: int
    cx: float
    cy: float
    compactness: float
    eccentricity: float
    holes: int
    aspect: float
    extent: float
    hu1: float
    hu2: float
    mask: np.ndarray


@dataclass
class Lab2Result:
    original: np.ndarray
    cleaned: np.ndarray
    mask: np.ndarray
    eroded: np.ndarray
    distance: np.ndarray
    tiles: np.ndarray
    tiles_color: np.ndarray
    inner: np.ndarray
    colored: np.ndarray
    objects: list[ObjectFeat]
    classes: np.ndarray


def erode_binary(mask: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    """Эрозия: объект сжимается, слипшиеся фигуры расходятся."""
    kernel = np.ones((ksize, ksize), np.uint8)
    out = (mask > 0).astype(np.uint8) * 255
    return cv2.erode(out, kernel, iterations=iterations)


def dilate_binary(mask: np.ndarray, ksize: int = 3, iterations: int = 1) -> np.ndarray:
    kernel = np.ones((ksize, ksize), np.uint8)
    out = (mask > 0).astype(np.uint8) * 255
    return cv2.dilate(out, kernel, iterations=iterations)


def morphological_close(mask: np.ndarray, ksize: int = 5) -> np.ndarray:
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.morphologyEx((mask > 0).astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)


def distance_3d(mask: np.ndarray) -> np.ndarray:
    """3D-представление: высота = расстояние до фона (карта расстояний)."""
    binary = (mask > 0).astype(np.uint8)
    return cv2.distanceTransform(binary, cv2.DIST_L2, 5)


def label_connected(binary: np.ndarray) -> np.ndarray:
    """
    Итеративная разметка 4-связных областей (алгоритм последовательного
    сканирования из методички, п. 2.1) с объединением эквивалентных меток.
    """
    img = (binary > 0).astype(np.uint8)
    h, w = img.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent = [0]

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    n = 0
    for y in range(h):
        row = img[y]
        if not row.any():
            continue
        prev = labels[y - 1] if y else None
        out = labels[y]
        for x in range(w):
            if row[x] == 0:
                continue
            neigh = []
            if x and out[x - 1]:
                neigh.append(int(out[x - 1]))
            if prev is not None:
                if prev[x]:
                    neigh.append(int(prev[x]))
                if x and prev[x - 1]:
                    neigh.append(int(prev[x - 1]))
                if x + 1 < w and prev[x + 1]:
                    neigh.append(int(prev[x + 1]))
            if not neigh:
                n += 1
                parent.append(n)
                out[x] = n
            else:
                mlab = min(neigh)
                out[x] = mlab
                for q in neigh:
                    if q != mlab:
                        union(mlab, q)

    remap = np.zeros(len(parent), dtype=np.int32)
    next_id = 0
    for i in range(1, len(parent)):
        r = find(i)
        if remap[r] == 0:
            next_id += 1
            remap[r] = next_id
        remap[i] = remap[r]
    return remap[labels]


def fill_holes_mask(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    h, w = m.shape
    flood = m.copy()
    ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(m, holes)


def perimeter_internal(mask: np.ndarray) -> int:
    """Внутренняя граница: пиксель объекта, у которого есть сосед-фон (4-связность)."""
    m = mask > 0
    up = np.roll(m, 1, 0)
    down = np.roll(m, -1, 0)
    left = np.roll(m, 1, 1)
    right = np.roll(m, -1, 1)
    up[0] = False
    down[-1] = False
    left[:, 0] = False
    right[:, -1] = False
    border = m & ~(up & down & left & right)
    return int(border.sum())


def hu_invariants(
    xs_c: np.ndarray,
    ys_c: np.ndarray,
    area: float,
    mu20: float,
    mu02: float,
    mu11: float,
) -> tuple[float, float]:
    """Моменты Ху φ1, φ2 — форма без учёта поворота и масштаба."""
    if area < 1:
        return 0.0, 0.0

    def eta(mu: float, p: int, q: int) -> float:
        return mu / (area ** ((p + q) / 2.0 + 1.0))

    n20, n02, n11 = eta(mu20, 2, 0), eta(mu02, 0, 2), eta(mu11, 1, 1)
    hu1 = n20 + n02
    hu2 = (n20 - n02) ** 2 + 4.0 * n11 * n11

    def signed_log(value: float) -> float:
        return float(np.sign(value) * np.log1p(abs(value) * 1e6))

    return signed_log(hu1), signed_log(hu2)


def object_features(mask: np.ndarray, label: int) -> ObjectFeat | None:
    """Площадь (2.1), центр масс (2.2), периметр, компактность, эксцентриситет."""
    ys, xs = np.nonzero(mask)
    area = int(len(xs))
    if area < 30:
        return None
    cx = float(xs.mean())
    cy = float(ys.mean())
    per = max(perimeter_internal(mask), 1)
    compactness = float(4.0 * np.pi * area / (per * per))
    xs_c = xs.astype(np.float64) - cx
    ys_c = ys.astype(np.float64) - cy
    mu20 = float(np.sum(xs_c * xs_c))
    mu02 = float(np.sum(ys_c * ys_c))
    mu11 = float(np.sum(xs_c * ys_c))
    eig = np.sort(np.linalg.eigvalsh(np.array([[mu20, mu11], [mu11, mu02]])))
    eccentricity = float(np.sqrt(max(0.0, 1.0 - eig[0] / max(eig[1], 1e-12))))
    holes = 1 if count_holes(mask) > 0 else 0
    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    aspect = float(max(bw, bh) / max(min(bw, bh), 1))
    extent = float(area / max(bw * bh, 1))
    hu1, hu2 = hu_invariants(xs_c, ys_c, float(area), mu20, mu02, mu11)
    return ObjectFeat(
        label=label,
        area=area,
        perimeter=per,
        cx=cx,
        cy=cy,
        compactness=compactness,
        eccentricity=eccentricity,
        holes=holes,
        aspect=aspect,
        extent=extent,
        hu1=hu1,
        hu2=hu2,
        mask=mask.astype(bool),
    )


def kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 50) -> np.ndarray:
    """k-means: случайные центры (k-means++), евклидова метрика, пересчёт средних."""
    n = len(X)
    k = max(1, min(k, n))
    rng = np.random.default_rng(seed)
    centers = [X[int(rng.integers(0, n))]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.stack(centers)[None, :, :]) ** 2).sum(-1), axis=1)
        p = d2 / max(float(d2.sum()), 1e-12)
        centers.append(X[int(rng.choice(n, p=p))])
    centers = np.stack(centers).astype(np.float64)
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        dist = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new = dist.argmin(axis=1).astype(np.int32)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            pts = X[labels == j]
            if len(pts):
                centers[j] = pts.mean(axis=0)
    return labels


def count_holes(mask: np.ndarray) -> int:
    m = (mask > 0).astype(np.uint8)
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None or len(hier) == 0:
        filled = fill_holes_mask(m)
        extra = int(filled.sum() // 255 - int(m.sum()))
        return 1 if extra > max(40, 0.08 * int(m.sum())) else 0
    return int(np.sum(hier[0][:, 3] >= 0))


def separate_tiles(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Эрозия по 3D-карте расстояний + watershed → отдельные плитки."""
    binary = (mask > 0).astype(np.uint8)
    dist = distance_3d(binary)
    peak = float(dist.max()) if dist.size else 0.0
    sure = (dist > 0.45 * max(peak, 1e-6)).astype(np.uint8)
    sure = cv2.morphologyEx(sure, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    seeds = label_connected(sure)
    markers = np.zeros(binary.shape, dtype=np.int32)
    markers[binary == 0] = 1
    markers[seeds > 0] = seeds[seeds > 0] + 1
    ws_src = cv2.cvtColor(binary * 255, cv2.COLOR_GRAY2BGR)
    ws = cv2.watershed(ws_src, markers)
    tiles = np.zeros_like(seeds, dtype=np.int32)
    inside = (ws > 1) & (binary > 0)
    tiles[inside] = ws[inside] - 1
    if int(tiles.max()) == 0:
        tiles = label_connected(binary)
    return dist, (sure * 255).astype(np.uint8), tiles


def color_labels(labels: np.ndarray) -> np.ndarray:
    h, w = labels.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    palette = np.array(
        [
            (230, 70, 70),
            (70, 120, 230),
            (50, 190, 90),
            (230, 180, 50),
            (180, 80, 210),
            (50, 200, 200),
            (230, 120, 50),
            (120, 80, 40),
        ],
        dtype=np.uint8,
    )
    for i in range(1, int(labels.max()) + 1):
        out[labels == i] = palette[(i - 1) % len(palette)]
    return out


def _bbox(mask: np.ndarray, pad: int = 2) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape[:2]
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, h)
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, w)
    return y0, y1, x0, x1


def inner_from_tile(cleaned: np.ndarray, tile: np.ndarray) -> np.ndarray:
    """Зелёная цифра/рисунок внутри плитки: HSV + канал G−B."""
    full = np.zeros(tile.shape, dtype=np.uint8)
    box = _bbox(tile)
    if box is None:
        return full
    y0, y1, x0, x1 = box
    tile_u8 = tile[y0:y1, x0:x1].astype(np.uint8)
    roi = cleaned[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    green = cv2.inRange(hsv, np.array([36, 50, 40]), np.array([95, 255, 255]))
    gb = roi[..., 1].astype(np.int16) - roi[..., 2].astype(np.int16)
    gb_mask = np.where((gb > 18) & (tile_u8 > 0), 255, 0).astype(np.uint8)
    inner = np.where(((green > 0) | (gb_mask > 0)) & (tile_u8 > 0), 255, 0).astype(np.uint8)
    inner = morphological_close(inner, 5)
    inner = np.where(tile_u8 > 0, inner, 0).astype(np.uint8)
    if int(inner.sum()) < 80:
        med = np.median(roi[tile_u8 > 0].reshape(-1, 3), axis=0) if tile_u8.any() else np.zeros(3)
        delta = np.abs(roi.astype(np.int16) - med.astype(np.int16)).sum(axis=-1)
        inner = np.where((delta > 55) & (tile_u8 > 0), 255, 0).astype(np.uint8)
        inner = morphological_close(inner, 5)
    labels = label_connected(inner)
    if int(labels.max()) == 0:
        full[y0:y1, x0:x1] = inner
        return full
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    keep = int(areas.argmax())
    full[y0:y1, x0:x1] = np.where(labels == keep, 255, 0).astype(np.uint8)
    return full


def map_classes(feats: list[ObjectFeat], raw: np.ndarray) -> np.ndarray:
    """
    Класс 1 (красный) — самые круглые (0/8),
    класс 2 (синий) — самые вытянутые (1),
    класс 3 (зелёный) — остальные (7).
    """
    k = int(raw.max()) + 1 if len(raw) else 0
    remaining = list(range(k))
    mapping: dict[int, int] = {}

    def mean_of(j: int, attr: str) -> float:
        vals = [getattr(f, attr) for f, c in zip(feats, raw) if c == j]
        return float(np.mean(vals)) if vals else 0.0

    if remaining:
        red = min(remaining, key=lambda j: mean_of(j, "eccentricity") - 0.8 * mean_of(j, "holes"))
        mapping[red] = 0
        remaining.remove(red)
    if remaining:
        blue = max(remaining, key=lambda j: mean_of(j, "eccentricity"))
        mapping[blue] = 1
        remaining.remove(blue)
    for j in remaining:
        mapping[j] = 2
    return np.array([mapping[int(c)] for c in raw], dtype=np.int32)


def paint_result(shape: tuple[int, int], tiles: np.ndarray, feats: list[ObjectFeat], classes: np.ndarray) -> np.ndarray:
    h, w = shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(1, int(tiles.max()) + 1):
        cnts, _ = cv2.findContours((tiles == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, TILE_CONTOUR, 2)
    for feat, cls in zip(feats, classes):
        filled = fill_holes_mask(feat.mask.astype(np.uint8) * 255) > 0
        if feat.holes:
            region = filled
        else:
            region = feat.mask
        out[region] = CLASS_RGB[int(cls) % 3]
    return out


def cluster_objects(feats: list[ObjectFeat], k: int = 3) -> np.ndarray:
    if not feats:
        return np.zeros(0, dtype=np.int32)
    X = np.array(
        [
            [f.compactness, f.eccentricity, f.holes, f.hu1, f.hu2]
            for f in feats
        ],
        dtype=np.float64,
    )
    std = X.std(axis=0)
    std[std < 1e-9] = 1.0
    Xn = (X - X.mean(axis=0)) / std
    raw = kmeans(Xn, k=min(k, len(feats)), seed=0)
    return map_classes(feats, raw)


def extract_inner_figures(cleaned: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, list[ObjectFeat]]:
    """Все внутренние фигуры на маске карточек (зелёный канал и G−B)."""
    hsv = cv2.cvtColor(cleaned, cv2.COLOR_RGB2HSV)
    green = cv2.inRange(hsv, np.array([36, 35, 30]), np.array([95, 255, 255]))
    gb = cleaned[..., 1].astype(np.int16) - cleaned[..., 2].astype(np.int16)
    inner = np.where(((green > 0) | (gb > 12)) & (mask > 0), 255, 0).astype(np.uint8)
    inner = morphological_close(inner, 5)
    inner = np.where(mask > 0, inner, 0).astype(np.uint8)
    labels = label_connected(inner)
    extracted: list[ObjectFeat] = []
    for i in range(1, int(labels.max()) + 1):
        feat = object_features(labels == i, i)
        if feat is not None:
            extracted.append(feat)
    if not extracted:
        return inner, []
    if len(extracted) >= 2:
        max_a = max(f.area for f in extracted)
        big = [
            f
            for f in extracted
            if f.area >= 0.18 * max_a and f.extent >= 0.14 and f.aspect < 4.0
        ]
        if big:
            extracted = big
    for i, f in enumerate(extracted, start=1):
        f.label = i
    shown = np.zeros_like(inner)
    for f in extracted:
        shown[f.mask] = 255
    return shown, extracted


def analyze(rgb: np.ndarray, backend: str = "library") -> Lab2Result:
    lab1 = process(rgb, backend=backend)
    dist, sure, tiles = separate_tiles(lab1.mask)
    inner_all, feats = extract_inner_figures(lab1.cleaned, lab1.mask)
    covered = np.zeros(lab1.mask.shape, dtype=bool)
    for f in feats:
        covered |= f.mask
    extra: list[ObjectFeat] = []
    for i in range(1, int(tiles.max()) + 1):
        tile = tiles == i
        if not tile.any():
            continue
        if covered[tile].mean() > 0.004:
            continue
        inner = inner_from_tile(lab1.cleaned, tile)
        feat = object_features(inner > 0, 0)
        if feat is None or feat.extent < 0.20 or feat.aspect >= 3.5 or feat.compactness < 0.15:
            continue
        extra.append(feat)
        inner_all = np.maximum(inner_all, inner)
        covered |= feat.mask
    feats.extend(extra)
    for i, f in enumerate(feats, start=1):
        f.label = i
    classes = cluster_objects(feats, k=3)
    colored = paint_result(lab1.cleaned.shape, tiles, feats, classes)
    return Lab2Result(
        original=lab1.original,
        cleaned=lab1.cleaned,
        mask=lab1.mask,
        eroded=sure,
        distance=dist,
        tiles=tiles,
        tiles_color=color_labels(tiles),
        inner=inner_all,
        colored=colored,
        objects=feats,
        classes=classes,
    )
