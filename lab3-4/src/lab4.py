"""
Лабораторная №4: распознавание цифр своей нейронной сетью.
Сеть — MLP на NumPy. Вход — повёрнутые фигуры ЛР3, плюс проверка переворота 180°.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.lab3 import Lab3Result, run_lab3, square_pad
from src.lab4_data import (
    augment_images,
    flatten,
    load_mnist,
    make_synthetic_set,
    rgb_to_mnist,
)
from src.nn import MLP
from src.pipeline import list_input_images, load_rgb

WEIGHTS_PATH = Path("data/models/lab4_mlp.npz")
ARCH = [784, 256, 128, 10]
LAB2_TO_DIGIT = {0: 8, 1: 1, 2: 7}


def _border_background(mask: np.ndarray) -> np.ndarray:
    """Фон, достижимый с края (своя заливка, без cv2.floodFill)."""
    ink = mask.astype(bool)
    h, w = ink.shape
    reach = np.zeros((h, w), dtype=bool)
    stack = []
    for x in range(w):
        if not ink[0, x]:
            stack.append((0, x))
        if not ink[h - 1, x]:
            stack.append((h - 1, x))
    for y in range(h):
        if not ink[y, 0]:
            stack.append((y, 0))
        if not ink[y, w - 1]:
            stack.append((y, w - 1))
    while stack:
        y, x = stack.pop()
        if y < 0 or y >= h or x < 0 or x >= w or reach[y, x] or ink[y, x]:
            continue
        reach[y, x] = True
        stack.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
    return reach


def count_interior_holes(mask: np.ndarray) -> int:
    """Сколько замкнутых дыр внутри штриха (4-связность)."""
    ink = mask.astype(bool)
    if ink.sum() < 20:
        return 0
    interior = ~ink & ~_border_background(ink)
    if not interior.any():
        return 0
    seen = np.zeros_like(interior)
    holes = 0
    ys, xs = np.nonzero(interior)
    min_area = max(12, int(0.004 * ink.size))
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        area = 0
        while stack:
            y, x = stack.pop()
            if y < 0 or y >= ink.shape[0] or x < 0 or x >= ink.shape[1]:
                continue
            if seen[y, x] or not interior[y, x]:
                continue
            seen[y, x] = True
            area += 1
            stack.extend(((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
        if area >= min_area:
            holes += 1
    return holes


def waist_ratio(mask: np.ndarray) -> float:
    """Отношение ширины в середине к плечам. У «8» узкая талия, у «0» нет."""
    ink = mask.astype(bool)
    filled = ink | (~ink & ~_border_background(ink))
    ys, xs = np.nonzero(filled)
    if len(xs) < 20:
        return 1.0
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    patch = filled[y0:y1]
    h = patch.shape[0]
    widths = np.zeros(h, dtype=np.float64)
    for y in range(h):
        row = np.flatnonzero(patch[y])
        if row.size:
            widths[y] = float(row[-1] - row[0] + 1)
    top = float(widths[int(0.12 * h) : max(int(0.34 * h), 1)].max() or 1.0)
    bot = float(widths[int(0.66 * h) : max(int(0.88 * h), int(0.66 * h) + 1)].max() or 1.0)
    mid = float(widths[int(0.42 * h) : max(int(0.58 * h), int(0.42 * h) + 1)].min() or 0.0)
    return mid / max(0.5 * (top + bot), 1.0)


def classify_zero_or_eight(mask: np.ndarray) -> int | None:
    """
    0 — одно отверстие и почти овальный силуэт.
    8 — две дыры или явная перетяжка посередине.
    """
    holes = count_interior_holes(mask)
    if holes >= 2:
        return 8
    if holes == 1:
        return 8 if waist_ratio(mask) < 0.58 else 0
    return None


def _relabel_loop(pred: int, probs: np.ndarray, mask: np.ndarray) -> tuple[int, float, np.ndarray]:
    """Если сеть путает 0 и 8, поправляем по форме."""
    loop_mass = float(probs[0] + probs[8])
    if pred not in (0, 6, 8, 9) and loop_mass < 0.35:
        return pred, float(probs[pred]), probs
    geom = classify_zero_or_eight(mask)
    if geom is None:
        return pred, float(probs[pred]), probs
    if pred == geom:
        return pred, float(probs[pred]), probs
    out = probs.astype(np.float64).copy()
    other = 8 if geom == 0 else 0
    out[other] *= 0.04
    out[geom] = max(out[geom], loop_mass, 0.96)
    out = out / out.sum()
    return geom, float(out[geom]), out.astype(np.float32)


def label_from_figure(fig) -> int:
    """Метка для дообучения: круглые делим на 0 и 8, не всё подряд в «8»."""
    digit = LAB2_TO_DIGIT.get(int(fig.cls) % 3, 8)
    if digit == 8:
        geom = classify_zero_or_eight(fig.after_mask)
        if geom is not None:
            return geom
    return digit


@dataclass
class DigitGuess:
    label: int
    pred: int
    conf: float
    pred_flip: int
    conf_flip: float
    probs: np.ndarray
    probs_flip: np.ndarray
    color: np.ndarray
    mnist: np.ndarray
    mnist_flip: np.ndarray


@dataclass
class Lab4Result:
    lab3: Lab3Result
    guesses: list[DigitGuess]
    model: MLP


def mnist_probs(model: MLP, image28: np.ndarray) -> np.ndarray:
    x = flatten(image28[None, ...])
    return model.predict_proba(x)[0]


def guess_figure(model: MLP, rgb: np.ndarray, mask: np.ndarray, label: int) -> DigitGuess:
    mnist = rgb_to_mnist(rgb, mask)
    flipped = mnist[::-1, ::-1].copy()
    probs = mnist_probs(model, mnist)
    probs_f = mnist_probs(model, flipped)
    pred = int(probs.argmax())
    pred_f = int(probs_f.argmax())
    pred, conf, probs = _relabel_loop(pred, probs, mask)
    pred_f, conf_f, probs_f = _relabel_loop(pred_f, probs_f, mask[::-1, ::-1])
    return DigitGuess(
        label=label,
        pred=pred,
        conf=conf,
        pred_flip=pred_f,
        conf_flip=conf_f,
        probs=probs,
        probs_flip=probs_f,
        color=rgb,
        mnist=mnist,
        mnist_flip=flipped,
    )


def run_lab4(rgb: np.ndarray, model: MLP | None = None) -> Lab4Result:
    if model is None:
        model = ensure_model()
    lab3 = run_lab3(rgb)
    guesses = [guess_figure(model, fig.after, fig.after_mask, fig.label) for fig in lab3.figures]
    return Lab4Result(lab3=lab3, guesses=guesses, model=model)


def _accuracy(model: MLP, images: np.ndarray, labels: np.ndarray, batch: int = 512) -> float:
    correct = 0
    n = len(images)
    for i in range(0, n, batch):
        pred = model.predict(flatten(images[i : i + batch]))
        correct += int((pred == labels[i : i + batch]).sum())
    return correct / max(n, 1)


def _iterate_minibatches(
    images: np.ndarray,
    labels: np.ndarray,
    batch: int,
    rng: np.random.Generator,
    shuffle: bool = True,
):
    idx = rng.permutation(len(images)) if shuffle else np.arange(len(images))
    for start in range(0, len(images), batch):
        take = idx[start : start + batch]
        yield images[take], labels[take]


def collect_photo_digits() -> tuple[np.ndarray, np.ndarray]:
    """Свои карточки: класс ЛР2 → {1, 7, 8}, уже после поворота ЛР3."""
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for path in list_input_images("data/input"):
        lab3 = run_lab3(load_rgb(path))
        for fig in lab3.figures:
            digit = label_from_figure(fig)
            img = rgb_to_mnist(fig.after, fig.after_mask)
            xs.append(img)
            ys.append(digit)
            if digit in (0, 1, 8):
                xs.append(img[::-1, ::-1].copy())
                ys.append(digit)
    if not xs:
        return np.zeros((0, 28, 28), np.float32), np.zeros(0, np.int32)
    return np.stack(xs), np.asarray(ys, dtype=np.int32)


def train_model(
    weights: Path = WEIGHTS_PATH,
    epochs: int = 6,
    fine_epochs: int = 5,
    batch: int = 128,
    seed: int = 0,
) -> MLP:
    print("MNIST: загрузка…")
    try:
        x_tr, y_tr, x_te, y_te = load_mnist()
        print(f"MNIST train {len(x_tr)}  test {len(x_te)}")
    except Exception as exc:  # noqa: BLE001
        print(f"MNIST недоступен ({exc}), обучаем на синтетике")
        x_tr, y_tr = make_synthetic_set(900, seed=seed + 1)
        x_te, y_te = make_synthetic_set(180, seed=seed + 2)
    syn_x, syn_y = make_synthetic_set(350, seed=seed + 3)
    x_tr = np.concatenate([x_tr, syn_x], axis=0)
    y_tr = np.concatenate([y_tr, syn_y], axis=0)

    rng = np.random.default_rng(seed)
    model = MLP(ARCH, seed=seed)
    vel_w, vel_b = model.zero_velocity()

    n_tr = len(x_tr)
    for epoch in range(1, epochs + 1):
        lr = 0.09 * (0.72 ** (epoch - 1))
        order = rng.permutation(n_tr)
        running = 0.0
        seen = 0
        for start in range(0, n_tr, batch):
            take = order[start : start + batch]
            imgs = augment_images(x_tr[take], rng)
            loss = model.loss_backward(flatten(imgs), y_tr[take], l2=1e-4)
            model.step(lr, vel_w, vel_b, momentum=0.9)
            running += loss * len(take)
            seen += len(take)
        acc = _accuracy(model, x_te, y_te)
        print(f"эпоха {epoch}/{epochs}  loss={running / seen:.4f}  MNIST test={acc:.3f}  lr={lr:.4f}")

    print("Свои фото: сбор цифр ЛР3…")
    own_x, own_y = collect_photo_digits()
    print(f"своих образцов: {len(own_x)}")
    if len(own_x):
        mask178 = np.isin(y_tr, (1, 7, 8))
        mix_x = np.concatenate([x_tr[mask178][:8000], own_x.repeat(12, axis=0)], axis=0)
        mix_y = np.concatenate([y_tr[mask178][:8000], own_y.repeat(12, axis=0)], axis=0)
        vel_w, vel_b = model.zero_velocity()
        for epoch in range(1, fine_epochs + 1):
            lr = 0.012 * (0.8 ** (epoch - 1))
            running = 0.0
            seen = 0
            for imgs, labs in _iterate_minibatches(mix_x, mix_y, batch, rng):
                imgs = augment_images(imgs, rng)
                loss = model.loss_backward(flatten(imgs), labs, l2=2e-4)
                model.step(lr, vel_w, vel_b, momentum=0.9)
                running += loss * len(labs)
                seen += len(labs)
            acc = _accuracy(model, x_te, y_te)
            own_acc = _accuracy(model, own_x, own_y)
            print(
                f"дообучение {epoch}/{fine_epochs}  loss={running / seen:.4f}  "
                f"MNIST={acc:.3f}  свои={own_acc:.3f}"
            )

    model.save(weights)
    print(f"веса: {weights}")
    return model


def ensure_model(weights: Path = WEIGHTS_PATH) -> MLP:
    weights = Path(weights)
    if weights.exists():
        return MLP.load(weights)
    return train_model(weights)


def upscale_mnist(image28: np.ndarray, side: int = 140) -> np.ndarray:
    from src.lab4_data import resize_gray

    big = np.clip(resize_gray(image28, side, side), 0.0, 1.0)
    rgb = np.zeros((side, side, 3), dtype=np.uint8)
    val = (big * 255.0).astype(np.uint8)
    rgb[..., 0] = val
    rgb[..., 1] = val
    rgb[..., 2] = val
    return rgb


def color_square(rgb: np.ndarray, side: int = 160) -> np.ndarray:
    return square_pad(rgb, side)
