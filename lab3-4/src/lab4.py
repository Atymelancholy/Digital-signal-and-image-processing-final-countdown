"""
Лабораторная №4: распознавание цифр своей нейронной сетью.
Сеть — MLP на NumPy. Вход — повёрнутые фигуры ЛР3, плюс проверка переворота 180°.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import hashlib

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

WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "data/models/lab4_mlp.npz"
ARCH = [784, 256, 128, 10]


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


def guess_mnist_image(model: MLP, gray_or_rgb: np.ndarray, label: int = 1) -> DigitGuess:
    """Прямое распознавание картинки в стиле MNIST (без ЛР2/ЛР3)."""
    from src.lab4_data import fit_to_mnist

    img = np.asarray(gray_or_rgb, dtype=np.float64)
    if img.ndim == 3:
        img = img.max(axis=2)
    peak = float(img.max()) if img.size else 0.0
    if peak > 1.5:
        img = img / 255.0
    elif peak > 1.0:
        img = img / peak
    img = np.clip(img, 0.0, 1.0)
    # Чёрная цифра на белом фоне → как в MNIST (светлая цифра на тёмном).
    if float(img.mean()) > 0.5:
        img = 1.0 - img
    mnist = fit_to_mnist(img)
    flipped = mnist[::-1, ::-1].copy()
    probs = mnist_probs(model, mnist)
    probs_f = mnist_probs(model, flipped)
    pred = int(probs.argmax())
    pred_f = int(probs_f.argmax())
    return DigitGuess(
        label=label,
        pred=pred,
        conf=float(probs[pred]),
        pred_flip=pred_f,
        conf_flip=float(probs_f[pred_f]),
        probs=probs,
        probs_flip=probs_f,
        color=upscale_mnist(mnist, 160),
        mnist=mnist,
        mnist_flip=flipped,
    )


def guess_figure(model: MLP, rgb: np.ndarray, mask: np.ndarray, label: int) -> DigitGuess:
    mnist = rgb_to_mnist(rgb, mask)
    flipped = mnist[::-1, ::-1].copy()
    probs = mnist_probs(model, mnist)
    probs_f = mnist_probs(model, flipped)
    pred = int(probs.argmax())
    pred_f = int(probs_f.argmax())
    conf = float(probs[pred])
    conf_f = float(probs_f[pred_f])
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


def train_model(
    weights: Path = WEIGHTS_PATH,
    epochs: int = 20,
    batch: int = 128,
    seed: int = 0,
) -> MLP:
    """Обучение с нуля: MNIST train + синтетика всех 10 классов.

    5000 примеров из train отложены для выбора эпохи. MNIST test и все
    фотографии проекта не используются ни для градиентов, ни для выбора эпохи.
    """
    x_all, y_all, x_test, y_test = load_mnist()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x_all))
    val_ids, train_ids = order[:5000], order[5000:]
    x_val, y_val = x_all[val_ids], y_all[val_ids]
    print("Preparing balanced synthetic digits...", flush=True)
    syn_x, syn_y = make_synthetic_set(1000, seed=seed + 3)
    x_train = np.concatenate([x_all[train_ids], syn_x])
    y_train = np.concatenate([y_all[train_ids], syn_y])
    model = MLP(ARCH, seed=seed)
    velocity_w, velocity_b = model.zero_velocity()
    best_score, best_parameters = -1.0, None
    history = []
    for epoch in range(1, epochs + 1):
        lr = 0.06 * (0.90 ** (epoch - 1))
        total_loss = 0.0
        for images, labels in _iterate_minibatches(x_train, y_train, batch, rng):
            images = augment_images(images, rng)
            loss = model.loss_backward(flatten(images), labels, l2=1e-5)
            model.step(lr, velocity_w, velocity_b, momentum=0.9)
            total_loss += loss * len(labels)
        val_accuracy = _accuracy(model, x_val, y_val)
        history.append(dict(epoch=epoch, loss=total_loss / len(y_train),
                            validation_accuracy=val_accuracy))
        print(f"Epoch {epoch}/{epochs}: loss={history[-1]['loss']:.4f} "
              f"validation={val_accuracy:.4f}", flush=True)
        if val_accuracy > best_score:
            best_score = val_accuracy
            best_parameters = ([w.copy() for w in model.W], [b.copy() for b in model.b])
    if best_parameters is None:
        raise ValueError("epochs must be positive")
    model.W, model.b = best_parameters
    model.save(weights)
    predictions = np.concatenate([model.predict(flatten(x_test[i:i+512]))
                                  for i in range(0, len(x_test), 512)])
    report = dict(seed=seed, architecture=ARCH, train_mnist=len(train_ids),
                  weights_sha256=hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
                  validation_mnist=len(val_ids), synthetic_per_class=1000,
                  photo_training_samples=0, selection="best MNIST validation accuracy",
                  test_samples=len(y_test), test_accuracy=float((predictions == y_test).mean()),
                  per_class_accuracy={str(d): float((predictions[y_test == d] == d).mean())
                                      for d in range(10)}, history=history)
    Path(weights).with_suffix(".metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print("Held-out MNIST test:", report["test_accuracy"], flush=True)
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
