"""Вспомогательные функции отображения для ноутбука."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def show_images(images: list[np.ndarray], titles: list[str], cmap=None, cols: int = 3) -> None:
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, img, title in zip(axes, images, titles):
        if img.ndim == 2:
            ax.imshow(img, cmap=cmap or "gray")
        else:
            ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def show_pipeline(result, name: str = "") -> None:
    """Вход → промежуточные шаги → итог — одна лента без внутренней прокрутки."""
    from io import BytesIO

    from IPython.display import Image, display

    images = [
        result.original,
        result.denoised,
        result.mask,
        result.gray,
        result.contrasted,
        result.sobel,
        result.overlay,
        result.cleaned,
    ]
    titles = [
        "Вход\nисходное фото",
        "Медиана\nфильтр, убирает шум",
        "Маска HSV\nцвет отделяет карточки от фона",
        "Серое\nградации яркости",
        "Контраст\nлинейное растяжение гистограммы",
        "Собель\nконтуры фигуры и цифр",
        "Наложение\nконтур Собеля поверх объекта",
        "Итог\nфон убран, объект сохранён",
    ]
    heading = "входное изображение  →  промежуточные шаги  →  итог"
    if name:
        heading = f"{name}\n{heading}"

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(heading, fontsize=14, fontweight="bold")
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.85], hspace=0.42, wspace=0.14)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]
    for ax, img, title in zip(axes, images, titles):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    ax_h1 = fig.add_subplot(gs[2, :2])
    ax_h2 = fig.add_subplot(gs[2, 2:])
    for ax, hist, title in (
        (ax_h1, result.hist_original, "Гистограмма входа — распределение яркостей исходника"),
        (ax_h2, result.hist_cleaned, "Гистограмма итога — после удаления фона"),
    ):
        ax.bar(np.arange(256), hist, width=1.0, color="steelblue")
        ax.set_title(title)
        ax.set_xlabel("Яркость")
        ax.set_ylabel("Число пикселей")
        ax.set_xlim(0, 255)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    display(Image(data=buf.getvalue()))


def show_histograms(hists: list[np.ndarray], titles: list[str]) -> None:
    fig, axes = plt.subplots(1, len(hists), figsize=(5 * len(hists), 3.2))
    axes = np.atleast_1d(axes)
    for ax, hist, title in zip(axes, hists, titles):
        ax.bar(np.arange(256), hist, width=1.0, color="steelblue")
        ax.set_title(title)
        ax.set_xlabel("Яркость")
        ax.set_ylabel("Число пикселей")
        ax.set_xlim(0, 255)
    fig.tight_layout()
    plt.show()
