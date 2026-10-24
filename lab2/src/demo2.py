"""Демонстрация ЛР2 на сдаче: вход → разделение → кластеры, без кода в UI."""

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from IPython.display import HTML, Image, display

from src.demo import _HIDE_CODE
from src.lab2 import Lab2Result, analyze
from src.pipeline import list_input_images, load_rgb, save_rgb


def _setup_font() -> None:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Segoe UI", "Arial", "Calibri", "DejaVu Sans"):
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110


def _to_uint8_heat(dist: np.ndarray) -> np.ndarray:
    peak = float(dist.max()) if dist.size else 1.0
    if peak < 1e-9:
        return np.zeros(dist.shape, dtype=np.uint8)
    return np.clip(dist / peak * 255.0, 0, 255).astype(np.uint8)


_CLASS_NAMES = ("красный", "синий", "зелёный")
_CLASS_RGB = [(0.86, 0.14, 0.14), (0.14, 0.31, 0.90), (0.14, 0.78, 0.22)]


def _draw_feature_table(ax, result: Lab2Result) -> None:
    ax.set_axis_off()
    ax.set_title("Таблица признаков", fontsize=13, pad=10)
    if not result.objects:
        ax.text(0.5, 0.5, "Объекты не найдены", ha="center", va="center", fontsize=12)
        return

    headers = ["№", "Площадь S", "Периметр P", "Центр (x, y)", "Компактность", "Эксцентриситет", "Дыры", "Класс"]
    cells = []
    face = []
    for f, c in zip(result.objects, result.classes):
        cls = int(c) % 3
        cells.append(
            [
                f"#{f.label}",
                f"{f.area}",
                f"{f.perimeter}",
                f"{f.cx:.0f}, {f.cy:.0f}",
                f"{f.compactness:.2f}",
                f"{f.eccentricity:.2f}",
                f"{f.holes}",
                f"{cls + 1}  {_CLASS_NAMES[cls]}",
            ]
        )
        tint = (*_CLASS_RGB[cls], 0.16)
        face.append([tint] * len(headers))

    table = ax.table(
        cellText=cells,
        colLabels=headers,
        cellColours=face,
        loc="center",
        cellLoc="center",
        bbox=[0.02, 0.08, 0.96, 0.82],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor((0.80, 0.80, 0.80))
        cell.set_linewidth(0.5)
        cell.PAD = 0.04
        if row == 0:
            cell.set_facecolor((0.20, 0.27, 0.38))
            cell.set_text_props(color="white", weight="bold", fontsize=10)
        elif col == 0:
            cell.set_text_props(weight="bold")
        elif col == 7:
            cls = int(result.classes[row - 1]) % 3
            cell.set_text_props(color=_CLASS_RGB[cls], weight="bold")


def _draw_feature_plot(ax, result: Lab2Result) -> None:
    ax.set_title("Признаковое пространство", fontsize=13, pad=10)
    if not result.objects:
        ax.set_axis_off()
        return

    xs = np.array([f.eccentricity for f in result.objects], dtype=np.float64)
    ys = np.array([f.compactness for f in result.objects], dtype=np.float64)
    for cls, name, color in zip(range(3), _CLASS_NAMES, _CLASS_RGB):
        mask = np.array([int(c) % 3 == cls for c in result.classes])
        if not mask.any():
            continue
        ax.scatter(
            xs[mask],
            ys[mask],
            c=[color],
            s=110,
            edgecolors="k",
            linewidths=0.6,
            label=f"класс {cls + 1} — {name}",
            zorder=3,
        )
    for f, x, y in zip(result.objects, xs, ys):
        ax.annotate(
            f" #{f.label}",
            (x, y),
            textcoords="offset points",
            xytext=(5, 6),
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xlabel("Эксцентриситет")
    ax.set_ylabel("Компактность")
    pad_x = 0.10 if float(np.ptp(xs)) < 0.15 else 0.07
    pad_y = 0.07 if float(np.ptp(ys)) < 0.15 else 0.05
    ax.set_xlim(max(-0.08, float(xs.min()) - pad_x), min(1.10, float(xs.max()) + pad_x))
    ax.set_ylim(max(0.0, float(ys.min()) - pad_y), float(ys.max()) + pad_y)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)


def show_lab2(result: Lab2Result, name: str = "") -> None:
    _setup_font()
    dist_u8 = _to_uint8_heat(result.distance)
    images = [
        result.original,
        result.cleaned,
        result.mask,
        dist_u8,
        result.eroded,
        result.tiles_color,
        result.inner,
        result.colored,
    ]
    titles = [
        "Вход\nисходное фото",
        "ЛР1: удаление фона\nостаются карточки",
        "Бинарная маска\nобъект / фон",
        "Карта расстояний\nрасстояние пикселя объекта до фона",
        "Маркеры объектов\nвнутренние области по порогу карты",
        "Watershed: разделённые плитки\nкаждая карточка — своя метка",
        "Внутренние фигуры\nцифры и рисунки",
        "K-means\nитоговая кластеризация",
    ]
    heading = "вход  →  разделение объектов  →  признаки  →  3 кластера"
    if name:
        heading = f"{name}\n{heading}"

    n_rows = max(len(result.objects), 1)
    table_h = 0.42 + 0.11 * n_rows
    fig = plt.figure(figsize=(16.2, 14.8))
    fig.suptitle(heading, fontsize=14, fontweight="bold", y=0.98)
    gs = fig.add_gridspec(
        4,
        4,
        height_ratios=[1.05, 1.05, table_h, 1.15],
        hspace=0.42,
        wspace=0.16,
    )
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]
    for ax, img, title in zip(axes, images, titles):
        if img.ndim == 2:
            cmap = "inferno" if title.startswith("Карта") else "gray"
            ax.imshow(img, cmap=cmap)
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    _draw_feature_table(fig.add_subplot(gs[2, :]), result)
    _draw_feature_plot(fig.add_subplot(gs[3, :]), result)

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, pad_inches=0.25)
    plt.close(fig)
    display(Image(data=buf.getvalue()))


def run_demo() -> None:
    display(HTML(_HIDE_CODE))
    _setup_font()
    out = Path("data/output_lab2")
    out.mkdir(parents=True, exist_ok=True)
    paths = list_input_images("data/input")
    for path in paths:
        result = analyze(load_rgb(path), backend="library")
        n_obj = len(result.objects)
        show_lab2(result, f"{path.name}  ·  объектов: {n_obj}  ·  k-means, k=3")
        save_rgb(out / f"{path.stem}_clusters.png", result.colored)
        save_rgb(out / f"{path.stem}_tiles.png", result.tiles_color)
