"""Демонстрация ЛР3: поворот цифр в читаемый вид, без кода в UI."""

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import HTML, Image, display

from src.demo import _HIDE_CODE
from src.demo2 import _setup_font
from src.lab3 import Lab3Result, run_lab3, square_pad
from src.pipeline import list_input_images, load_rgb, save_rgb


def _show(ax, img, title: str) -> None:
    ax.imshow(img)
    ax.set_title(title, fontsize=11, pad=10)
    ax.axis("off")


def show_lab3(result: Lab3Result, name: str = "") -> None:
    _setup_font()
    n = max(len(result.figures), 1)
    heading = "каждую фигуру поворачиваем в читаемый вид"
    if name:
        heading = f"{name}\n{heading}"

    fig = plt.figure(figsize=(16.4, 13.4))
    fig.suptitle(heading, fontsize=14, fontweight="bold", y=0.98)
    outer = fig.add_gridspec(3, 1, height_ratios=[1.2, 0.08, 2.0], hspace=0.18)

    gs_top = outer[0].subgridspec(1, 4, wspace=0.14)
    overview = [
        (result.lab2.original, "Вход\nисходное фото"),
        (result.lab2.cleaned, "ЛР2\nвыделенные фигуры"),
        (result.method1, "Способ 1\nповорот всех цифр на месте"),
        (result.method1_color, "Способ 1\nтот же поворот, цвет класса"),
    ]
    for col, (img, title) in enumerate(overview):
        _show(fig.add_subplot(gs_top[0, col]), img, title)

    ax_cap = fig.add_subplot(outer[1])
    ax_cap.set_axis_off()
    ax_cap.text(
        0.5,
        0.45,
        "Способ 2: вырезать каждую цифру и повернуть отдельно   (было → стало)",
        ha="center",
        va="center",
        fontsize=13,
    )

    gs_bot = outer[2].subgridspec(2, n, hspace=0.48, wspace=0.10)
    if result.figures:
        side = max(
            max(f.before.shape[0], f.before.shape[1], f.after.shape[0], f.after.shape[1])
            for f in result.figures
        )
        side = max(int(side), 80)
        for i, figu in enumerate(result.figures):
            a = figu.angle_deg
            if abs(a) < 3.5:
                after_title = f"#{figu.label}  стало\nуже читаемо"
            elif abs(a) <= 16:
                after_title = f"#{figu.label}  стало  ·  {a:+.0f}°\nвыровняли перекладину"
            else:
                after_title = f"#{figu.label}  стало  ·  {a:+.0f}°"
            _show(
                fig.add_subplot(gs_bot[0, i]),
                square_pad(figu.before, side),
                f"#{figu.label}  было",
            )
            _show(
                fig.add_subplot(gs_bot[1, i]),
                square_pad(figu.after, side),
                after_title,
            )
    else:
        ax = fig.add_subplot(gs_bot[:, :])
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Фигуры не найдены", ha="center", va="center")

    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, pad_inches=0.25)
    plt.close(fig)
    display(Image(data=buf.getvalue()))


def run_demo() -> None:
    display(HTML(_HIDE_CODE))
    _setup_font()
    out = Path("data/output_lab3")
    out.mkdir(parents=True, exist_ok=True)
    for path in list_input_images("data/input"):
        result = run_lab3(load_rgb(path))
        n_obj = len(result.figures)
        show_lab3(result, f"{path.name}  ·  фигур: {n_obj}")
        save_rgb(out / f"{path.stem}_method1.png", result.method1)
        save_rgb(out / f"{path.stem}_method1_color.png", result.method1_color)
        for figu in result.figures:
            save_rgb(out / f"{path.stem}_digit{figu.label}_before.png", square_pad(figu.before, 160))
            save_rgb(out / f"{path.stem}_digit{figu.label}_after.png", square_pad(figu.after, 160))
