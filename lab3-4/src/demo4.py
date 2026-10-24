"""Демонстрация ЛР4: распознавание цифр, без кода в UI."""

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import HTML, Image, display

from src.demo import _HIDE_CODE
from src.demo2 import _setup_font
from src.lab3 import square_pad
from src.lab4 import DigitGuess, Lab4Result, ensure_model, run_lab4, upscale_mnist
from src.pipeline import list_input_images, load_rgb, save_rgb


def _show(ax, img, title: str) -> None:
    ax.imshow(img)
    ax.set_title(title, fontsize=11, pad=8)
    ax.axis("off")


def _guess_titles(g: DigitGuess) -> tuple[str, str, str]:
    pred = f"#{g.label}  предсказанная цифра: {g.pred}"
    orig = f"оригинал: {g.pred}  ({g.conf * 100:.1f}%)"
    flip = f"перевернуть: {g.pred_flip}  ({g.conf_flip * 100:.1f}%)"
    return pred, orig, flip


def show_lab4(result: Lab4Result, name: str = "") -> None:
    _setup_font()
    n = max(len(result.guesses), 1)
    heading = "нейронная сеть распознаёт каждую цифру"
    if name:
        heading = f"{name}\n{heading}"

    fig = plt.figure(figsize=(16.6, 14.6))
    fig.suptitle(heading, fontsize=14, fontweight="bold", y=0.98)
    outer = fig.add_gridspec(3, 1, height_ratios=[1.05, 0.07, 2.35], hspace=0.16)

    gs_top = outer[0].subgridspec(1, 3, wspace=0.12)
    overview = [
        (result.lab3.lab2.original, "Вход\nисходное фото"),
        (result.lab3.lab2.cleaned, "ЛР2\nвыделенные фигуры"),
        (result.lab3.method1, "ЛР3\nповорот в читаемый вид"),
    ]
    for col, (img, title) in enumerate(overview):
        _show(fig.add_subplot(gs_top[0, col]), img, title)

    ax_cap = fig.add_subplot(outer[1])
    ax_cap.set_axis_off()
    ax_cap.text(
        0.5,
        0.45,
        "Сеть видит канонический 28×28 и тот же кадр, перевёрнутый на 180°",
        ha="center",
        va="center",
        fontsize=13,
    )

    gs_bot = outer[2].subgridspec(4, n, hspace=0.42, wspace=0.10)
    if result.guesses:
        side = 160
        for i, g in enumerate(result.guesses):
            pred, orig, flip = _guess_titles(g)
            _show(fig.add_subplot(gs_bot[0, i]), square_pad(g.color, side), f"#{g.label}  цифра с карточки")
            _show(fig.add_subplot(gs_bot[1, i]), upscale_mnist(g.mnist, side), pred)
            _show(fig.add_subplot(gs_bot[2, i]), upscale_mnist(g.mnist, side), orig)
            _show(fig.add_subplot(gs_bot[3, i]), upscale_mnist(g.mnist_flip, side), flip)
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
    model = ensure_model()
    out = Path("data/output_lab4")
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.txt").write_text("", encoding="utf-8")
    for path in list_input_images("data/input"):
        result = run_lab4(load_rgb(path), model=model)
        show_lab4(result, f"{path.name}  ·  фигур: {len(result.guesses)}")
        for g in result.guesses:
            save_rgb(out / f"{path.stem}_digit{g.label}_card.png", square_pad(g.color, 160))
            save_rgb(out / f"{path.stem}_digit{g.label}_mnist.png", upscale_mnist(g.mnist, 160))
            save_rgb(out / f"{path.stem}_digit{g.label}_flip.png", upscale_mnist(g.mnist_flip, 160))
            line = (
                f"{path.name} #{g.label} pred={g.pred} {g.conf * 100:.1f}%  "
                f"flip={g.pred_flip} {g.conf_flip * 100:.1f}%\n"
            )
            (out / "predictions.txt").open("a", encoding="utf-8").write(line)
