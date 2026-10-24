"""Демонстрация ЛР4: распознавание цифр, без кода в UI."""

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import Image, display

from src.demo2 import _setup_font
from src.lab3 import square_pad
from src.lab4 import Lab4Result, ensure_model, run_lab4, upscale_mnist
from src.pipeline import list_input_images, load_rgb, save_rgb

ROOT = Path(__file__).resolve().parents[1]


def _show(ax, img, title: str) -> None:
    ax.imshow(img)
    ax.set_title(title, fontsize=11, pad=8)
    ax.axis("off")


def show_lab4(result: Lab4Result, name: str = "") -> None:
    _setup_font()
    n = max(len(result.guesses), 1)
    heading = "нейронная сеть распознаёт каждую цифру"
    if name:
        heading = f"{name}\n{heading}"

    fig = plt.figure(figsize=(14.5, 9.5))
    fig.suptitle(heading, fontsize=14, fontweight="bold", y=0.98)
    outer = fig.add_gridspec(3, 1, height_ratios=[1.0, 0.07, 1.6], hspace=0.16)

    gs_top = outer[0].subgridspec(1, 2, wspace=0.12)
    overview = [
        (result.lab3.lab2.original, "Вход\nисходное фото"),
        (result.lab3.lab2.cleaned, "ЛР2\nвыделенные фигуры"),
    ]
    for col, (img, title) in enumerate(overview):
        _show(fig.add_subplot(gs_top[0, col]), img, title)

    ax_cap = fig.add_subplot(outer[1])
    ax_cap.set_axis_off()
    ax_cap.text(
        0.5,
        0.45,
        "После поворота ЛР3 · softmax — оценка модели, не гарантия правильности",
        ha="center",
        va="center",
        fontsize=13,
    )

    gs_bot = outer[2].subgridspec(2, n, hspace=0.42, wspace=0.10)
    if result.guesses:
        side = 160
        for i, g in enumerate(result.guesses):
            answer = f"#{g.label}  ответ сети: {g.pred}  ·  {g.conf * 100:.1f}%"
            _show(fig.add_subplot(gs_bot[0, i]), square_pad(g.color, side), f"#{g.label}  цифра с карточки")
            _show(fig.add_subplot(gs_bot[1, i]), upscale_mnist(g.mnist, side), answer)
    else:
        ax = fig.add_subplot(gs_bot[:, :])
        ax.set_axis_off()
        ax.text(0.5, 0.5, "Фигуры не найдены", ha="center", va="center")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    try:
        ip = get_ipython()  # type: ignore[name-defined]
    except NameError:
        ip = None
    if ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell":
        plt.show()
    else:
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, pad_inches=0.25)
        display(Image(data=buf.getvalue()))
    plt.close(fig)


def show_mnist_guesses(guesses, name: str = "") -> None:
    _setup_font()
    n = max(len(guesses), 1)
    heading = "ЛР4 · прямое распознавание MNIST-картинок"
    if name:
        heading = f"{name}\n{heading}"
    fig = plt.figure(figsize=(max(3.2 * n, 6), 4.2))
    fig.suptitle(heading, fontsize=13, fontweight="bold", y=0.98)
    for i, g in enumerate(guesses):
        ax = fig.add_subplot(1, n, i + 1)
        title = f"{g.pred}  ·  {g.conf * 100:.1f}%"
        _show(ax, upscale_mnist(g.mnist, 160), title)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    try:
        ip = get_ipython()  # type: ignore[name-defined]
    except NameError:
        ip = None
    if ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell":
        plt.show()
    else:
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=120, pad_inches=0.2)
        display(Image(data=buf.getvalue()))
    plt.close(fig)


def run_demo() -> None:
    _setup_font()
    print("ЛР4: запуск, загрузка модели…", flush=True)
    model = ensure_model()
    inp = ROOT / "data/input"
    out = ROOT / "data/output_lab4"
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictions.txt").write_text("", encoding="utf-8")
    paths = list_input_images(inp)
    print(f"ЛР4: папка {inp}, уникальных фото: {len(paths)}.", flush=True)
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] Обработка {path.name}…", flush=True)
        result = run_lab4(load_rgb(path), model=model)
        show_lab4(result, f"{path.name}  ·  фигур: {len(result.guesses)}")
        for g in result.guesses:
            save_rgb(out / f"{path.stem}_digit{g.label}_card.png", square_pad(g.color, 160))
            save_rgb(out / f"{path.stem}_digit{g.label}_mnist.png", upscale_mnist(g.mnist, 160))
            line = (
                f"{path.name} #{g.label} pred={g.pred} {g.conf * 100:.1f}%\n"
            )
            (out / "predictions.txt").open("a", encoding="utf-8").write(line)
    print("ЛР4: готово.", flush=True)


def run_demo_mnist() -> None:
    """Только сеть: картинки из data/input_mnist (без ЛР2/ЛР3)."""
    from src.lab4 import guess_mnist_image

    _setup_font()
    print("ЛР4 (MNIST): загрузка модели…", flush=True)
    model = ensure_model()
    inp = ROOT / "data/input_mnist"
    out = ROOT / "data/output_lab4_mnist"
    inp.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    paths = list_input_images(inp)
    if not paths:
        print(f"ЛР4 (MNIST): положите картинки цифр в {inp}", flush=True)
        return
    print(f"ЛР4 (MNIST): {inp}, файлов: {len(paths)}.", flush=True)
    lines = []
    guesses = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path.name}…", flush=True)
        g = guess_mnist_image(model, load_rgb(path), label=index)
        guesses.append(g)
        save_rgb(out / f"{path.stem}_pred{g.pred}.png", upscale_mnist(g.mnist, 160))
        line = f"{path.name} pred={g.pred} {g.conf * 100:.1f}%"
        lines.append(line)
        print(line, flush=True)
    show_mnist_guesses(guesses, f"файлов: {len(paths)}")
    (out / "predictions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("ЛР4 (MNIST): готово.", flush=True)
