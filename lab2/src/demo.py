"""Демонстрация на сдаче: вход → шаги → итог, без текста в интерфейсе."""

from pathlib import Path

import matplotlib.pyplot as plt
from IPython.display import HTML, display

from src.pipeline import list_input_images, load_rgb, process, save_rgb
from src.viz import show_pipeline

_HIDE_CODE = """
<style>
.jp-Cell .jp-InputArea,
.jp-Cell .jp-InputPrompt,
.jp-Collapser,
div.input,
div.input_area,
.prompt {
    display: none !important;
}

html, body,
#main, .jp-MainAreaWidget,
.jp-NotebookPanel, .jp-NotebookPanel-notebook,
.jp-WindowedPanel, .jp-WindowedPanel-outer, .jp-WindowedPanel-inner,
.jp-Notebook, .jp-Cell, .jp-Cell-outputWrapper,
.jp-OutputArea, .jp-OutputArea-child, .jp-OutputArea-output,
.jp-RenderedImage, .jp-OutputArea-executeResult,
.output, .output_wrapper, .output_scroll, .output_area,
div.output_subarea, .jp-mod-outputsScrolled {
    max-height: none !important;
    height: auto !important;
    overflow: visible !important;
}

.jp-WindowedPanel-outer,
.jp-NotebookPanel-notebook {
    position: static !important;
}

.jp-OutputArea img, .output_area img {
    max-height: none !important;
    height: auto !important;
}
</style>
<script>
document.querySelectorAll('.output_scroll, .jp-mod-outputsScrolled').forEach(function (el) {
    el.classList.remove('output_scroll', 'jp-mod-outputsScrolled');
});
</script>
"""


def run_demo() -> None:
    display(HTML(_HIDE_CODE))
    plt.rcParams["figure.dpi"] = 110
    out = Path("data/output")
    out.mkdir(parents=True, exist_ok=True)
    for i, path in enumerate(list_input_images("data/input")):
        backend = "author" if i == 0 else "library"
        how = "вариант A, авторский NumPy" if backend == "author" else "вариант B, OpenCV"
        result = process(load_rgb(path), backend=backend)
        show_pipeline(result, f"{path.name}  ·  {how}")
        save_rgb(out / f"{path.stem}_cleaned.png", result.cleaned)
        save_rgb(out / f"{path.stem}_sobel.png", result.sobel)
