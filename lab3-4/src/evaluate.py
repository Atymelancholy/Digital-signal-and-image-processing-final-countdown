"""Воспроизводимая проверка фотографий по ручной разметке: python -m src.evaluate."""
from pathlib import Path
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw

from src.lab3 import square_pad
from src.lab4 import ensure_model, run_lab4, upscale_mnist
from src.pipeline import load_rgb, save_rgb

ROOT = Path(__file__).resolve().parents[1]


def evaluate_photos() -> dict:
    annotations = json.loads((ROOT / "data/photo_labels.json").read_text(encoding="utf-8"))
    model = ensure_model()
    rows = []
    out = ROOT / "data/evaluation"
    out.mkdir(parents=True, exist_ok=True)
    predictions = []
    for entry in annotations["images"]:
        path = ROOT / "data/input" / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Image changed: recheck annotations for {path.name}")
        result = run_lab4(load_rgb(path), model)
        if len(result.guesses) != len(entry["digits"]):
            raise ValueError(f"Detection count changed: {path.name}")
        canvas = Image.new("RGB", (250 + 170 * len(result.guesses), 260), "white")
        draw = ImageDraw.Draw(canvas)
        original = Image.fromarray(result.lab3.lab2.original)
        original.thumbnail((240, 215))
        canvas.paste(original, (0, 30))
        draw.text((5, 8), path.name, fill="black")
        for i, (truth, fig, guess) in enumerate(zip(entry["digits"], result.lab3.figures, result.guesses)):
            row = dict(file=path.name, figure=fig.label, expected=truth, predicted=guess.pred,
                       softmax=guess.conf, angle=fig.angle_deg, correct=guess.pred == truth)
            rows.append(row)
            image = Image.fromarray(fig.after)
            image.thumbnail((160, 190))
            canvas.paste(image, (250 + 170*i, 55))
            draw.text((250 + 170*i, 8), f"#{fig.label}: expected {truth}\nnetwork {guess.pred} ({guess.conf:.1%})",
                      fill="green" if row["correct"] else "red")
            base = f"{path.stem}_digit{fig.label}"
            save_rgb(ROOT / "data/output_lab3" / f"{base}_before.png", square_pad(fig.before, 160))
            save_rgb(ROOT / "data/output_lab3" / f"{base}_after.png", square_pad(fig.after, 160))
            save_rgb(ROOT / "data/output_lab4" / f"{base}_card.png", square_pad(guess.color, 160))
            save_rgb(ROOT / "data/output_lab4" / f"{base}_mnist.png", upscale_mnist(guess.mnist, 160))
            predictions.append(f"{path.name} #{fig.label} expected={truth} network={guess.pred} "
                               f"softmax={guess.conf:.4f}")
        canvas.save(out / f"{path.stem}.png")
        print(path.name, [g.pred for g in result.guesses], "expected", entry["digits"], flush=True)
    report = dict(images=len(annotations["images"]), figures=len(rows),
                  correct=sum(r["correct"] for r in rows),
                  accuracy=float(np.mean([r["correct"] for r in rows])),
                  note="Known regression images; no photo training. Not an independent estimate of generalization.",
                  rows=rows)
    (out / "photos.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ROOT / "data/output_lab4/predictions.txt").write_text("\n".join(predictions)+"\n", encoding="utf-8")
    print("Photo regression accuracy:", report["accuracy"], flush=True)
    return report


if __name__ == "__main__":
    evaluate_photos()
