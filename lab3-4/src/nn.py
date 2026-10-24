"""
Свой многослойный перцептрон: ReLU, softmax, кросс-энтропия, SGD с моментом.
Без PyTorch / TensorFlow / sklearn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class MLP:
    def __init__(self, sizes: list[int], seed: int = 0) -> None:
        self.sizes = [int(s) for s in sizes]
        rng = np.random.default_rng(seed)
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for n_in, n_out in zip(self.sizes[:-1], self.sizes[1:]):
            self.W.append(rng.normal(0.0, np.sqrt(2.0 / n_in), (n_in, n_out)).astype(np.float32))
            self.b.append(np.zeros(n_out, dtype=np.float32))

    def logits(self, X: np.ndarray) -> np.ndarray:
        a = np.asarray(X, dtype=np.float32)
        self._acts = [a]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            a = z if i == len(self.W) - 1 else np.maximum(z, 0.0)
            self._acts.append(a)
        return a

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z, dtype=np.float64)
        return (e / e.sum(axis=1, keepdims=True)).astype(np.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.softmax(self.logits(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def loss_backward(self, X: np.ndarray, y: np.ndarray, l2: float = 1e-4) -> float:
        logits = self.logits(X)
        p = self.softmax(logits)
        n = X.shape[0]
        y = np.asarray(y, dtype=np.int32)
        logp = np.log(np.clip(p[np.arange(n), y], 1e-9, 1.0))
        loss = float(-logp.mean())
        if l2:
            loss += 0.5 * l2 * float(sum(float((W * W).sum()) for W in self.W))

        grad = p.copy()
        grad[np.arange(n), y] -= 1.0
        grad /= n

        self.dW = [np.empty_like(W) for W in self.W]
        self.db = [np.empty_like(b) for b in self.b]
        for i in range(len(self.W) - 1, -1, -1):
            a_prev = self._acts[i]
            self.dW[i] = a_prev.T @ grad
            if l2:
                self.dW[i] += l2 * self.W[i]
            self.db[i] = grad.sum(axis=0)
            if i > 0:
                grad = grad @ self.W[i].T
                grad *= (self._acts[i] > 0).astype(np.float32)
        return loss

    def zero_velocity(self) -> tuple[list[np.ndarray], list[np.ndarray]]:
        return [np.zeros_like(W) for W in self.W], [np.zeros_like(b) for b in self.b]

    def step(
        self,
        lr: float,
        vel_w: list[np.ndarray],
        vel_b: list[np.ndarray],
        momentum: float = 0.9,
    ) -> None:
        for i in range(len(self.W)):
            vel_w[i] *= momentum
            vel_w[i] -= lr * self.dW[i]
            vel_b[i] *= momentum
            vel_b[i] -= lr * self.db[i]
            self.W[i] += vel_w[i]
            self.b[i] += vel_b[i]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {"sizes": np.array(self.sizes, dtype=np.int32)}
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            payload[f"W{i}"] = W
            payload[f"b{i}"] = b
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "MLP":
        data = np.load(path)
        model = cls(data["sizes"].tolist(), seed=0)
        for i in range(len(model.W)):
            model.W[i] = data[f"W{i}"].astype(np.float32)
            model.b[i] = data[f"b{i}"].astype(np.float32)
        return model
