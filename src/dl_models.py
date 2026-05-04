\
"""PyTorch tabular DL models and short-budget training utilities."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .preprocess import make_train_valid_test
from .utils import compute_metrics, ensure_dir, get_path_size_mb, resource_tracker, set_global_seed

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


class MLPHead(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(16, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x).squeeze(-1)
        return self.head(x)


class RecurrentTabular(nn.Module):
    def __init__(self, n_features: int, cell: str = "GRU", n_classes: int = 2):
        super().__init__()
        rnn_cls = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}[cell]
        self.rnn = rnn_cls(input_size=1, hidden_size=24, batch_first=True)
        self.head = nn.Linear(24, n_classes)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out, hidden = self.rnn(x)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        return self.head(hidden[-1])


class AutoEncoderHead(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(8, n_classes)

    def forward(self, x):
        z = self.encoder(x)
        return self.classifier(z)


def build_dl_model(name: str, n_features: int, n_classes: int = 2):
    if torch is None:
        raise ImportError("PyTorch is required for dl_models.py")

    name = name.upper()
    if name == "1D-CNN":
        return CNN1D(n_features, n_classes)
    if name in {"RNN", "GRU", "LSTM"}:
        return RecurrentTabular(n_features, cell=name, n_classes=n_classes)
    if name in {"AE+HEAD", "AUTOENCODER"}:
        return AutoEncoderHead(n_features, n_classes)
    if name == "MLP":
        return MLPHead(n_features, n_classes)
    raise ValueError(f"Unknown DL architecture: {name}")


def _to_loader(X, y, batch_size: int = 32, shuffle: bool = True):
    X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
    y_t = torch.tensor(np.asarray(y), dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


def train_one_dl_model(
    model,
    X_train,
    y_train,
    X_valid,
    y_valid,
    max_seconds: int = 60,
    max_epochs: int = 100,
    patience: int = 8,
    lr: float = 1e-3,
    seed: int = 42,
):
    """Train one DL model with validation-based early stopping and a soft time budget."""
    set_global_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = _to_loader(X_train, y_train, shuffle=True)
    valid_loader = _to_loader(X_valid, y_valid, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_state = None
    best_f1 = -np.inf
    bad = 0
    start = time.perf_counter()

    for _epoch in range(max_epochs):
        if time.perf_counter() - start > max_seconds:
            break

        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in valid_loader:
                logits = model(xb.to(device))
                preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                trues.extend(yb.numpy())
        f1 = compute_metrics(np.asarray(trues), np.asarray(preds))["f1_pct"]

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_dl(model, X):
    device = next(model.parameters()).device
    X_t = torch.tensor(np.asarray(X), dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        proba = torch.softmax(logits, dim=1).cpu().numpy()
    pred = np.argmax(proba, axis=1)
    return pred, proba


def run_dl_benchmark(
    df: pd.DataFrame,
    output_dir: str | Path,
    regime: str = "Light",
    seed: int = 42,
    architectures: list[str] | None = None,
) -> pd.DataFrame:
    """Run short-budget DL benchmarks."""
    if torch is None:
        raise ImportError("PyTorch is required for DL benchmarks.")

    set_global_seed(seed)
    out = ensure_dir(output_dir)
    split = make_train_valid_test(df, seed=seed, scale=True)

    if architectures is None:
        architectures = ["1D-CNN", "RNN", "GRU", "LSTM", "AE+Head"]

    max_seconds = 60 if regime.lower() == "light" else 180
    max_epochs = 60 if regime.lower() == "light" else 150
    patience = 5 if regime.lower() == "light" else 12

    rows = []
    n_classes = len(np.unique(split.y_train))
    n_features = np.asarray(split.X_train).shape[1]

    for arch in architectures:
        model_path = out / f"dl_{arch.replace('+', '_').replace('-', '_')}_{regime}_seed{seed}.pt"
        with resource_tracker() as res:
            model = build_dl_model(arch, n_features=n_features, n_classes=n_classes)
            model = train_one_dl_model(
                model,
                split.X_train,
                split.y_train,
                split.X_valid,
                split.y_valid,
                max_seconds=max_seconds,
                max_epochs=max_epochs,
                patience=patience,
                seed=seed,
            )
            y_pred, y_proba = predict_dl(model, split.X_test)
            metrics = compute_metrics(split.y_test, y_pred, y_proba)

        torch.save(model.state_dict(), model_path)
        rows.append(
            {
                "framework": f"DL_{regime}",
                "model": arch,
                "seed": seed,
                "status": "ok",
                **metrics,
                **res,
                "model_size_mb": get_path_size_mb(model_path),
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(out / f"dl_{regime.lower()}_seed{seed}.csv", index=False)
    return results
