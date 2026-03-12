from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NEURAL_TOTAL_MARKET_LADDER_PATH_VERSION = "totals_surface_neural_ladder_calibrated"
NEURAL_TOTAL_MARKET_LADDER_BUNDLE_FILENAME = "neural_total_market_ladder_bundle.pt"
NEURAL_TOTAL_MARKET_LADDER_ARTIFACT_FORMAT = "torch_bundle_v1"
NEURAL_TOTAL_MARKET_LADDER_TARGET_KIND = "conditional_total_market_ladder"
NEURAL_TOTAL_MARKET_LADDER_HIDDEN_DIMS = (32, 16)
NEURAL_TOTAL_MARKET_LADDER_DROPOUT = 0.05
LADDER_MARKETS = ("c75", "c85", "c95", "c105")


def _torch_runtime_error(path: Path | None = None) -> RuntimeError:
    detail = f" for artifact {path}" if path is not None else ""
    return RuntimeError(
        "PyTorch is required for the corners neural totals ladder path"
        f"{detail}. Install it with `python -m pip install torch`."
    )


def _require_torch(path: Path | None = None) -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise _torch_runtime_error(path) from exc


def _conditional_prior_columns(prior_total_probs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    c75 = np.clip(np.asarray(prior_total_probs["c75"], dtype=float), 0.001, 0.999)
    c85 = np.clip(np.asarray(prior_total_probs["c85"], dtype=float), 0.001, 0.999)
    c95 = np.clip(np.asarray(prior_total_probs["c95"], dtype=float), 0.001, 0.999)
    c105 = np.clip(np.asarray(prior_total_probs["c105"], dtype=float), 0.001, 0.999)
    return {
        "__prior_c75": c75,
        "__prior_c85_cond": np.clip(c85 / np.clip(c75, 1e-3, None), 0.001, 0.999),
        "__prior_c95_cond": np.clip(c95 / np.clip(c85, 1e-3, None), 0.001, 0.999),
        "__prior_c105_cond": np.clip(c105 / np.clip(c95, 1e-3, None), 0.001, 0.999),
    }


def _feature_matrix(
    features_frame: pd.DataFrame,
    *,
    prior_total_probs: dict[str, np.ndarray],
    input_columns: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    frame = features_frame.reset_index(drop=True).copy()
    for column, values in _conditional_prior_columns(prior_total_probs).items():
        frame[column] = np.asarray(values, dtype=float)
    columns = list(input_columns) if input_columns is not None else list(frame.columns)
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise RuntimeError(f"Missing neural totals ladder feature columns: {sorted(missing)}")
    matrix = (
        frame.loc[:, columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    return matrix, columns


def _build_mlp(torch: Any, input_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> Any:
    layers: list[Any] = []
    current_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.extend([torch.nn.Linear(current_dim, int(hidden_dim)), torch.nn.ReLU()])
        if float(dropout) > 0.0:
            layers.append(torch.nn.Dropout(float(dropout)))
        current_dim = int(hidden_dim)
    layers.append(torch.nn.Linear(current_dim, len(LADDER_MARKETS)))
    return torch.nn.Sequential(*layers)


def _ladder_targets(total_corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = np.asarray(total_corners, dtype=np.float32)
    c75 = (total > 7.5).astype(np.float32)
    c85 = (total > 8.5).astype(np.float32)
    c95 = (total > 9.5).astype(np.float32)
    c105 = (total > 10.5).astype(np.float32)
    targets = np.column_stack([c75, c85, c95, c105]).astype(np.float32)
    mask = np.column_stack([
        np.ones(len(total), dtype=np.float32),
        c75,
        c85,
        c95,
    ])
    return targets, mask


def fit_neural_total_market_ladder_bundle(
    features_frame: pd.DataFrame,
    *,
    prior_total_probs: dict[str, np.ndarray],
    total_corners: np.ndarray,
    hidden_dims: tuple[int, ...] = NEURAL_TOTAL_MARKET_LADDER_HIDDEN_DIMS,
    dropout: float = NEURAL_TOTAL_MARKET_LADDER_DROPOUT,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    validation_fraction: float = 0.2,
    patience: int = 20,
) -> dict[str, Any]:
    torch = _require_torch()
    x, input_columns = _feature_matrix(features_frame, prior_total_probs=prior_total_probs)
    if len(x) == 0:
        raise RuntimeError("No rows available for neural totals ladder training.")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    x_norm = (x - mean) / std
    targets, mask = _ladder_targets(total_corners)
    val_n = int(len(x_norm) * float(validation_fraction)) if len(x_norm) >= 12 else 0
    val_n = min(max(val_n, 0), max(len(x_norm) - 4, 0))
    split_idx = len(x_norm) - val_n if val_n > 0 else len(x_norm)

    x_train = torch.tensor(x_norm[:split_idx], dtype=torch.float32)
    y_train = torch.tensor(targets[:split_idx], dtype=torch.float32)
    m_train = torch.tensor(mask[:split_idx], dtype=torch.float32)
    x_val = torch.tensor(x_norm[split_idx:], dtype=torch.float32) if val_n > 0 else None
    y_val = torch.tensor(targets[split_idx:], dtype=torch.float32) if val_n > 0 else None
    m_val = torch.tensor(mask[split_idx:], dtype=torch.float32) if val_n > 0 else None

    model = _build_mlp(torch, x_train.shape[1], tuple(int(dim) for dim in hidden_dims), float(dropout))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    def masked_loss(logits: Any, targets_tensor: Any, mask_tensor: Any) -> Any:
        raw_loss = loss_fn(logits, targets_tensor)
        return (raw_loss * mask_tensor).sum() / mask_tensor.sum().clamp_min(1.0)

    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    stale_epochs = 0
    for _epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        train_logits = model(x_train)
        train_loss = masked_loss(train_logits, y_train, m_train)
        train_loss.backward()
        optimizer.step()
        metric = float(train_loss.detach().cpu())
        if x_val is not None and y_val is not None and m_val is not None and len(x_val) > 0:
            model.eval()
            with torch.no_grad():
                metric = float(masked_loss(model(x_val), y_val, m_val).detach().cpu())
        if metric + 1e-8 < best_loss:
            best_loss = metric
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                break
    model.load_state_dict(best_state)
    return {
        "state_dict": best_state,
        "input_mean": mean.astype(float).tolist(),
        "input_std": std.astype(float).tolist(),
        "input_columns": input_columns,
        "hidden_dims": [int(dim) for dim in hidden_dims],
        "dropout": float(dropout),
        "target_kind": NEURAL_TOTAL_MARKET_LADDER_TARGET_KIND,
        "artifact_format": NEURAL_TOTAL_MARKET_LADDER_ARTIFACT_FORMAT,
        "ladder_markets": list(LADDER_MARKETS),
    }


def save_neural_total_market_ladder_bundle(bundle: dict[str, Any], path: Path) -> None:
    torch = _require_torch(path)
    torch.save(bundle, path)


def load_neural_total_market_ladder_bundle(path: Path) -> dict[str, Any]:
    torch = _require_torch(path)
    if not path.exists():
        raise RuntimeError(f"Missing neural totals ladder sidecar: {path}")
    bundle = torch.load(path, map_location="cpu")
    if not isinstance(bundle, dict):
        raise RuntimeError(f"Invalid neural totals ladder sidecar payload: {path}")
    return bundle


def predict_neural_total_market_ladder_probs(
    bundle: dict[str, Any],
    features_frame: pd.DataFrame,
    *,
    prior_total_probs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    torch = _require_torch()
    input_columns = [str(col) for col in (bundle.get("input_columns") or [])]
    x, _ = _feature_matrix(
        features_frame,
        prior_total_probs=prior_total_probs,
        input_columns=input_columns,
    )
    mean = np.asarray(bundle.get("input_mean") or [], dtype=np.float32)
    std = np.asarray(bundle.get("input_std") or [], dtype=np.float32)
    if mean.shape[0] != x.shape[1] or std.shape[0] != x.shape[1]:
        raise RuntimeError("Neural totals ladder normalization stats do not match the saved feature columns.")
    std = np.where(std < 1e-6, 1.0, std)
    x_norm = (x - mean) / std
    hidden_dims = tuple(int(dim) for dim in (bundle.get("hidden_dims") or NEURAL_TOTAL_MARKET_LADDER_HIDDEN_DIMS))
    dropout = float(bundle.get("dropout") or 0.0)
    model = _build_mlp(torch, x.shape[1], hidden_dims, dropout)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    with torch.no_grad():
        raw = torch.sigmoid(model(torch.tensor(x_norm, dtype=torch.float32))).detach().cpu().numpy()
    stage = np.clip(raw, 0.001, 0.999)
    c75 = stage[:, 0]
    c85 = np.clip(c75 * stage[:, 1], 0.001, 0.999)
    c95 = np.clip(c85 * stage[:, 2], 0.001, 0.999)
    c105 = np.clip(c95 * stage[:, 3], 0.001, 0.999)
    return {
        "c75": c75,
        "c85": c85,
        "c95": c95,
        "c105": c105,
    }