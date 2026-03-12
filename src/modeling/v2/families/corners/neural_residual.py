from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


NEURAL_SHARE_RESIDUAL_PATH_VERSION = "totals_first_neural_share_residual"
NEURAL_TOTAL_SHARE_RESIDUAL_PATH_VERSION = "totals_first_neural_total_share_residual"
NEURAL_TOTAL_RESIDUAL_BUNDLE_FILENAME = "neural_total_residual_bundle.pt"
NEURAL_SHARE_RESIDUAL_BUNDLE_FILENAME = "neural_share_residual_bundle.pt"
NEURAL_SHARE_RESIDUAL_ARTIFACT_FORMAT = "torch_bundle_v1"
NEURAL_TOTAL_RESIDUAL_TARGET_KIND = "total_corners_residual"
NEURAL_SHARE_RESIDUAL_TARGET_KIND = "home_share_residual"
NEURAL_SHARE_RESIDUAL_HIDDEN_DIMS = (32, 16)
NEURAL_SHARE_RESIDUAL_DROPOUT = 0.05
NEURAL_TOTAL_RESIDUAL_DEFAULT_BOUND = 3.0
NEURAL_SHARE_RESIDUAL_DEFAULT_BOUND = 0.2


def _torch_runtime_error(path: Path | None = None) -> RuntimeError:
    detail = f" for artifact {path}" if path is not None else ""
    return RuntimeError(
        "PyTorch is required for the corners neural residual path"
        f"{detail}. Install it with `python -m pip install torch`."
    )


def _require_torch(path: Path | None = None) -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        raise _torch_runtime_error(path) from exc


def _feature_matrix(
    features_frame: pd.DataFrame,
    *,
    base_total_mu: np.ndarray,
    base_home_share: np.ndarray,
    input_columns: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    frame = features_frame.reset_index(drop=True).copy()
    frame["__base_total_mu"] = np.asarray(base_total_mu, dtype=float)
    frame["__base_home_share"] = np.asarray(base_home_share, dtype=float)
    columns = list(input_columns) if input_columns is not None else list(frame.columns)
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise RuntimeError(f"Missing neural residual feature columns: {sorted(missing)}")
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
    layers.append(torch.nn.Linear(current_dim, 1))
    return torch.nn.Sequential(*layers)


def _fit_neural_residual_bundle(
    features_frame: pd.DataFrame,
    *,
    base_total_mu: np.ndarray,
    base_home_share: np.ndarray,
    target_delta: np.ndarray,
    target_kind: str,
    hidden_dims: tuple[int, ...] = NEURAL_SHARE_RESIDUAL_HIDDEN_DIMS,
    dropout: float = NEURAL_SHARE_RESIDUAL_DROPOUT,
    delta_bound: float = NEURAL_TOTAL_RESIDUAL_DEFAULT_BOUND,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    validation_fraction: float = 0.2,
    patience: int = 20,
) -> dict[str, Any]:
    torch = _require_torch()
    x, input_columns = _feature_matrix(
        features_frame,
        base_total_mu=base_total_mu,
        base_home_share=base_home_share,
    )
    target_delta = np.clip(np.asarray(target_delta, dtype=np.float32), -float(delta_bound), float(delta_bound))
    if len(x) == 0:
        raise RuntimeError("No rows available for neural residual training.")
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-6] = 1.0
    x_norm = (x - mean) / std
    val_n = int(len(x_norm) * float(validation_fraction)) if len(x_norm) >= 12 else 0
    val_n = min(max(val_n, 0), max(len(x_norm) - 4, 0))
    split_idx = len(x_norm) - val_n if val_n > 0 else len(x_norm)
    x_train = torch.tensor(x_norm[:split_idx], dtype=torch.float32)
    y_train = torch.tensor(target_delta[:split_idx].reshape(-1, 1), dtype=torch.float32)
    x_val = torch.tensor(x_norm[split_idx:], dtype=torch.float32) if val_n > 0 else None
    y_val = torch.tensor(target_delta[split_idx:].reshape(-1, 1), dtype=torch.float32) if val_n > 0 else None
    model = _build_mlp(torch, x_train.shape[1], tuple(int(dim) for dim in hidden_dims), float(dropout))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    loss_fn = torch.nn.MSELoss()
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    stale_epochs = 0
    for _epoch in range(int(epochs)):
        model.train()
        optimizer.zero_grad()
        train_pred = torch.tanh(model(x_train)) * float(delta_bound)
        train_loss = loss_fn(train_pred, y_train)
        train_loss.backward()
        optimizer.step()
        metric = float(train_loss.detach().cpu())
        if x_val is not None and y_val is not None and len(x_val) > 0:
            model.eval()
            with torch.no_grad():
                val_pred = torch.tanh(model(x_val)) * float(delta_bound)
                metric = float(loss_fn(val_pred, y_val).detach().cpu())
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
        "delta_bound": float(delta_bound),
        "target_kind": str(target_kind),
        "artifact_format": NEURAL_SHARE_RESIDUAL_ARTIFACT_FORMAT,
    }


def fit_neural_total_residual_bundle(
    features_frame: pd.DataFrame,
    *,
    base_total_mu: np.ndarray,
    base_home_share: np.ndarray,
    target_total_mu: np.ndarray,
    hidden_dims: tuple[int, ...] = NEURAL_SHARE_RESIDUAL_HIDDEN_DIMS,
    dropout: float = NEURAL_SHARE_RESIDUAL_DROPOUT,
    delta_bound: float = NEURAL_TOTAL_RESIDUAL_DEFAULT_BOUND,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    validation_fraction: float = 0.2,
    patience: int = 20,
) -> dict[str, Any]:
    return _fit_neural_residual_bundle(
        features_frame,
        base_total_mu=base_total_mu,
        base_home_share=base_home_share,
        target_delta=(
            np.asarray(target_total_mu, dtype=np.float32)
            - np.asarray(base_total_mu, dtype=np.float32)
        ),
        target_kind=NEURAL_TOTAL_RESIDUAL_TARGET_KIND,
        hidden_dims=hidden_dims,
        dropout=dropout,
        delta_bound=delta_bound,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        validation_fraction=validation_fraction,
        patience=patience,
    )


def fit_neural_share_residual_bundle(
    features_frame: pd.DataFrame,
    *,
    base_total_mu: np.ndarray,
    base_home_share: np.ndarray,
    target_home_share: np.ndarray,
    hidden_dims: tuple[int, ...] = NEURAL_SHARE_RESIDUAL_HIDDEN_DIMS,
    dropout: float = NEURAL_SHARE_RESIDUAL_DROPOUT,
    delta_bound: float = NEURAL_SHARE_RESIDUAL_DEFAULT_BOUND,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    validation_fraction: float = 0.2,
    patience: int = 20,
) -> dict[str, Any]:
    return _fit_neural_residual_bundle(
        features_frame,
        base_total_mu=base_total_mu,
        base_home_share=base_home_share,
        target_delta=(
            np.asarray(target_home_share, dtype=np.float32)
            - np.asarray(base_home_share, dtype=np.float32)
        ),
        target_kind=NEURAL_SHARE_RESIDUAL_TARGET_KIND,
        hidden_dims=hidden_dims,
        dropout=dropout,
        delta_bound=delta_bound,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        validation_fraction=validation_fraction,
        patience=patience,
    )


def save_neural_residual_bundle(bundle: dict[str, Any], path: Path) -> None:
    torch = _require_torch(path)
    torch.save(bundle, path)


def load_neural_residual_bundle(path: Path) -> dict[str, Any]:
    torch = _require_torch(path)
    if not path.exists():
        raise RuntimeError(f"Missing neural residual sidecar: {path}")
    bundle = torch.load(path, map_location="cpu")
    if not isinstance(bundle, dict):
        raise RuntimeError(f"Invalid neural residual sidecar payload: {path}")
    return bundle


def _predict_neural_residual_delta(
    bundle: dict[str, Any],
    features_frame: pd.DataFrame,
    *,
    base_total_mu: np.ndarray,
    base_home_share: np.ndarray,
) -> np.ndarray:
    torch = _require_torch()
    input_columns = [str(col) for col in (bundle.get("input_columns") or [])]
    x, _ = _feature_matrix(
        features_frame,
        base_total_mu=base_total_mu,
        base_home_share=base_home_share,
        input_columns=input_columns,
    )
    mean = np.asarray(bundle.get("input_mean") or [], dtype=np.float32)
    std = np.asarray(bundle.get("input_std") or [], dtype=np.float32)
    if mean.shape[0] != x.shape[1] or std.shape[0] != x.shape[1]:
        raise RuntimeError("Neural residual normalization stats do not match the saved feature columns.")
    std = np.where(std < 1e-6, 1.0, std)
    x_norm = (x - mean) / std
    hidden_dims = tuple(int(dim) for dim in (bundle.get("hidden_dims") or NEURAL_SHARE_RESIDUAL_HIDDEN_DIMS))
    dropout = float(bundle.get("dropout") or 0.0)
    delta_bound = float(bundle.get("delta_bound") or NEURAL_TOTAL_RESIDUAL_DEFAULT_BOUND)
    model = _build_mlp(torch, x.shape[1], hidden_dims, dropout)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    with torch.no_grad():
        raw = model(torch.tensor(x_norm, dtype=torch.float32)).detach().cpu().numpy().reshape(-1)
    return np.tanh(raw) * delta_bound


def save_neural_share_residual_bundle(bundle: dict[str, Any], path: Path) -> None:
    save_neural_residual_bundle(bundle, path)


def save_neural_total_residual_bundle(bundle: dict[str, Any], path: Path) -> None:
    save_neural_residual_bundle(bundle, path)


def load_neural_share_residual_bundle(path: Path) -> dict[str, Any]:
    return load_neural_residual_bundle(path)


def load_neural_total_residual_bundle(path: Path) -> dict[str, Any]:
    return load_neural_residual_bundle(path)


def predict_neural_share_residual_delta(
    bundle: dict[str, Any],
    features_frame: pd.DataFrame,
    *,
    base_total_mu: np.ndarray,
    base_home_share: np.ndarray,
) -> np.ndarray:
    return _predict_neural_residual_delta(
        bundle,
        features_frame,
        base_total_mu=base_total_mu,
        base_home_share=base_home_share,
    )


def predict_neural_total_residual_delta(
    bundle: dict[str, Any],
    features_frame: pd.DataFrame,
    *,
    base_total_mu: np.ndarray,
    base_home_share: np.ndarray,
) -> np.ndarray:
    return _predict_neural_residual_delta(
        bundle,
        features_frame,
        base_total_mu=base_total_mu,
        base_home_share=base_home_share,
    )