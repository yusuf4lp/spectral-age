"""
SpectralAge: PyTorch Implementation

Architecture:
  Input (n_cpgs beta values)
    → mean-centering
    → torch.fft.rfft  (differentiable FFT layer)
    → log1p(|magnitude|)
    → FrequencyAttention  (learnable per-frequency soft weights)
    → MLP regression head → predicted age

Two model variants:
  1. SpectralAgeNet      — full learnable spectral model
  2. SpectralAgeLinear   — FFT + single linear layer (interpretable baseline)

Baseline for comparison:
  3. ElasticNetBaseline  — sklearn ElasticNet on raw beta values (Horvath style)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, field
import warnings


@dataclass
class EvalResult:
    model_name: str
    mae: float
    rmse: float
    pearson_r: float
    r2: float
    n_samples: int
    n_features: int
    predicted_ages: np.ndarray
    true_ages: np.ndarray
    extra: Dict = field(default_factory=dict)

    def __str__(self):
        return (
            f"{self.model_name:<40} "
            f"MAE={self.mae:.3f}y  RMSE={self.rmse:.3f}y  "
            f"r={self.pearson_r:.4f}  R²={self.r2:.4f}  "
            f"(n={self.n_samples})"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MethylationDataset(Dataset):
    """
    PyTorch Dataset wrapping a beta-value matrix and age labels.

    beta  : pd.DataFrame  shape (n_samples, n_cpgs)
    ages  : pd.Series     shape (n_samples,)
    """
    def __init__(self, beta: pd.DataFrame, ages: pd.Series):
        X = beta.values.astype(np.float32)
        y = ages.values.astype(np.float32)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
#  Spectral Layer  (differentiable FFT + frequency attention)
# ─────────────────────────────────────────────────────────────────────────────

class SpectralLayer(nn.Module):
    """
    Differentiable spectral feature extraction.

    Forward pass:
      1. Subtract per-sample mean (de-trend the methylation signal)
      2. torch.fft.rfft along the CpG axis
      3. log1p(|magnitude|)  — compresses dynamic range
      4. Element-wise multiply by learnable frequency weights
         (lets the network learn which frequencies predict age)

    The weights start at 1.0 (neutral) and are free to increase or
    suppress any frequency bin during training.
    """

    def __init__(self, n_cpgs: int, trainable_weights: bool = True):
        super().__init__()
        self.n_cpgs = n_cpgs
        n_freqs = n_cpgs // 2 + 1
        self.n_freqs = n_freqs
        self.bn_input = nn.BatchNorm1d(n_cpgs)
        self.bn_freq = nn.BatchNorm1d(n_freqs)

        if trainable_weights:
            self.freq_weights = nn.Parameter(torch.ones(n_freqs))
        else:
            self.register_buffer("freq_weights", torch.ones(n_freqs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn_input(x)

        fft_out = torch.fft.rfft(x, dim=1)

        mag = torch.abs(fft_out)
        mag = torch.log1p(mag)

        weights = torch.softmax(self.freq_weights, dim=0) * self.n_freqs
        mag = mag * weights.unsqueeze(0)

        mag = self.bn_freq(mag)

        return mag


# ─────────────────────────────────────────────────────────────────────────────
#  SpectralAgeNet  (full model)
# ─────────────────────────────────────────────────────────────────────────────

class SpectralAgeNet(nn.Module):
    """
    SpectralAge full neural network.

    SpectralLayer → Layer Norm → MLP → age
    """

    def __init__(
        self,
        n_cpgs: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.3,
        trainable_freq_weights: bool = True,
    ):
        super().__init__()
        self.spectral = SpectralLayer(n_cpgs, trainable_weights=trainable_freq_weights)
        n_freqs = self.spectral.n_freqs

        layers = []
        in_dim = n_freqs
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.spectral(x)           # (batch, n_freqs)
        age = self.mlp(features).squeeze(-1)  # (batch,)
        return age

    def get_frequency_importances(self) -> np.ndarray:
        """Return learned per-frequency attention weights (numpy)."""
        raw = self.spectral.freq_weights.detach().cpu()
        return torch.softmax(raw, dim=0).numpy() * self.spectral.n_freqs


class SpectralAgeLinear(nn.Module):
    """
    Interpretable variant: FFT + single linear layer.
    Analogous to Horvath (linear model), but in spectral domain.
    """

    def __init__(self, n_cpgs: int):
        super().__init__()
        self.spectral = SpectralLayer(n_cpgs, trainable_weights=True)
        n_freqs = self.spectral.n_freqs
        self.linear = nn.Linear(n_freqs, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.spectral(x)
        return self.linear(features).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
#  PyTorch Training Loop
# ─────────────────────────────────────────────────────────────────────────────

def train_spectral_model(
    model: nn.Module,
    beta: pd.DataFrame,
    ages: pd.Series,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    cv_folds: int = 5,
    device: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[EvalResult, nn.Module]:
    """
    Train SpectralAgeNet using k-fold cross-validation.

    For each fold:
      - Fit the model on train split
      - Predict on val split
    Concatenate out-of-fold predictions to get unbiased MAE/r.

    Parameters
    ----------
    model : nn.Module
        SpectralAgeNet or SpectralAgeLinear instance (re-initialized per fold)
    beta : pd.DataFrame
        Beta matrix (n_samples, n_cpgs)
    ages : pd.Series
        Chronological ages
    epochs : int
        Training epochs per fold
    lr : float
        Learning rate
    batch_size : int
        Batch size
    weight_decay : float
        L2 regularization
    cv_folds : int
        Number of CV folds
    device : str, optional
        'cuda', 'mps', or 'cpu'. Auto-detected if None.
    verbose : bool
        Print training progress

    Returns
    -------
    (EvalResult, fitted model on full data)
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dev = torch.device(device)
    if verbose:
        print(f"\nDevice: {device}")
        print(f"Training {model.__class__.__name__}  ({cv_folds}-fold CV, {epochs} epochs/fold)")
        print(f"  Input: {beta.shape[1]} CpGs  |  Samples: {len(ages)}")

    X_np = beta.values.astype(np.float32)
    y_np = ages.values.astype(np.float32)

    y_mean = y_np.mean()
    y_std = y_np.std() + 1e-8
    y_norm = (y_np - y_mean) / y_std

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(y_np))

    model_class = model.__class__
    model_kwargs = _get_model_kwargs(model)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_np)):
        fold_model = model_class(**model_kwargs).to(dev)
        optimizer = torch.optim.AdamW(
            fold_model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

        X_train = torch.from_numpy(X_np[train_idx]).to(dev)
        y_train = torch.from_numpy(y_norm[train_idx]).to(dev)
        X_val = torch.from_numpy(X_np[val_idx]).to(dev)

        ds = torch.utils.data.TensorDataset(X_train, y_train)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            fold_model.train()
            total_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = fold_model(xb)
                loss = F.mse_loss(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(fold_model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item() * len(xb)
            scheduler.step()

            if (epoch + 1) % 50 == 0 and verbose:
                avg_loss = total_loss / len(y_train)
                print(f"  Fold {fold+1}/{cv_folds}  Epoch {epoch+1}/{epochs}  Loss={avg_loss:.4f}")

        fold_model.eval()
        with torch.no_grad():
            pred_norm = fold_model(X_val).cpu().numpy()
            oof_pred[val_idx] = pred_norm * y_std + y_mean

    mae = mean_absolute_error(y_np, oof_pred)
    rmse = np.sqrt(mean_squared_error(y_np, oof_pred))
    r, _ = pearsonr(y_np, oof_pred)
    r2 = r2_score(y_np, oof_pred)

    if verbose:
        print(f"\n{cv_folds}-fold OOF results:")
        print(f"  MAE  = {mae:.3f} years")
        print(f"  RMSE = {rmse:.3f} years")
        print(f"  r    = {r:.4f}")
        print(f"  R²   = {r2:.4f}")

    full_model = model_class(**model_kwargs).to(dev)
    optimizer = torch.optim.AdamW(
        full_model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    X_all = torch.from_numpy(X_np).to(dev)
    y_all_norm = torch.from_numpy(y_norm).to(dev)
    ds_full = torch.utils.data.TensorDataset(X_all, y_all_norm)
    loader_full = DataLoader(ds_full, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        full_model.train()
        for xb, yb in loader_full:
            optimizer.zero_grad()
            pred = full_model(xb)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(full_model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

    full_model.eval()

    result = EvalResult(
        model_name=model.__class__.__name__,
        mae=mae,
        rmse=rmse,
        pearson_r=r,
        r2=r2,
        n_samples=len(y_np),
        n_features=beta.shape[1],
        predicted_ages=oof_pred,
        true_ages=y_np,
    )
    return result, full_model


# ─────────────────────────────────────────────────────────────────────────────
#  Horvath ElasticNet Baseline (sklearn)
# ─────────────────────────────────────────────────────────────────────────────

class ElasticNetBaseline:
    """
    Sklearn ElasticNet on raw beta values — Horvath-style time-domain baseline.
    """

    def __init__(self, alpha: float = 0.1, l1_ratio: float = 0.5, cv_folds: int = 5):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.cv_folds = cv_folds
        self._pipeline = None

    def fit_and_evaluate(self, beta: pd.DataFrame, ages: pd.Series) -> EvalResult:
        X = beta.values.astype(np.float64)
        y = ages.values.astype(np.float64)

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                max_iter=5000,
                random_state=42,
            )),
        ])

        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        print(f"\nFitting ElasticNet baseline ({self.cv_folds}-fold CV)...")
        print(f"  Features: {X.shape[1]} raw beta values")
        print(f"  Samples:  {X.shape[0]}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_pred = cross_val_predict(pipeline, X, y, cv=kf)

        self._pipeline = pipeline.fit(X, y)

        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r, _ = pearsonr(y, y_pred)
        r2 = r2_score(y, y_pred)

        return EvalResult(
            model_name="ElasticNet Baseline (time-domain)",
            mae=mae, rmse=rmse, pearson_r=r, r2=r2,
            n_samples=len(y), n_features=X.shape[1],
            predicted_ages=y_pred, true_ages=y,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_model_kwargs(model: nn.Module) -> dict:
    """Extract constructor kwargs from a model instance for re-initialization per fold."""
    if isinstance(model, SpectralAgeNet):
        return {
            "n_cpgs": model.spectral.n_cpgs,
            "hidden_dims": _infer_hidden_dims(model),
            "dropout": _infer_dropout(model),
            "trainable_freq_weights": True,
        }
    elif isinstance(model, SpectralAgeLinear):
        return {"n_cpgs": model.spectral.n_cpgs}
    else:
        raise ValueError(f"Unsupported model class: {type(model)}")


def _infer_hidden_dims(model: SpectralAgeNet) -> Tuple[int, ...]:
    dims = []
    for layer in model.mlp:
        if isinstance(layer, nn.Linear) and layer.out_features != 1:
            dims.append(layer.out_features)
    return tuple(dims)


def _infer_dropout(model: SpectralAgeNet) -> float:
    for layer in model.mlp:
        if isinstance(layer, nn.Dropout):
            return layer.p
    return 0.0


def compare_models(results: list) -> pd.DataFrame:
    rows = []
    baseline_mae = next(
        (r.mae for r in results if "Baseline" in r.model_name or "ElasticNet" in r.model_name),
        None,
    )
    for r in results:
        improvement = (
            f"{((baseline_mae - r.mae) / baseline_mae * 100):+.1f}%"
            if baseline_mae else "—"
        )
        rows.append({
            "Model": r.model_name,
            "MAE (years)": round(r.mae, 3),
            "RMSE (years)": round(r.rmse, 3),
            "Pearson r": round(r.pearson_r, 4),
            "R²": round(r.r2, 4),
            "vs baseline": improvement,
            "N": r.n_samples,
        })
    df = pd.DataFrame(rows).sort_values("MAE (years)")
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(df.to_string(index=False))
    print("=" * 80)
    return df
