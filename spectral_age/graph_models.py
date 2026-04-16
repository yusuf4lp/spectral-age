"""
SpectralAge — Gelismis Modeller

FFT yeterli olmadığında devreye giren üç yaklaşım:

1. SpectralForest      — FFT features → GBM/RF ensemble
2. SparseGraphNet      — CpG'ler arası genomik yakınlık sparse graph + attention
3. LocalPlasticityNet  — Hebbian plasticity ile per-sample adaptif ağırlıklar
                         (World Model / Miconi et al. 2018 tarzı)
4. HybridSpectralAge   — FFT + Graph + Plasticity streams → gated fusion → age

Bağımlılık: sadece torch, numpy, scipy, sklearn — ekstra graph kütüphanesi yok.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from scipy.sparse import csr_matrix
from typing import Optional, Tuple, List
from dataclasses import dataclass
import warnings

from .models import EvalResult
from .spectral_features import SpectralFeatureSet


# ═══════════════════════════════════════════════════════════════════════════
#  1. SpectralForest — FFT features → GBM/RF ensemble
# ═══════════════════════════════════════════════════════════════════════════

class SpectralForest:
    """
    FFT magnitude features → Gradient Boosting / Random Forest.

    FFT zayıf kaldığında nonlinear ensemble ile frekans etkileşimlerini
    yakalayarak daha iyi tahmin üretir. Feature importance'lar hangi
    frekans bileşenlerinin önemli olduğunu doğrudan gösterir.
    """

    def __init__(
        self,
        method: str = "gbm",
        n_estimators: int = 500,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        self.method = method
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.cv_folds = cv_folds
        self.random_state = random_state
        self._model = None
        self._scaler = None
        self._feature_importances = None

    def fit_and_evaluate(
        self,
        spectral_features: SpectralFeatureSet,
        ages: pd.Series,
    ) -> EvalResult:
        X = spectral_features.magnitudes.astype(np.float64)
        y = ages.values.astype(np.float64)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        if self.method == "gbm":
            base = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=0.8,
                random_state=self.random_state,
            )
        else:
            base = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
            )

        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        print(f"\nFitting SpectralForest ({self.method}) on {X.shape[1]} FFT features...")

        y_pred = cross_val_predict(base, X_scaled, y, cv=kf)

        self._model = base.fit(X_scaled, y)
        self._feature_importances = self._model.feature_importances_

        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r, _ = pearsonr(y, y_pred)
        r2 = r2_score(y, y_pred)

        return EvalResult(
            model_name=f"SpectralForest ({self.method})",
            mae=mae, rmse=rmse, pearson_r=r, r2=r2,
            n_samples=len(y), n_features=X.shape[1],
            predicted_ages=y_pred, true_ages=y,
            extra={"feature_importances": self._feature_importances},
        )

    def get_top_frequencies(self, n: int = 10) -> np.ndarray:
        if self._feature_importances is None:
            raise RuntimeError("Fit first.")
        return np.argsort(self._feature_importances)[::-1][:n]


# ═══════════════════════════════════════════════════════════════════════════
#  2. SparseGraphNet — Genomik yakınlık grafiği + sparse attention
# ═══════════════════════════════════════════════════════════════════════════

def build_adjacency_from_positions(
    cpg_positions: pd.DataFrame,
    cpg_order: list,
    k_neighbors: int = 10,
    max_distance_bp: int = 500_000,
) -> torch.Tensor:
    """
    CpG siteleri arasında genomik yakınlık tabanlı sparse adjacency matrisi.

    Aynı kromozom üzerinde, genomik pozisyona göre en yakın k komşuyu bağlar.
    max_distance_bp'den uzak CpG'ler bağlanmaz (uzak CpG'ler biyolojik olarak
    bağımsız).

    Returns
    -------
    edge_index : torch.LongTensor, shape (2, n_edges) — COO format
    """
    pos_df = cpg_positions.set_index("cpg_id")
    n = len(cpg_order)

    src_list = []
    dst_list = []

    chr_groups = {}
    for i, cpg in enumerate(cpg_order):
        if cpg in pos_df.index:
            row = pos_df.loc[cpg]
            chrom = str(row["chromosome"])
            position = float(row["position"])
            chr_groups.setdefault(chrom, []).append((i, position))

    for chrom, nodes in chr_groups.items():
        nodes.sort(key=lambda x: x[1])
        for idx_in_group, (i, pos_i) in enumerate(nodes):
            start = max(0, idx_in_group - k_neighbors)
            end = min(len(nodes), idx_in_group + k_neighbors + 1)
            for j_in_group in range(start, end):
                if j_in_group == idx_in_group:
                    continue
                j, pos_j = nodes[j_in_group]
                if abs(pos_i - pos_j) <= max_distance_bp:
                    src_list.append(i)
                    dst_list.append(j)

    if not src_list:
        for i in range(n):
            for j in [max(0, i-1), min(n-1, i+1)]:
                if j != i:
                    src_list.append(i)
                    dst_list.append(j)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    print(f"  Graph: {n} nodes, {len(src_list)} edges (k={k_neighbors}, max_dist={max_distance_bp}bp)")
    return edge_index


def build_sequential_adjacency(n_cpgs: int, k: int = 5) -> torch.Tensor:
    """Pozisyon bilgisi yoksa sıralı komşuluk grafiği."""
    src, dst = [], []
    for i in range(n_cpgs):
        for j in range(max(0, i - k), min(n_cpgs, i + k + 1)):
            if j != i:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


class SparseGraphAttention(nn.Module):
    """
    Sparse Graph Attention Layer (GAT tarzı, PyG bağımsız).

    Her CpG node'u komşularından attention-weighted mesajlar alır.
    Attention: a^T [Wh_i || Wh_j] → softmax → mesaj toplama
    """

    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        assert out_dim % n_heads == 0

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.attn = nn.Parameter(torch.randn(n_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.attn.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x          : (batch, n_nodes, in_dim)
        edge_index : (2, n_edges) — paylaşılan topoloji
        return     : (batch, n_nodes, out_dim)
        """
        B, N, _ = x.shape
        h = self.W(x)
        h = h.view(B, N, self.n_heads, self.head_dim)

        src, dst = edge_index[0], edge_index[1]
        n_edges = src.shape[0]

        h_src = h[:, src]
        h_dst = h[:, dst]

        attn_input = torch.cat([h_src, h_dst], dim=-1)
        attn_scores = (attn_input * self.attn.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        attn_scores = F.leaky_relu(attn_scores, 0.2)

        attn_max = torch.zeros(B, N, self.n_heads, device=x.device)
        attn_max.scatter_reduce_(
            1,
            dst.unsqueeze(0).unsqueeze(-1).expand(B, n_edges, self.n_heads),
            attn_scores,
            reduce="amax",
            include_self=True,
        )
        attn_scores = attn_scores - attn_max[:, dst]
        attn_exp = torch.exp(attn_scores)

        attn_sum = torch.zeros(B, N, self.n_heads, device=x.device)
        attn_sum.scatter_add_(
            1,
            dst.unsqueeze(0).unsqueeze(-1).expand(B, n_edges, self.n_heads),
            attn_exp,
        )
        attn_norm = attn_exp / (attn_sum[:, dst] + 1e-8)
        attn_norm = self.dropout(attn_norm)

        messages = h_src * attn_norm.unsqueeze(-1)
        out = torch.zeros(B, N, self.n_heads, self.head_dim, device=x.device)
        out.scatter_add_(
            1,
            dst.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(B, n_edges, self.n_heads, self.head_dim),
            messages,
        )

        return out.reshape(B, N, self.out_dim)


class SparseGraphNet(nn.Module):
    """
    Sparse Graph Neural Network for epigenetic age prediction.

    Her CpG sitesi bir node. Beta değeri node feature.
    Genomik yakınlık sparse graph attention ile işlenir.
    Global readout → MLP → age.

    graph mesajları:  methylation beta → embed → GAT × 2 → global pool → MLP → age
    """

    def __init__(
        self,
        n_cpgs: int,
        embed_dim: int = 64,
        gat_dim: int = 64,
        n_heads: int = 4,
        mlp_dims: Tuple[int, ...] = (128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_cpgs = n_cpgs

        self.node_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        self.gat1 = SparseGraphAttention(embed_dim, gat_dim, n_heads=n_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(gat_dim)

        self.gat2 = SparseGraphAttention(gat_dim, gat_dim, n_heads=n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(gat_dim)

        layers = []
        in_d = gat_dim
        for d in mlp_dims:
            layers += [nn.Linear(in_d, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout)]
            in_d = d
        layers.append(nn.Linear(in_d, 1))
        self.readout = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x          : (batch, n_cpgs) beta values
        edge_index : (2, n_edges)
        """
        B, N = x.shape
        h = x.unsqueeze(-1)
        h = self.node_embed(h)

        h = h + self.norm1(F.gelu(self.gat1(h, edge_index)))
        h = h + self.norm2(F.gelu(self.gat2(h, edge_index)))

        g = h.mean(dim=1)
        return self.readout(g).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
#  3. LocalPlasticityNet — World Model tarzı Hebbian plasticity
# ═══════════════════════════════════════════════════════════════════════════

class HebbianPlasticLayer(nn.Module):
    """
    Differentiable Hebbian Plasticity Layer (Miconi et al. 2018).

    Her bağlantının iki bileşeni var:
      - W (slow weights): normal backprop ile öğrenilir
      - A (Hebbian trace): forward pass sırasında per-sample güncellenir
      - α (plasticity coefficient): hangi bağlantıların adaptif olacağını öğrenir

    Etkili ağırlık: W + α ⊙ A

    Bu sayede model, her yeni sample'ın yerel metilasyon yapısına
    (world model gibi) anında adapte olur — özellikle farklı doku tipleri
    veya yaş grupları arasındaki dağılım kaymaları için kritik.
    """

    def __init__(self, in_features: int, out_features: int, plasticity_init: float = 0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.alpha = nn.Parameter(torch.ones(out_features, in_features) * plasticity_init)
        self.eta = nn.Parameter(torch.tensor(0.1))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_features)

        Her sample için Hebbian trace hesapla ve uygula.
        Trace: A_ij += η * y_i * x_j   (outer product, Hebb kuralı)
        Output: y = (W + α⊙A) x + b
        """
        B, D = x.shape

        y_slow = F.linear(x, self.W, self.bias)

        x_norm = x / (x.norm(dim=-1, keepdim=True) + 1e-8)

        y_pre = torch.tanh(y_slow)
        hebb_trace = self.eta * torch.bmm(
            y_pre.unsqueeze(-1),
            x_norm.unsqueeze(-2),
        )

        alpha_clamped = torch.sigmoid(self.alpha)
        plastic_contribution = torch.bmm(
            (alpha_clamped.unsqueeze(0) * hebb_trace),
            x.unsqueeze(-1),
        ).squeeze(-1)

        return y_slow + plastic_contribution


class LocalPlasticityNet(nn.Module):
    """
    World Model tarzı Epigenetik Saat.

    FFT spektral özellikler → Hebbian Plastic Layers → age

    Neden plasticity?
    - Farklı doku tipleri farklı metilasyon dağılımları gösterir
    - Yaş grupları arasında non-stationary dağılım kaymaları var
    - Plasticity, modelin her sample'a lokal olarak adapte olmasını sağlar
    - World model analojisi: model, metilasyonun "dünyasını" her sample için
      yeniden modeller

    Slow weights (W): popülasyon düzeyinde yaşlanma desenleri
    Fast weights (α⊙A): bireysel/doku-spesifik adaptasyon
    """

    def __init__(
        self,
        n_cpgs: int,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        use_spectral: bool = True,
        dropout: float = 0.2,
        plasticity_init: float = 0.01,
    ):
        super().__init__()
        self.n_cpgs = n_cpgs
        self.use_spectral = use_spectral

        if use_spectral:
            n_freqs = n_cpgs // 2 + 1
            self.spectral_proj = nn.Sequential(
                nn.Linear(n_cpgs, n_cpgs),
            )
            in_dim = n_freqs
        else:
            in_dim = n_cpgs

        layers = nn.ModuleList()
        norms = nn.ModuleList()
        drops = nn.ModuleList()

        for h in hidden_dims:
            layers.append(HebbianPlasticLayer(in_dim, h, plasticity_init=plasticity_init))
            norms.append(nn.LayerNorm(h))
            drops.append(nn.Dropout(dropout))
            in_dim = h

        self.plastic_layers = layers
        self.layer_norms = norms
        self.dropouts = drops
        self.head = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_cpgs)"""
        if self.use_spectral:
            x_centered = x - x.mean(dim=1, keepdim=True)
            fft_out = torch.fft.rfft(x_centered, dim=1)
            h = torch.log1p(torch.abs(fft_out))
        else:
            h = x

        for plastic, norm, drop in zip(self.plastic_layers, self.layer_norms, self.dropouts):
            h = drop(F.gelu(norm(plastic(h))))

        return self.head(h).squeeze(-1)

    def get_plasticity_stats(self) -> dict:
        stats = {}
        for i, layer in enumerate(self.plastic_layers):
            alpha = torch.sigmoid(layer.alpha).detach()
            stats[f"layer_{i}"] = {
                "alpha_mean": float(alpha.mean()),
                "alpha_std": float(alpha.std()),
                "alpha_max": float(alpha.max()),
                "eta": float(layer.eta.detach()),
                "n_plastic_connections": int((alpha > 0.1).sum()),
                "total_connections": int(alpha.numel()),
            }
        return stats


# ═══════════════════════════════════════════════════════════════════════════
#  4. HybridSpectralAge — FFT + Graph + Plasticity → Gated Fusion
# ═══════════════════════════════════════════════════════════════════════════

class GatedFusion(nn.Module):
    """
    Üç stream'i dinamik olarak ağırlıklandıran gated fusion.
    Her sample için hangi stream'in daha güvenilir olduğunu öğrenir.
    """

    def __init__(self, stream_dim: int, n_streams: int = 3):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(stream_dim * n_streams, n_streams * 2),
            nn.GELU(),
            nn.Linear(n_streams * 2, n_streams),
        )
        self.n_streams = n_streams

    def forward(self, *streams: torch.Tensor) -> torch.Tensor:
        concat = torch.cat(streams, dim=-1)
        gate_logits = self.gate(concat)
        gate_weights = F.softmax(gate_logits, dim=-1)

        stacked = torch.stack(streams, dim=-1)
        fused = (stacked * gate_weights.unsqueeze(-2)).sum(dim=-1)
        return fused


class HybridSpectralAge(nn.Module):
    """
    Üç stream'i birleştiren hybrid model:

    Stream 1 (FFT):        beta → rfft → magnitude → MLP → features
    Stream 2 (Graph):      beta → node_embed → GAT → global_pool → features
    Stream 3 (Plasticity): beta → rfft → HebbianPlastic → features

    Fusion: GatedFusion(stream1, stream2, stream3) → MLP → age

    Her stream 'fusion_dim' boyutunda bir temsil üretir.
    Gate mekanizması her sample için hangi stream'in güvenilir olduğuna karar verir.
    """

    def __init__(
        self,
        n_cpgs: int,
        fusion_dim: int = 64,
        fft_hidden: int = 128,
        graph_hidden: int = 64,
        plasticity_hidden: int = 128,
        n_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_cpgs = n_cpgs
        n_freqs = n_cpgs // 2 + 1

        self.fft_stream = nn.Sequential(
            nn.Linear(n_freqs, fft_hidden),
            nn.LayerNorm(fft_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fft_hidden, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.fft_freq_weights = nn.Parameter(torch.ones(n_freqs))

        self.graph_embed = nn.Sequential(
            nn.Linear(1, graph_hidden),
            nn.LayerNorm(graph_hidden),
            nn.GELU(),
        )
        self.graph_attn = SparseGraphAttention(graph_hidden, graph_hidden, n_heads=n_heads, dropout=dropout)
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_hidden, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

        self.plastic1 = HebbianPlasticLayer(n_freqs, plasticity_hidden, plasticity_init=0.01)
        self.plastic_norm = nn.LayerNorm(plasticity_hidden)
        self.plastic_proj = nn.Sequential(
            nn.Linear(plasticity_hidden, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )

        self.fusion = GatedFusion(fusion_dim, n_streams=3)

        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x          : (batch, n_cpgs) beta values
        edge_index : (2, n_edges), optional — None ise sadece FFT + plasticity çalışır
        """
        B = x.shape[0]
        x_centered = x - x.mean(dim=1, keepdim=True)

        fft_out = torch.fft.rfft(x_centered, dim=1)
        mag = torch.log1p(torch.abs(fft_out))
        freq_w = torch.softmax(self.fft_freq_weights, dim=0) * len(self.fft_freq_weights)
        mag_weighted = mag * freq_w.unsqueeze(0)

        fft_feat = self.fft_stream(mag_weighted)

        if edge_index is not None:
            h = x.unsqueeze(-1)
            h = self.graph_embed(h)
            h = h + F.gelu(self.graph_attn(h, edge_index))
            graph_feat = self.graph_proj(h.mean(dim=1))
        else:
            graph_feat = torch.zeros(B, fft_feat.shape[-1], device=x.device)

        plastic_out = self.plastic_norm(F.gelu(self.plastic1(mag)))
        plastic_feat = self.plastic_proj(plastic_out)

        fused = self.fusion(fft_feat, graph_feat, plastic_feat)
        return self.head(fused).squeeze(-1)

    def get_gate_distribution(self, x: torch.Tensor, edge_index: torch.Tensor = None) -> np.ndarray:
        """Her sample için gate ağırlıklarını döndür (hangi stream ne kadar önemli)."""
        self.eval()
        with torch.no_grad():
            x_centered = x - x.mean(dim=1, keepdim=True)
            fft_out = torch.fft.rfft(x_centered, dim=1)
            mag = torch.log1p(torch.abs(fft_out))
            freq_w = torch.softmax(self.fft_freq_weights, dim=0) * len(self.fft_freq_weights)
            mag_weighted = mag * freq_w.unsqueeze(0)
            fft_feat = self.fft_stream(mag_weighted)

            B = x.shape[0]
            if edge_index is not None:
                h = x.unsqueeze(-1)
                h = self.graph_embed(h)
                h = h + F.gelu(self.graph_attn(h, edge_index))
                graph_feat = self.graph_proj(h.mean(dim=1))
            else:
                graph_feat = torch.zeros(B, fft_feat.shape[-1], device=x.device)

            plastic_out = self.plastic_norm(F.gelu(self.plastic1(mag)))
            plastic_feat = self.plastic_proj(plastic_out)

            concat = torch.cat([fft_feat, graph_feat, plastic_feat], dim=-1)
            gate_logits = self.fusion.gate(concat)
            gates = F.softmax(gate_logits, dim=-1).cpu().numpy()
        return gates


# ═══════════════════════════════════════════════════════════════════════════
#  Training utilities
# ═══════════════════════════════════════════════════════════════════════════

def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_graph_model(
    model: nn.Module,
    beta: pd.DataFrame,
    ages: pd.Series,
    edge_index: torch.Tensor,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
    cv_folds: int = 5,
    verbose: bool = True,
) -> Tuple[EvalResult, nn.Module]:
    """
    Graph/Hybrid/Plasticity model eğitimi — edge_index destekli.
    """
    dev = _get_device()
    if verbose:
        print(f"\nDevice: {dev}")
        name = model.__class__.__name__
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Training {name} ({total_params:,} params, {cv_folds}-fold CV, {epochs} ep)")

    X_np = beta.values.astype(np.float32)
    y_np = ages.values.astype(np.float32)
    edge_dev = edge_index.to(dev)

    y_mean = y_np.mean()
    y_std = y_np.std() + 1e-8
    y_norm = (y_np - y_mean) / y_std

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    oof_pred = np.zeros(len(y_np))

    model_class = model.__class__
    model_kwargs = _extract_kwargs(model)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_np)):
        fold_model = model_class(**model_kwargs).to(dev)
        optimizer = torch.optim.AdamW(fold_model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

        X_train = torch.from_numpy(X_np[train_idx]).to(dev)
        y_train = torch.from_numpy(y_norm[train_idx]).to(dev)
        X_val = torch.from_numpy(X_np[val_idx]).to(dev)

        for epoch in range(epochs):
            fold_model.train()
            perm = torch.randperm(len(X_train))
            total_loss = 0.0
            for start in range(0, len(X_train), batch_size):
                idx = perm[start:start + batch_size]
                xb, yb = X_train[idx], y_train[idx]
                optimizer.zero_grad()

                if isinstance(fold_model, (SparseGraphNet, HybridSpectralAge)):
                    pred = fold_model(xb, edge_dev)
                else:
                    pred = fold_model(xb)

                loss = F.mse_loss(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(fold_model.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item() * len(xb)
            scheduler.step()

            if verbose and (epoch + 1) % 50 == 0:
                avg = total_loss / len(X_train)
                print(f"  Fold {fold+1}/{cv_folds}  Epoch {epoch+1}/{epochs}  Loss={avg:.4f}")

        fold_model.eval()
        with torch.no_grad():
            if isinstance(fold_model, (SparseGraphNet, HybridSpectralAge)):
                pred_norm = fold_model(X_val, edge_dev).cpu().numpy()
            else:
                pred_norm = fold_model(X_val).cpu().numpy()
            oof_pred[val_idx] = pred_norm * y_std + y_mean

    mae = mean_absolute_error(y_np, oof_pred)
    rmse = np.sqrt(mean_squared_error(y_np, oof_pred))
    r, _ = pearsonr(y_np, oof_pred)
    r2 = r2_score(y_np, oof_pred)

    if verbose:
        print(f"\n{cv_folds}-fold OOF:  MAE={mae:.3f}y  r={r:.4f}  R²={r2:.4f}")

    full_model = model_class(**model_kwargs).to(dev)
    optimizer = torch.optim.AdamW(full_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    X_all = torch.from_numpy(X_np).to(dev)
    y_all_norm = torch.from_numpy(y_norm).to(dev)

    for epoch in range(epochs):
        full_model.train()
        perm = torch.randperm(len(X_all))
        for start in range(0, len(X_all), batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            if isinstance(full_model, (SparseGraphNet, HybridSpectralAge)):
                pred = full_model(X_all[idx], edge_dev)
            else:
                pred = full_model(X_all[idx])
            loss = F.mse_loss(pred, y_all_norm[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(full_model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

    full_model.eval()

    result = EvalResult(
        model_name=model.__class__.__name__,
        mae=mae, rmse=rmse, pearson_r=r, r2=r2,
        n_samples=len(y_np), n_features=beta.shape[1],
        predicted_ages=oof_pred, true_ages=y_np,
    )
    return result, full_model


def _extract_kwargs(model: nn.Module) -> dict:
    if isinstance(model, SparseGraphNet):
        return {
            "n_cpgs": model.n_cpgs,
            "embed_dim": model.node_embed[0].out_features,
            "gat_dim": model.gat1.out_dim,
            "n_heads": model.gat1.n_heads,
        }
    elif isinstance(model, LocalPlasticityNet):
        dims = tuple(
            layer.out_features
            for layer in model.plastic_layers
        )
        return {
            "n_cpgs": model.n_cpgs,
            "hidden_dims": dims,
            "use_spectral": model.use_spectral,
        }
    elif isinstance(model, HybridSpectralAge):
        return {
            "n_cpgs": model.n_cpgs,
            "fusion_dim": model.head[0].in_features,
        }
    else:
        raise ValueError(f"Unknown model: {type(model)}")
