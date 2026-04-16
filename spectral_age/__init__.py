"""
SpectralAge — PyTorch FFT + Graph + Plasticity Epigenetik Saat

Katmanlı mimari:
  1. SpectralAgeNet      — FFT + learnable frequency attention + MLP
  2. SpectralForest       — FFT features → GBM/RF ensemble (fallback)
  3. SparseGraphNet       — CpG genomik yakınlık grafiği + sparse attention
  4. LocalPlasticityNet   — Hebbian plasticity (world model tarzı per-sample adaptasyon)
  5. HybridSpectralAge    — Tümünü birleştiren gated fusion

FFT yetersiz kaldığında Graph → Plasticity → Hybrid devreye girer.
"""

from .models import (
    SpectralAgeNet,
    SpectralAgeLinear,
    ElasticNetBaseline,
    train_spectral_model,
    compare_models,
    EvalResult,
    MethylationDataset,
)
from .graph_models import (
    SpectralForest,
    SparseGraphNet,
    LocalPlasticityNet,
    HybridSpectralAge,
    GatedFusion,
    SparseGraphAttention,
    HebbianPlasticLayer,
    build_adjacency_from_positions,
    build_sequential_adjacency,
    train_graph_model,
)
from .spectral_features import (
    extract_spectral_features,
    select_frequency_components,
    SpectralFeatureSet,
    cross_chromosome_fft,
)
from .preprocessing import (
    preprocess_beta_matrix,
    load_horvath_cpgs,
)
from .geo_loader import (
    download_geo_matrix,
    parse_geo_matrix,
    load_450k_manifest,
    load_local_csv,
)
from .visualization import (
    plot_spectral_landscape,
    plot_model_comparison,
    plot_frequency_weights,
    plot_intervention_effects,
)

__version__ = "0.2.0"
