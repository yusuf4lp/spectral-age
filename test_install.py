"""
Kurulum ve import testi — veri indirmeden hızlıca çalışır.
Gerçek boyutlu sentetik bir batch ile forward pass doğrular.
"""
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")

print("=" * 60)
print("SpectralAge — Kurulum Testi")
print("=" * 60)

print(f"\nPython  : {sys.version.split()[0]}")
print(f"NumPy   : {np.__version__}")
print(f"Pandas  : {pd.__version__}")
print(f"PyTorch : {torch.__version__}")

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Device  : {device}")

print("\n-- Import testi --")
from spectral_age import (
    SpectralAgeNet,
    SpectralAgeLinear,
    ElasticNetBaseline,
    extract_spectral_features,
    select_frequency_components,
    preprocess_beta_matrix,
    plot_spectral_landscape,
)
print("  OK: tüm modüller yüklendi")

print("\n-- FFT forward pass --")
N_SAMPLES = 20
N_CPGS = 353  # Horvath CpG sayısı

beta_np = np.random.beta(2, 5, size=(N_SAMPLES, N_CPGS)).astype(np.float32)
beta = pd.DataFrame(beta_np, columns=[f"cg{i:08d}" for i in range(N_CPGS)])
ages = pd.Series(np.random.uniform(20, 80, N_SAMPLES), name="age")

beta_proc, _ = preprocess_beta_matrix(beta, sort_by_position=False)
features = extract_spectral_features(beta_proc, window_type="hann", log_magnitude=True)

print(f"  Input  : {beta.shape} (ornek x CpG)")
print(f"  Output : {features.magnitudes.shape} (ornek x frekans bini)")
print(f"  Frekans sayisi: {features.n_features} (n_cpgs//2 + 1 = {N_CPGS//2+1})")

print("\n-- PyTorch model forward pass --")
model = SpectralAgeNet(n_cpgs=N_CPGS, hidden_dims=(64, 32), dropout=0.1)
model.eval()

x = torch.from_numpy(beta_np)
with torch.no_grad():
    pred = model(x)

print(f"  Input shape  : {x.shape}")
print(f"  Output shape : {pred.shape}")
print(f"  Tahmin aralik: {pred.min().item():.2f} – {pred.max().item():.2f} yil")
print(f"  Parametre sayisi: {sum(p.numel() for p in model.parameters()):,}")

print("\n-- SpectralLayer ağırlık kontrolü --")
weights = model.get_frequency_importances()
top3 = np.argsort(weights)[::-1][:3]
print(f"  En yüksek ağırlıklı frekans binleri: {top3.tolist()}")
print(f"  Ağırlık aralığı: {weights.min():.4f} – {weights.max():.4f}")

print("\n-- Spektral özellik seçimi --")
sel_features, stats = select_frequency_components(features, ages, top_k=20)
print(f"  Seçilen frekans sayisi: {sel_features.n_features}")
print(f"  En iyi r: {stats['abs_r'].max():.4f}")

print("\n" + "=" * 60)
print("TÜM TESTLER BAŞARILI")
print("=" * 60)
print("\nGerçek veri ile çalıştırmak için:")
print("  python train.py --gse GSE40279 --data_dir ./data --epochs 200")
