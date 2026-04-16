"""
Pickle'dan veri yukleyip tum modelleri calistir.
GSE40279: 656 ornek, 10K CpG, yas 19-101
"""
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, ".")
np.random.seed(42)
torch.manual_seed(42)

from spectral_age import (
    SpectralAgeNet,
    SpectralAgeLinear,
    ElasticNetBaseline,
    train_spectral_model,
    compare_models,
    extract_spectral_features,
    select_frequency_components,
    preprocess_beta_matrix,
    SpectralForest,
    SparseGraphNet,
    LocalPlasticityNet,
    HybridSpectralAge,
    build_sequential_adjacency,
    train_graph_model,
)

save_dir = Path("results")
save_dir.mkdir(exist_ok=True)

print("=" * 70)
print("  SpectralAge v0.2 — GSE40279 Gercek Veri")
print("=" * 70)

beta = pd.read_pickle("data/beta_10k.pkl")
ages = pd.read_pickle("data/ages.pkl")
print(f"Yuklendi: {beta.shape[0]} ornek, {beta.shape[1]} CpG")
print(f"Yas: {ages.min():.0f}-{ages.max():.0f} (ort={ages.mean():.1f})")

beta, cpg_order = preprocess_beta_matrix(beta, sort_by_position=False)
n_cpgs = beta.shape[1]

edge_index = build_sequential_adjacency(n_cpgs, k=5)

features = extract_spectral_features(beta, window_type="hann", log_magnitude=True)
sel_features, comp_stats = select_frequency_components(features, ages, top_k=100)

EPOCHS = 100
CV = 5
LR = 1e-3
BS = 32
all_results = []

print(f"\n{'='*70}")
print("  1/5: SpectralAgeNet (FFT + Attention + MLP)")
print(f"{'='*70}")
t0 = time.time()
m1 = SpectralAgeNet(n_cpgs=n_cpgs, hidden_dims=(256, 128, 64), dropout=0.3)
r1, fitted_spectral = train_spectral_model(m1, beta, ages, epochs=EPOCHS, lr=LR, batch_size=BS, cv_folds=CV)
r1.model_name = "SpectralAgeNet (FFT+Attention+MLP)"
all_results.append(r1)
print(f"  Sure: {(time.time()-t0)/60:.1f} dk")

print(f"\n{'='*70}")
print("  2/5: SpectralForest (GBM on FFT features)")
print(f"{'='*70}")
t0 = time.time()
sf = SpectralForest(method="gbm", n_estimators=500, max_depth=5, cv_folds=CV)
r2 = sf.fit_and_evaluate(features, ages)
all_results.append(r2)
print(f"  Sure: {(time.time()-t0)/60:.1f} dk")

print(f"\n{'='*70}")
print("  3/5: LocalPlasticityNet (Hebbian)")
print(f"{'='*70}")
t0 = time.time()
m3 = LocalPlasticityNet(n_cpgs=n_cpgs, hidden_dims=(256, 128, 64), use_spectral=True, dropout=0.3)
r3, fitted_plastic = train_graph_model(m3, beta, ages, edge_index, epochs=EPOCHS, lr=LR, batch_size=BS, cv_folds=CV)
r3.model_name = "LocalPlasticityNet (Hebbian)"
all_results.append(r3)
stats = fitted_plastic.get_plasticity_stats()
for layer, s in stats.items():
    print(f"  {layer}: alpha={s['alpha_mean']:.4f}, eta={s['eta']:.4f}, plastic={s['n_plastic_connections']}/{s['total_connections']}")
print(f"  Sure: {(time.time()-t0)/60:.1f} dk")

print(f"\n{'='*70}")
print("  4/5: HybridSpectralAge (FFT+Graph+Plasticity)")
print(f"{'='*70}")
t0 = time.time()
m4 = HybridSpectralAge(n_cpgs=n_cpgs, fusion_dim=64, dropout=0.3)
r4, fitted_hybrid = train_graph_model(m4, beta, ages, edge_index, epochs=EPOCHS, lr=LR, batch_size=BS, cv_folds=CV)
r4.model_name = "HybridSpectralAge (FFT+Graph+Plasticity)"
all_results.append(r4)

X_t = torch.from_numpy(beta.values.astype(np.float32))
gates = fitted_hybrid.get_gate_distribution(X_t, edge_index)
print(f"  Gate: FFT={gates[:,0].mean():.3f}, Graph={gates[:,1].mean():.3f}, Plasticity={gates[:,2].mean():.3f}")
print(f"  Sure: {(time.time()-t0)/60:.1f} dk")

print(f"\n{'='*70}")
print("  5/5: ElasticNet Baseline (Horvath-style)")
print(f"{'='*70}")
t0 = time.time()
bl = ElasticNetBaseline(cv_folds=CV)
r_bl = bl.fit_and_evaluate(beta, ages)
all_results.append(r_bl)
print(f"  Sure: {(time.time()-t0)/60:.1f} dk")

print(f"\n{'='*70}")
print("  FINAL SONUCLAR")
print(f"{'='*70}")
df = compare_models(all_results)
df.to_csv(save_dir / "model_comparison.csv", index=False)

summary = {
    "dataset": "GSE40279",
    "n_samples": int(beta.shape[0]),
    "n_cpgs": int(beta.shape[1]),
    "models": {},
}
best = min(all_results, key=lambda r: r.mae)
for r in all_results:
    summary["models"][r.model_name] = {
        "mae": round(float(r.mae), 3),
        "rmse": round(float(r.rmse), 3),
        "pearson_r": round(float(r.pearson_r), 4),
        "r2": round(float(r.r2), 4),
    }
summary["best_model"] = best.model_name
summary["best_mae"] = round(float(best.mae), 3)

with open(save_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

torch.save(fitted_spectral.state_dict(), save_dir / "spectral_model.pt")
torch.save(fitted_hybrid.state_dict(), save_dir / "hybrid_model.pt")
torch.save(fitted_plastic.state_dict(), save_dir / "plasticity_model.pt")

print(f"\nCiktilar: {save_dir.resolve()}")
print(f"En iyi model: {best.model_name} (MAE={best.mae:.3f}y)")

comp_stats.to_csv(save_dir / "frequency_correlations.csv", index=False)
print(f"\nTop 10 yas-frekans korelasyonu:")
print(comp_stats.head(10)[["component_index","frequency_bin","pearson_r","abs_r","p_value"]].to_string(index=False))
