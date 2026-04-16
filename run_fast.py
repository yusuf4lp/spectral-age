import sys, time, json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, ".")
np.random.seed(42)
torch.manual_seed(42)

from spectral_age import (
    SpectralAgeNet, SpectralAgeLinear, ElasticNetBaseline,
    train_spectral_model, compare_models,
    extract_spectral_features, select_frequency_components,
    preprocess_beta_matrix,
    SpectralForest, SparseGraphNet, LocalPlasticityNet, HybridSpectralAge,
    build_sequential_adjacency, train_graph_model,
)

save_dir = Path("results"); save_dir.mkdir(exist_ok=True)

print("=" * 70)
print("  SpectralAge v0.2 — GSE40279 (656 samples, 2K CpGs)")
print("=" * 70)

beta = pd.read_pickle("data/beta_2k.pkl")
ages = pd.read_pickle("data/ages.pkl")
print(f"Yuklendi: {beta.shape[0]} ornek, {beta.shape[1]} CpG, Yas: {ages.min():.0f}-{ages.max():.0f}")

beta, cpg_order = preprocess_beta_matrix(beta, sort_by_position=False)
n_cpgs = beta.shape[1]
edge_index = build_sequential_adjacency(n_cpgs, k=5)

features = extract_spectral_features(beta, window_type="hann", log_magnitude=True)
sel_features, comp_stats = select_frequency_components(features, ages, top_k=100)

EPOCHS = 50
CV = 5
LR = 1e-3
BS = 64
all_results = []

def run_model(name, idx, total, fn):
    print(f"\n{'='*70}")
    print(f"  {idx}/{total}: {name}")
    print(f"{'='*70}")
    t0 = time.time()
    result = fn()
    elapsed = time.time() - t0
    print(f"  MAE={result.mae:.2f}y  R={result.pearson_r:.4f}  [{elapsed/60:.1f} dk]")
    all_results.append(result)
    return result

def fn_spectral():
    m = SpectralAgeNet(n_cpgs=n_cpgs, hidden_dims=(128, 64), dropout=0.3)
    r, _ = train_spectral_model(m, beta, ages, epochs=EPOCHS, lr=LR, batch_size=BS, cv_folds=CV)
    r.model_name = "SpectralAgeNet (FFT+Attention+MLP)"
    return r

def fn_forest():
    sf = SpectralForest(method="gbm", n_estimators=300, max_depth=4, cv_folds=CV)
    return sf.fit_and_evaluate(features, ages)

def fn_plasticity():
    m = LocalPlasticityNet(n_cpgs=n_cpgs, hidden_dims=(128, 64), use_spectral=True, dropout=0.3)
    r, _ = train_graph_model(m, beta, ages, edge_index, epochs=EPOCHS, lr=LR, batch_size=BS, cv_folds=CV)
    r.model_name = "LocalPlasticityNet (Hebbian)"
    return r

def fn_hybrid():
    m = HybridSpectralAge(n_cpgs=n_cpgs, fusion_dim=32, dropout=0.3)
    r, fitted = train_graph_model(m, beta, ages, edge_index, epochs=EPOCHS, lr=LR, batch_size=BS, cv_folds=CV)
    r.model_name = "HybridSpectralAge"
    X_t = torch.from_numpy(beta.values.astype(np.float32))
    gates = fitted.get_gate_distribution(X_t[:20], edge_index)
    print(f"  Gate: FFT={gates[:,0].mean():.3f}, Graph={gates[:,1].mean():.3f}, Plasticity={gates[:,2].mean():.3f}")
    return r

def fn_elasticnet():
    bl = ElasticNetBaseline(cv_folds=CV)
    return bl.fit_and_evaluate(beta, ages)

run_model("SpectralAgeNet", 1, 5, fn_spectral)
run_model("SpectralForest (GBM)", 2, 5, fn_forest)
run_model("LocalPlasticityNet", 3, 5, fn_plasticity)
run_model("HybridSpectralAge", 4, 5, fn_hybrid)
run_model("ElasticNet Baseline", 5, 5, fn_elasticnet)

print(f"\n{'='*70}")
print("  FINAL SONUCLAR — GSE40279")
print(f"{'='*70}")
df = compare_models(all_results)
df.to_csv(save_dir / "model_comparison.csv", index=False)
print(df.to_string(index=False))

best = min(all_results, key=lambda r: r.mae)
summary = {
    "dataset": "GSE40279",
    "n_samples": int(beta.shape[0]),
    "n_cpgs": int(beta.shape[1]),
    "epochs": EPOCHS,
    "cv_folds": CV,
    "best_model": best.model_name,
    "best_mae": round(float(best.mae), 3),
    "models": {r.model_name: {"mae": round(float(r.mae), 3), "rmse": round(float(r.rmse), 3), "pearson_r": round(float(r.pearson_r), 4), "r2": round(float(r.r2), 4)} for r in all_results},
}
with open(save_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

comp_stats.to_csv(save_dir / "frequency_correlations.csv", index=False)
print(f"\nTop 10 frekans-yas korelasyonu:")
print(comp_stats.head(10)[["component_index","frequency_bin","pearson_r","abs_r"]].to_string(index=False))
print(f"\nEn iyi: {best.model_name} (MAE={best.mae:.2f}y)")
