import sys, time, json, numpy as np, pandas as pd, torch, pickle
sys.path.insert(0, ".")
np.random.seed(42); torch.manual_seed(42)
from spectral_age import *

beta = pd.read_pickle("data/beta_2k.pkl")
ages = pd.read_pickle("data/ages.pkl")
beta, _ = preprocess_beta_matrix(beta, sort_by_position=False)
beta_500 = beta.iloc[:, :500]
n_cpgs = 500
edge_index = build_sequential_adjacency(n_cpgs, k=5)
features = extract_spectral_features(beta_500, window_type="hann", log_magnitude=True)

results = []
EP = 60
BS = 64
LR = 3e-3
CV = 5

print("=" * 60)
print("  1) ElasticNet Baseline")
print("=" * 60)
t0 = time.time()
bl = ElasticNetBaseline(cv_folds=CV)
r = bl.fit_and_evaluate(beta_500, ages)
results.append(r)
print(f"  MAE={r.mae:.2f}y  R={r.pearson_r:.4f}  [{time.time()-t0:.0f}s]")

print("\n" + "=" * 60)
print("  2) SpectralAgeNet (FFT+Attention+MLP)")
print("=" * 60)
t0 = time.time()
m = SpectralAgeNet(n_cpgs=n_cpgs, hidden_dims=(128, 64), dropout=0.2)
r, _ = train_spectral_model(m, beta_500, ages, epochs=EP, lr=LR, batch_size=BS, cv_folds=CV)
r.model_name = "SpectralAgeNet (FFT+Attention)"
results.append(r)
print(f"  MAE={r.mae:.2f}y  R={r.pearson_r:.4f}  [{time.time()-t0:.0f}s]")

print("\n" + "=" * 60)
print("  3) SpectralForest (GBM on FFT)")
print("=" * 60)
t0 = time.time()
sf = SpectralForest(method="gbm", n_estimators=200, max_depth=3, cv_folds=CV)
r = sf.fit_and_evaluate(features, ages)
results.append(r)
print(f"  MAE={r.mae:.2f}y  R={r.pearson_r:.4f}  [{time.time()-t0:.0f}s]")

print("\n" + "=" * 60)
print("  4) LocalPlasticityNet (Hebbian)")
print("=" * 60)
t0 = time.time()
m = LocalPlasticityNet(n_cpgs=n_cpgs, hidden_dims=(64, 32), use_spectral=True, dropout=0.2)
r, _ = train_graph_model(m, beta_500, ages, edge_index, epochs=EP, lr=LR, batch_size=BS, cv_folds=CV, verbose=True)
r.model_name = "LocalPlasticityNet (Hebbian)"
results.append(r)
print(f"  MAE={r.mae:.2f}y  R={r.pearson_r:.4f}  [{time.time()-t0:.0f}s]")

print("\n" + "=" * 70)
print("  FINAL COMPARISON — GSE40279 (656 samples, 500 CpGs)")
print("=" * 70)
df = compare_models(results)
print(df.to_string(index=False))

from pathlib import Path
save_dir = Path("results"); save_dir.mkdir(exist_ok=True)
df.to_csv(save_dir / "model_comparison.csv", index=False)

best = min(results, key=lambda r: r.mae)
summary = {
    "dataset": "GSE40279",
    "n_samples": 656,
    "n_cpgs": n_cpgs,
    "epochs": EP,
    "best_model": best.model_name,
    "best_mae": round(float(best.mae), 3),
    "models": {r.model_name: {"mae": round(float(r.mae), 3), "rmse": round(float(r.rmse), 3), "pearson_r": round(float(r.pearson_r), 4), "r2": round(float(r.r2), 4)} for r in results},
}
with open(save_dir / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

_, comp_stats = select_frequency_components(features, ages, top_k=50)
comp_stats.to_csv(save_dir / "frequency_correlations.csv", index=False)

print(f"\nTop 5 age-frequency correlations:")
print(comp_stats.head(5)[["component_index","pearson_r","abs_r"]].to_string(index=False))
print(f"\nBest: {best.model_name} (MAE={best.mae:.2f}y)")
