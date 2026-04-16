"""
Tum model mimarilerinin hizli dogrulama testi.
Kucuk sentetik batch ile forward pass + backward pass.
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from spectral_age import (
    SpectralAgeNet,
    SpectralAgeLinear,
    SpectralForest,
    SparseGraphNet,
    LocalPlasticityNet,
    HybridSpectralAge,
    HebbianPlasticLayer,
    extract_spectral_features,
    preprocess_beta_matrix,
    build_sequential_adjacency,
    train_spectral_model,
    train_graph_model,
    compare_models,
    ElasticNetBaseline,
)

N = 40
C = 200
np.random.seed(42)
torch.manual_seed(42)

beta_np = np.random.beta(2, 5, size=(N, C)).astype(np.float32)
beta = pd.DataFrame(beta_np, columns=[f"cg{i:08d}" for i in range(C)])
ages = pd.Series(np.linspace(20, 80, N) + np.random.normal(0, 3, N), name="age")
beta_proc, _ = preprocess_beta_matrix(beta, sort_by_position=False)

edge_index = build_sequential_adjacency(C, k=3)
x = torch.from_numpy(beta_np)

print("=" * 60)
print("  Tum Modeller — Forward + Backward Test")
print("=" * 60)


def test_model(name, model, needs_graph=False):
    model.eval()
    with torch.no_grad():
        if needs_graph:
            out = model(x, edge_index)
        else:
            out = model(x)
    assert out.shape == (N,), f"{name}: beklenmeyen output shape {out.shape}"

    model.train()
    if needs_graph:
        pred = model(x, edge_index)
    else:
        pred = model(x)
    loss = F.mse_loss(pred, torch.from_numpy(ages.values.astype(np.float32)))
    loss.backward()
    params = sum(p.numel() for p in model.parameters())
    print(f"  {name:<35} params={params:>8,}  loss={loss.item():.2f}  shape={out.shape}")


print("\n1. SpectralAgeNet (FFT + Attention + MLP)")
test_model("SpectralAgeNet", SpectralAgeNet(n_cpgs=C, hidden_dims=(64, 32)))

print("\n2. SpectralAgeLinear (FFT + Linear)")
test_model("SpectralAgeLinear", SpectralAgeLinear(n_cpgs=C))

print("\n3. SparseGraphNet (GAT)")
test_model("SparseGraphNet", SparseGraphNet(n_cpgs=C, embed_dim=32, gat_dim=32, n_heads=2), needs_graph=True)

print("\n4. LocalPlasticityNet (Hebbian)")
model_p = LocalPlasticityNet(n_cpgs=C, hidden_dims=(64, 32), use_spectral=True)
test_model("LocalPlasticityNet", model_p)

stats = model_p.get_plasticity_stats()
for layer, s in stats.items():
    print(f"     {layer}: alpha_mean={s['alpha_mean']:.4f}, eta={s['eta']:.4f}")

print("\n5. HybridSpectralAge (FFT + Graph + Plasticity)")
hybrid = HybridSpectralAge(n_cpgs=C, fusion_dim=32)
test_model("HybridSpectralAge", hybrid, needs_graph=True)

gates = hybrid.get_gate_distribution(x, edge_index)
print(f"     Gate (ort): FFT={gates[:,0].mean():.3f}, Graph={gates[:,1].mean():.3f}, Plasticity={gates[:,2].mean():.3f}")

print("\n6. HebbianPlasticLayer — unit test")
hpl = HebbianPlasticLayer(64, 32)
inp = torch.randn(4, 64)
out = hpl(inp)
assert out.shape == (4, 32)
print(f"     in=({4}, 64) -> out={out.shape}  alpha range={torch.sigmoid(hpl.alpha).min():.3f}-{torch.sigmoid(hpl.alpha).max():.3f}")

print("\n7. SpectralForest (GBM on FFT features)")
features = extract_spectral_features(beta_proc, window_type="hann")
sf = SpectralForest(method="gbm", n_estimators=50, max_depth=3, cv_folds=3)
sf_result = sf.fit_and_evaluate(features, ages)
print(f"     MAE={sf_result.mae:.2f}y  r={sf_result.pearson_r:.3f}")
top_f = sf.get_top_frequencies(5)
print(f"     Top frequencies: {top_f}")

print(f"\n{'='*60}")
print("  Mini egitim testi (5 epoch, 3-fold)")
print(f"{'='*60}")

results = []

m1 = SpectralAgeNet(n_cpgs=C, hidden_dims=(32,))
r1, _ = train_spectral_model(m1, beta_proc, ages, epochs=5, cv_folds=3, verbose=False)
r1.model_name = "SpectralAgeNet"
results.append(r1)

m2 = LocalPlasticityNet(n_cpgs=C, hidden_dims=(32,), use_spectral=True)
r2, _ = train_graph_model(m2, beta_proc, ages, edge_index, epochs=5, cv_folds=3, verbose=False)
r2.model_name = "LocalPlasticityNet"
results.append(r2)

m3 = HybridSpectralAge(n_cpgs=C, fusion_dim=32)
r3, _ = train_graph_model(m3, beta_proc, ages, edge_index, epochs=5, cv_folds=3, verbose=False)
r3.model_name = "HybridSpectralAge"
results.append(r3)

bl = ElasticNetBaseline(cv_folds=3)
r_bl = bl.fit_and_evaluate(beta_proc, ages)
results.append(r_bl)

results.append(sf_result)

compare_models(results)

print(f"\n{'='*60}")
print("  TUM TESTLER BASARILI")
print(f"{'='*60}")
