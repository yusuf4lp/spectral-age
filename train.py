"""
SpectralAge — Ana Egitim ve Degerlendirme Scripti

Kullanim:
  # FFT modeli (varsayilan)
  python train.py --gse GSE40279 --data_dir ./data --epochs 200

  # Tum modeller karsilastirmali
  python train.py --gse GSE40279 --model all --epochs 200

  # Sadece hybrid (FFT + Graph + Plasticity)
  python train.py --gse GSE40279 --model hybrid --epochs 200

  # Plasticity (world model tarzi)
  python train.py --gse GSE40279 --model plasticity --epochs 200

  # Graph
  python train.py --gse GSE40279 --model graph --manifest data/manifest.csv

  # Kendi verinizle
  python train.py --beta data/beta.csv --pheno data/pheno.csv --model all

Modeller:
  full       — SpectralAgeNet (FFT + frequency attention + MLP)
  linear     — SpectralAgeLinear (FFT + single linear)
  forest     — SpectralForest (FFT features -> GBM ensemble)
  graph      — SparseGraphNet (genomik yakinlik grafi + sparse attention)
  plasticity — LocalPlasticityNet (Hebbian plasticity, world model tarzi)
  hybrid     — HybridSpectralAge (FFT + Graph + Plasticity gated fusion)
  all        — Tum modeller karsilastirmali
"""

import argparse
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from spectral_age import (
    SpectralAgeNet,
    SpectralAgeLinear,
    ElasticNetBaseline,
    train_spectral_model,
    compare_models,
    extract_spectral_features,
    select_frequency_components,
    preprocess_beta_matrix,
    download_geo_matrix,
    parse_geo_matrix,
    load_450k_manifest,
    load_local_csv,
    plot_spectral_landscape,
    plot_model_comparison,
    plot_frequency_weights,
    SpectralForest,
    SparseGraphNet,
    LocalPlasticityNet,
    HybridSpectralAge,
    build_adjacency_from_positions,
    build_sequential_adjacency,
    train_graph_model,
)

MODEL_CHOICES = ["full", "linear", "forest", "graph", "plasticity", "hybrid", "all"]


def parse_args():
    p = argparse.ArgumentParser(description="SpectralAge — PyTorch Epigenetik Saat")

    data = p.add_mutually_exclusive_group(required=True)
    data.add_argument("--gse", type=str, choices=["GSE40279", "GSE87571", "GSE55763"])
    data.add_argument("--beta", type=str)

    p.add_argument("--pheno", type=str, default=None)
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--manifest", type=str, default=None)
    p.add_argument("--max_cpgs", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None)

    p.add_argument("--model", type=str, default="full", choices=MODEL_CHOICES)
    p.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128, 64])
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--no_baseline", action="store_true")

    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--cv_folds", type=int, default=5)

    p.add_argument("--top_k_freqs", type=int, default=None)
    p.add_argument("--window", type=str, default="hann",
                   choices=["hann", "hamming", "blackman", "none"])

    p.add_argument("--k_neighbors", type=int, default=5,
                   help="Graph: her CpG icin komsu sayisi")
    p.add_argument("--max_distance_bp", type=int, default=500_000,
                   help="Graph: maksimum kenar uzunlugu (bp)")

    p.add_argument("--save_dir", type=str, default="./results")
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  SpectralAge v0.2 — PyTorch FFT + Graph + Plasticity")
    print("=" * 70)

    # ── 1. Veri yukleme ───────────────────────────────────────────────────
    if args.gse:
        matrix_file = download_geo_matrix(args.gse, data_dir=args.data_dir)
        beta, ages = parse_geo_matrix(
            str(matrix_file),
            max_cpgs=args.max_cpgs,
            max_samples=args.max_samples,
        )
    else:
        if args.pheno is None:
            print("HATA: --beta ile --pheno gerekli.")
            sys.exit(1)
        beta, ages = load_local_csv(args.beta, args.pheno)
        if args.max_cpgs:
            beta = beta.iloc[:, :args.max_cpgs]
        if args.max_samples:
            beta = beta.iloc[:args.max_samples]
            ages = ages.iloc[:args.max_samples]

    # ── 2. Manifest ──────────────────────────────────────────────────────
    cpg_positions = None
    if args.manifest:
        cpg_positions = load_450k_manifest(args.manifest)

    # ── 3. On isleme ─────────────────────────────────────────────────────
    beta, cpg_order = preprocess_beta_matrix(
        beta,
        cpg_positions=cpg_positions,
        sort_by_position=cpg_positions is not None,
    )

    n_cpgs = beta.shape[1]
    print(f"\nFinal: {beta.shape[0]} ornek, {n_cpgs} CpG")
    print(f"Yas: {ages.min():.1f} – {ages.max():.1f}  (ort={ages.mean():.1f})")

    # ── 4. Graph olusturma (gerekirse) ────────────────────────────────────
    need_graph = args.model in ("graph", "hybrid", "all")
    edge_index = None
    if need_graph:
        if cpg_positions is not None:
            edge_index = build_adjacency_from_positions(
                cpg_positions,
                list(cpg_order) if cpg_order is not None else list(beta.columns),
                k_neighbors=args.k_neighbors,
                max_distance_bp=args.max_distance_bp,
            )
        else:
            print("  Manifest yok — sirasal komsuluk grafi olusturuluyor")
            edge_index = build_sequential_adjacency(n_cpgs, k=args.k_neighbors)

    # ── 5. Spektral ozellikler ────────────────────────────────────────────
    features = extract_spectral_features(beta, window_type=args.window, log_magnitude=True)
    component_stats_df = None
    if args.top_k_freqs:
        features, component_stats_df = select_frequency_components(
            features, ages=ages, top_k=args.top_k_freqs
        )

    # ── 6. Spektral peyzaj grafigi ────────────────────────────────────────
    if not args.no_plots:
        plot_spectral_landscape(
            features.magnitudes, ages.values, features.frequencies,
            save_path=str(save_dir / "spectral_landscape.png"),
        )

    # ── 7. Model egitimi ─────────────────────────────────────────────────
    all_results = []
    models_to_run = (
        ["full", "forest", "graph", "plasticity", "hybrid"]
        if args.model == "all"
        else [args.model]
    )

    fitted_models = {}

    for model_name in models_to_run:
        print(f"\n{'─'*60}")
        print(f"  Model: {model_name.upper()}")
        print(f"{'─'*60}")

        t0 = time.time()

        if model_name == "full":
            model = SpectralAgeNet(
                n_cpgs=n_cpgs,
                hidden_dims=tuple(args.hidden_dims),
                dropout=args.dropout,
            )
            result, fitted = train_spectral_model(
                model, beta, ages,
                epochs=args.epochs, lr=args.lr,
                batch_size=args.batch_size,
                weight_decay=args.weight_decay,
                cv_folds=args.cv_folds,
            )
            result.model_name = "SpectralAgeNet (FFT+Attention+MLP)"
            fitted_models["full"] = fitted

        elif model_name == "linear":
            model = SpectralAgeLinear(n_cpgs=n_cpgs)
            result, fitted = train_spectral_model(
                model, beta, ages,
                epochs=args.epochs, lr=args.lr,
                batch_size=args.batch_size,
                cv_folds=args.cv_folds,
            )
            result.model_name = "SpectralAgeLinear (FFT+Linear)"
            fitted_models["linear"] = fitted

        elif model_name == "forest":
            sf = SpectralForest(
                method="gbm",
                n_estimators=500,
                max_depth=5,
                cv_folds=args.cv_folds,
            )
            result = sf.fit_and_evaluate(features, ages)
            fitted_models["forest"] = sf

        elif model_name == "graph":
            ei = edge_index if edge_index is not None else build_sequential_adjacency(n_cpgs, k=5)
            model = SparseGraphNet(
                n_cpgs=n_cpgs,
                embed_dim=64,
                gat_dim=64,
                n_heads=4,
            )
            result, fitted = train_graph_model(
                model, beta, ages, ei,
                epochs=args.epochs, lr=args.lr,
                batch_size=args.batch_size,
                cv_folds=args.cv_folds,
            )
            result.model_name = "SparseGraphNet (GAT)"
            fitted_models["graph"] = fitted

        elif model_name == "plasticity":
            model = LocalPlasticityNet(
                n_cpgs=n_cpgs,
                hidden_dims=tuple(args.hidden_dims),
                use_spectral=True,
                dropout=args.dropout,
            )
            ei = edge_index if edge_index is not None else build_sequential_adjacency(n_cpgs, k=5)
            result, fitted = train_graph_model(
                model, beta, ages, ei,
                epochs=args.epochs, lr=args.lr,
                batch_size=args.batch_size,
                cv_folds=args.cv_folds,
            )
            result.model_name = "LocalPlasticityNet (Hebbian)"
            fitted_models["plasticity"] = fitted

            stats = fitted.get_plasticity_stats()
            print("\n  Plasticity istatistikleri:")
            for layer_name, s in stats.items():
                print(f"    {layer_name}: alpha_mean={s['alpha_mean']:.4f}, "
                      f"eta={s['eta']:.4f}, "
                      f"plastic_connections={s['n_plastic_connections']}/{s['total_connections']}")

        elif model_name == "hybrid":
            ei = edge_index if edge_index is not None else build_sequential_adjacency(n_cpgs, k=5)
            model = HybridSpectralAge(
                n_cpgs=n_cpgs,
                fusion_dim=64,
                dropout=args.dropout,
            )
            result, fitted = train_graph_model(
                model, beta, ages, ei,
                epochs=args.epochs, lr=args.lr,
                batch_size=args.batch_size,
                cv_folds=args.cv_folds,
            )
            result.model_name = "HybridSpectralAge (FFT+Graph+Plasticity)"
            fitted_models["hybrid"] = fitted

            X_t = torch.from_numpy(beta.values.astype(np.float32))
            gates = fitted.get_gate_distribution(X_t, ei)
            print(f"\n  Gate dagilimi (ort): FFT={gates[:,0].mean():.3f}, "
                  f"Graph={gates[:,1].mean():.3f}, Plasticity={gates[:,2].mean():.3f}")

        elapsed = time.time() - t0
        print(f"\n  Egitim suresi: {elapsed/60:.1f} dk")
        all_results.append(result)

    # ── 8. Baseline ──────────────────────────────────────────────────────
    if not args.no_baseline:
        baseline = ElasticNetBaseline(cv_folds=args.cv_folds)
        bl_result = baseline.fit_and_evaluate(beta, ages)
        all_results.append(bl_result)

    # ── 9. Karsilastirma ─────────────────────────────────────────────────
    df = compare_models(all_results)
    df.to_csv(save_dir / "model_comparison.csv", index=False)

    # ── 10. Grafikler ────────────────────────────────────────────────────
    if not args.no_plots:
        plot_model_comparison(all_results, save_path=str(save_dir / "model_comparison.png"))

        if "full" in fitted_models and hasattr(fitted_models["full"], "spectral"):
            plot_frequency_weights(
                fitted_models["full"],
                component_stats=component_stats_df,
                save_path=str(save_dir / "frequency_weights.png"),
            )

    # ── 11. Sonuclari kaydet ─────────────────────────────────────────────
    summary = {
        "dataset": args.gse or args.beta,
        "n_samples": int(beta.shape[0]),
        "n_cpgs": int(beta.shape[1]),
        "models": {},
    }
    for r in all_results:
        summary["models"][r.model_name] = {
            "mae": round(float(r.mae), 3),
            "rmse": round(float(r.rmse), 3),
            "pearson_r": round(float(r.pearson_r), 4),
            "r2": round(float(r.r2), 4),
        }

    best = min(all_results, key=lambda r: r.mae)
    summary["best_model"] = best.model_name
    summary["best_mae"] = round(float(best.mae), 3)

    with open(save_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    for name, model in fitted_models.items():
        if isinstance(model, nn.Module):
            torch.save(model.state_dict(), save_dir / f"{name}_model.pt")

    print(f"\n{'='*70}")
    print("TAMAMLANDI")
    print(f"{'='*70}")
    print(f"  Cikti: {save_dir.resolve()}")
    print(f"  En iyi model: {best.model_name} (MAE={best.mae:.3f}y)")
    for r in all_results:
        print(f"    {r}")


if __name__ == "__main__":
    main()
