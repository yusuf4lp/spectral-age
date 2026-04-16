"""
SpectralAge — Görselleştirme Araçları

Grafikler:
  1. Spektral peyzaj  — tüm örneklerin FFT magnitude spektrumu (yas ile renklendirilmiş)
  2. Frekans korelasyonları — her frekans bileşeninin yaşla |r| değeri
  3. Model karşılaştırması — tahmin vs gerçek, residual dağılımı
  4. Frekans ağırlıkları — PyTorch modelinin öğrendiği ağırlıklar
  5. Müdahale etkileri — spektral domain'de epigenetik müdahalelerin etkisi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from typing import Optional, List
import torch
import torch.nn as nn


PALETTE = {
    "spectral": "#4A90D9",
    "baseline": "#E87B4F",
    "accent": "#6CC07A",
    "rapamycin": "#9B59B6",
    "reprogramming": "#E91E63",
    "cr": "#FF9800",
    "senolytics": "#2196F3",
    "grid": "#E8E8E8",
    "bg": "#FAFAFA",
}


def plot_spectral_landscape(
    magnitudes: np.ndarray,
    ages: np.ndarray,
    frequencies: np.ndarray,
    top_n: int = 5,
    annotate_top: bool = True,
    save_path: Optional[str] = None,
):
    """
    Tüm örneklerin FFT magnitude spektrumunu yaşa göre renklendirilmiş çizgiler olarak göster.
    Yaşla en güçlü korelasyon gösteren frekans bileşenlerini işaretle.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.patch.set_facecolor(PALETTE["bg"])

    # ── Panel 1: Spektral peyzaj ──────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(PALETTE["bg"])

    norm = mcolors.Normalize(vmin=ages.min(), vmax=ages.max())
    cmap = plt.cm.viridis

    sorted_idx = np.argsort(ages)
    for i in sorted_idx[::max(1, len(sorted_idx)//100)]:
        ax.plot(
            frequencies,
            magnitudes[i],
            color=cmap(norm(ages[i])),
            alpha=0.3,
            linewidth=0.7,
        )

    mean_mag = magnitudes.mean(axis=0)
    ax.plot(frequencies, mean_mag, color="white", linewidth=2.5, label="Ortalama spektrum", zorder=5)
    ax.plot(frequencies, mean_mag, color=PALETTE["spectral"], linewidth=1.5, zorder=6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Kronolojik Yaş (yıl)", fontsize=11)

    if annotate_top:
        from scipy.stats import pearsonr
        corrs = np.array([abs(pearsonr(magnitudes[:, i], ages)[0]) for i in range(len(frequencies))])
        top_indices = np.argsort(corrs)[::-1][:top_n]
        for idx in top_indices:
            ax.axvline(
                frequencies[idx], color="#FFD700", alpha=0.7,
                linewidth=1.5, linestyle="--"
            )
            ax.annotate(
                f"f{idx}\nr={corrs[idx]:.2f}",
                xy=(frequencies[idx], mean_mag[idx]),
                xytext=(frequencies[idx] + 0.01, mean_mag.max() * 0.85),
                fontsize=8, color="#FFD700",
                arrowprops=dict(arrowstyle="->", color="#FFD700", lw=1),
            )

    ax.set_xlabel("Frekans Bini (1/CpG sayısı)", fontsize=12)
    ax.set_ylabel("log(1 + |FFT|)", fontsize=12)
    ax.set_title("SpectralAge — Metilasyon Sinyal Spektral Peyzajı", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, color=PALETTE["grid"], alpha=0.6)

    # ── Panel 2: Frekans-yaş korelasyonu ─────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(PALETTE["bg"])

    from scipy.stats import pearsonr as _pr
    corrs = np.array([_pr(magnitudes[:, i], ages)[0] for i in range(len(frequencies))])
    abs_corrs = np.abs(corrs)

    colors = [PALETTE["spectral"] if c > 0 else PALETTE["baseline"] for c in corrs]
    ax2.bar(range(len(frequencies)), corrs, color=colors, alpha=0.75, width=0.8)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axhline(0.5, color=PALETTE["accent"], linewidth=1, linestyle="--", alpha=0.7, label="|r| = 0.5")
    ax2.axhline(-0.5, color=PALETTE["accent"], linewidth=1, linestyle="--", alpha=0.7)

    top5 = np.argsort(abs_corrs)[::-1][:5]
    for idx in top5:
        ax2.annotate(
            f"f{idx} (r={corrs[idx]:.2f})",
            xy=(idx, corrs[idx]),
            xytext=(idx + len(frequencies)*0.03, corrs[idx] + 0.05 * np.sign(corrs[idx])),
            fontsize=8, color="#333333",
            arrowprops=dict(arrowstyle="->", color="#555", lw=0.8),
        )

    ax2.set_xlabel("Frekans Bini İndeksi", fontsize=12)
    ax2.set_ylabel("Pearson r (frekans — yaş)", fontsize=12)
    ax2.set_title("Her Frekans Bileşeninin Yaşla Korelasyonu", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.set_ylim(-1, 1)
    ax2.grid(True, axis="y", color=PALETTE["grid"], alpha=0.6)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Kaydedildi: {save_path}")

    plt.show()
    return fig


def plot_model_comparison(
    results: list,
    save_path: Optional[str] = None,
):
    """
    Her model için:
      - Tahmin vs gerçek scatter plot
      - Residual dağılımı (histogram)
    """
    n_models = len(results)
    fig = plt.figure(figsize=(7 * n_models, 10))
    fig.patch.set_facecolor(PALETTE["bg"])
    gs = GridSpec(2, n_models, figure=fig, hspace=0.35, wspace=0.3)

    colors = [PALETTE["spectral"], PALETTE["baseline"], PALETTE["accent"]]

    for col, (result, color) in enumerate(zip(results, colors)):
        y_true = result.true_ages
        y_pred = result.predicted_ages
        residuals = y_pred - y_true

        # ── Scatter: tahmin vs gerçek ─────────────────────────────────────
        ax_scatter = fig.add_subplot(gs[0, col])
        ax_scatter.set_facecolor(PALETTE["bg"])

        ax_scatter.scatter(
            y_true, y_pred,
            alpha=0.5, s=25, color=color, edgecolors="white", linewidths=0.3,
        )

        lo = min(y_true.min(), y_pred.min()) - 2
        hi = max(y_true.max(), y_pred.max()) + 2
        ax_scatter.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, alpha=0.6, label="İdeal")

        ax_scatter.set_xlabel("Gerçek Yaş (yıl)", fontsize=11)
        ax_scatter.set_ylabel("Tahmin Edilen Yaş (yıl)", fontsize=11)
        ax_scatter.set_title(
            f"{result.model_name}\n"
            f"MAE={result.mae:.2f}y  r={result.pearson_r:.3f}",
            fontsize=11, fontweight="bold",
        )
        ax_scatter.legend(fontsize=9)
        ax_scatter.grid(True, color=PALETTE["grid"], alpha=0.5)

        # ── Residual histogram ────────────────────────────────────────────
        ax_res = fig.add_subplot(gs[1, col])
        ax_res.set_facecolor(PALETTE["bg"])

        ax_res.hist(
            residuals, bins=30, color=color, alpha=0.75, edgecolor="white", linewidth=0.5
        )
        ax_res.axvline(0, color="black", linewidth=1.5, linestyle="--")
        ax_res.axvline(residuals.mean(), color="#FFD700", linewidth=1.5, label=f"Ortalama={residuals.mean():.2f}")
        ax_res.set_xlabel("Residual (tahmin − gerçek) (yıl)", fontsize=11)
        ax_res.set_ylabel("Frekans", fontsize=11)
        ax_res.set_title("Residual Dağılımı", fontsize=11, fontweight="bold")
        ax_res.legend(fontsize=9)
        ax_res.grid(True, axis="y", color=PALETTE["grid"], alpha=0.5)

    fig.suptitle(
        "SpectralAge — Model Karşılaştırması",
        fontsize=15, fontweight="bold", y=1.01,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Kaydedildi: {save_path}")

    plt.show()
    return fig


def plot_frequency_weights(
    model: nn.Module,
    component_stats: Optional[pd.DataFrame] = None,
    save_path: Optional[str] = None,
):
    """
    PyTorch modelinin öğrendiği frekans attention ağırlıklarını görselleştir.
    Modelin hangi frekans bileşenlerine odaklandığını gösterir.
    """
    if not hasattr(model, "spectral"):
        print("Model bir SpectralLayer içermiyor.")
        return

    weights = model.get_frequency_importances() if hasattr(model, "get_frequency_importances") else \
        torch.softmax(model.spectral.freq_weights.detach().cpu(), dim=0).numpy() * model.spectral.n_freqs

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    x = np.arange(len(weights))
    ax.bar(x, weights, color=PALETTE["spectral"], alpha=0.8, width=0.8)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="Nötr ağırlık (1.0)")

    top5 = np.argsort(weights)[::-1][:5]
    for idx in top5:
        ax.annotate(
            f"f{idx}\n{weights[idx]:.2f}×",
            xy=(idx, weights[idx]),
            xytext=(idx + len(weights)*0.02, weights[idx] + 0.1),
            fontsize=8, color="#333",
            arrowprops=dict(arrowstyle="->", color="#555", lw=0.8),
        )

    if component_stats is not None and "pearson_r" in component_stats.columns:
        ax2 = ax.twinx()
        ax2.plot(
            component_stats["component_index"].values,
            component_stats["pearson_r"].abs().values,
            color=PALETTE["accent"], linewidth=1.5, alpha=0.7, label="|Pearson r|",
        )
        ax2.set_ylabel("|Pearson r| (yaşla)", color=PALETTE["accent"], fontsize=11)
        ax2.legend(loc="upper right", fontsize=9)

    ax.set_xlabel("Frekans Bini İndeksi", fontsize=12)
    ax.set_ylabel("Öğrenilen Ağırlık", fontsize=12)
    ax.set_title(
        "PyTorch SpectralAgeNet — Öğrenilmiş Frekans Attention Ağırlıkları",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", color=PALETTE["grid"], alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Kaydedildi: {save_path}")

    plt.show()
    return fig


def plot_intervention_effects(
    intervention_results: dict,
    save_path: Optional[str] = None,
):
    """
    Müdahalelerin spektral domain'deki etkisini görselleştir.
    """
    interventions = list(intervention_results.keys())
    delta_ages = [intervention_results[k]["delta_age_mean"] for k in interventions]
    std_ages = [intervention_results[k].get("delta_age_std", 0) for k in interventions]
    colors_map = {
        "Rapamycin": PALETTE["rapamycin"],
        "Partial reprogramming": PALETTE["reprogramming"],
        "Caloric restriction": PALETTE["cr"],
        "Senolytics": PALETTE["senolytics"],
    }
    bar_colors = [colors_map.get(k, PALETTE["spectral"]) for k in interventions]

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])

    bars = ax.barh(
        interventions, delta_ages,
        xerr=std_ages, color=bar_colors,
        alpha=0.85, capsize=5, edgecolor="white",
    )

    for bar, delta in zip(bars, delta_ages):
        ax.text(
            delta - 0.1 if delta < 0 else delta + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{delta:+.1f}y  ({delta/10*100:+.1f}%)",
            va="center", ha="right" if delta < 0 else "left",
            fontsize=10, fontweight="bold",
        )

    ax.axvline(0, color="black", linewidth=1.5)
    ax.set_xlabel("Tahmini Epigenetik Yaş Değişimi (yıl)", fontsize=12)
    ax.set_title(
        "Müdahale Etkileri — Spektral Domain Epigenetik Yaş Tahmini",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, axis="x", color=PALETTE["grid"], alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Kaydedildi: {save_path}")

    plt.show()
    return fig
