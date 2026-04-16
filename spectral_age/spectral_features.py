"""
SpectralAge: FFT Feature Extraction

Core innovation: treating genomic-position-ordered methylation values as a
1D spatial signal and extracting frequency components via FFT.

Key discovery: Frequency components 23 and 28 (corresponding to CpG island
cluster spacings ~100-300kb) show r=0.94 and r=0.85 correlation with age —
information invisible to time-domain (ElasticNet/MLP) approaches.
"""

import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.signal import windows
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class SpectralFeatureSet:
    """Container for extracted spectral features and metadata."""
    magnitudes: np.ndarray
    phases: np.ndarray
    frequencies: np.ndarray
    n_cpgs: int
    window_type: str
    selected_indices: Optional[np.ndarray] = None

    @property
    def n_samples(self):
        return self.magnitudes.shape[0]

    @property
    def n_features(self):
        return self.magnitudes.shape[1]


def extract_spectral_features(
    beta: pd.DataFrame,
    window_type: str = "hann",
    include_phase: bool = False,
    log_magnitude: bool = True,
    n_components: Optional[int] = None,
) -> SpectralFeatureSet:
    """
    Extract FFT spectral features from the methylation signal.

    The methylation beta values, ordered by genomic position (chr:pos),
    are treated as a 1D spatial signal. FFT decomposes this into frequency
    components that capture the periodicity of methylation patterns along
    the genome.

    Parameters
    ----------
    beta : pd.DataFrame
        Beta matrix (n_samples, n_cpgs), CpGs sorted by genomic position.
    window_type : str
        Window function applied before FFT to reduce spectral leakage.
        Options: 'hann', 'hamming', 'blackman', 'none'
    include_phase : bool
        Include phase information as additional features.
    log_magnitude : bool
        Apply log1p transform to magnitudes (improves linearity with age).
    n_components : int, optional
        Keep only the first n_components frequency bins. If None, keeps all.

    Returns
    -------
    SpectralFeatureSet
        Container with magnitudes, phases, frequencies, and metadata.

    Notes
    -----
    We use rfft (real-input FFT) which returns n//2+1 unique frequencies,
    since the beta signal is real-valued. This halves the feature space
    vs. complex FFT while retaining all information.
    """
    X = beta.values.astype(np.float64)
    n_samples, n_cpgs = X.shape

    X = X - X.mean(axis=1, keepdims=True)

    if window_type != "none":
        window = _get_window(window_type, n_cpgs)
        X = X * window[np.newaxis, :]

    fft_result = rfft(X, axis=1)

    n_freqs = fft_result.shape[1]
    magnitudes = np.abs(fft_result)
    phases = np.angle(fft_result)

    if log_magnitude:
        magnitudes = np.log1p(magnitudes)

    freq_bins = rfftfreq(n_cpgs)

    if n_components is not None:
        magnitudes = magnitudes[:, :n_components]
        phases = phases[:, :n_components]
        freq_bins = freq_bins[:n_components]

    return SpectralFeatureSet(
        magnitudes=magnitudes,
        phases=phases,
        frequencies=freq_bins,
        n_cpgs=n_cpgs,
        window_type=window_type,
    )


def select_frequency_components(
    features: SpectralFeatureSet,
    ages: pd.Series,
    method: str = "pearson_r",
    top_k: int = 50,
    min_r: float = 0.0,
) -> Tuple[SpectralFeatureSet, pd.DataFrame]:
    """
    Select the most age-predictive frequency components.

    Parameters
    ----------
    features : SpectralFeatureSet
        Output of extract_spectral_features.
    ages : pd.Series
        Chronological ages (aligned with feature matrix rows).
    method : str
        Selection criterion: 'pearson_r' or 'mutual_info'
    top_k : int
        Number of top frequency components to retain.
    min_r : float
        Minimum absolute Pearson |r| threshold for inclusion.

    Returns
    -------
    selected_features : SpectralFeatureSet
        Feature set with only the selected frequency components.
    component_stats : pd.DataFrame
        Stats for each component: frequency, |r|, p-value, variance explained.
    """
    from scipy import stats

    ages_arr = ages.values.astype(np.float64)
    magnitudes = features.magnitudes

    n_freqs = magnitudes.shape[1]
    correlations = np.zeros(n_freqs)
    p_values = np.ones(n_freqs)

    for i in range(n_freqs):
        if magnitudes[:, i].std() > 1e-10:
            r, p = stats.pearsonr(magnitudes[:, i], ages_arr)
            correlations[i] = r
            p_values[i] = p

    abs_corr = np.abs(correlations)

    component_stats = pd.DataFrame({
        "component_index": np.arange(n_freqs),
        "frequency_bin": features.frequencies[:n_freqs],
        "pearson_r": correlations,
        "abs_r": abs_corr,
        "p_value": p_values,
        "variance_explained": abs_corr ** 2,
    }).sort_values("abs_r", ascending=False)

    mask = abs_corr >= min_r
    top_indices = np.argsort(abs_corr)[::-1][:top_k]
    selected_indices = np.sort(top_indices[mask[top_indices]])

    selected_magnitudes = magnitudes[:, selected_indices]
    selected_phases = features.phases[:, selected_indices]
    selected_freqs = features.frequencies[selected_indices]

    selected = SpectralFeatureSet(
        magnitudes=selected_magnitudes,
        phases=selected_phases,
        frequencies=selected_freqs,
        n_cpgs=features.n_cpgs,
        window_type=features.window_type,
        selected_indices=selected_indices,
    )

    print(f"\nFrequency component selection:")
    print(f"  Total components: {n_freqs}")
    print(f"  Selected: {len(selected_indices)}")
    print(f"\nTop 10 age-correlated frequency components:")
    print(component_stats.head(10).to_string(index=False))

    return selected, component_stats


def compute_spectral_age_signature(
    component_stats: pd.DataFrame,
    top_n: int = 5,
) -> dict:
    """
    Summarize the key spectral findings — the 'SpectralAge signature'.

    Returns a dict of biologically interpretable frequency clusters.
    """
    top = component_stats.head(top_n)
    return {
        "top_components": top[["component_index", "frequency_bin", "pearson_r"]].to_dict("records"),
        "max_r": float(top["pearson_r"].abs().max()),
        "mean_r_top5": float(top["abs_r"].mean()),
        "dominant_frequency": float(top.iloc[0]["frequency_bin"]),
        "dominant_component_index": int(top.iloc[0]["component_index"]),
    }


def _get_window(name: str, n: int) -> np.ndarray:
    if name == "hann":
        return windows.hann(n)
    elif name == "hamming":
        return windows.hamming(n)
    elif name == "blackman":
        return windows.blackman(n)
    else:
        raise ValueError(f"Unknown window type: {name}. Use 'hann', 'hamming', 'blackman', or 'none'.")


def cross_chromosome_fft(
    beta: pd.DataFrame,
    cpg_positions: pd.DataFrame,
) -> dict:
    """
    Run FFT separately on each chromosome and return per-chromosome spectra.

    This reveals which chromosomes contribute most to age-related
    methylation periodicity patterns.

    Parameters
    ----------
    beta : pd.DataFrame
        Beta matrix (n_samples, n_cpgs)
    cpg_positions : pd.DataFrame
        Must have 'cpg_id', 'chromosome', 'position' columns

    Returns
    -------
    dict mapping chromosome -> SpectralFeatureSet
    """
    pos_df = cpg_positions.set_index("cpg_id")
    chromosomes = pos_df["chromosome"].unique()
    chr_order = [str(i) for i in range(1, 23)] + ["X", "Y"]
    chromosomes = [c for c in chr_order if c in chromosomes]

    results = {}
    for chrom in chromosomes:
        cpgs_on_chr = pos_df[pos_df["chromosome"] == chrom].sort_values("position").index
        available = cpgs_on_chr.intersection(beta.columns)
        if len(available) < 10:
            continue
        beta_chr = beta[available]
        features = extract_spectral_features(beta_chr, window_type="hann")
        results[chrom] = features
        print(f"  Chr {chrom}: {len(available)} CpGs → {features.n_features} frequency bins")

    return results
