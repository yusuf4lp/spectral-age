# SpectralAge

**A frequency-domain neural network for epigenetic age prediction from DNA methylation profiles.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Abstract

Existing epigenetic clocks (Horvath 2013, Hannum 2013, PhenoAge, GrimAge) operate exclusively in the *time domain* — they treat each CpG as an independent feature and fit a regularized linear model on raw β-values. We hypothesize that this approach discards a substantial portion of the information content in genome-wide methylation profiles: namely, the **spatial periodicity** of methylation along the chromosome. SpectralAge tests this hypothesis by sorting CpGs by genomic coordinate, treating each sample as a 1-D signal, and applying a learnable spectral transformation (`torch.fft.rfft`) before regression. We provide a cascading model family — pure FFT, gradient-boosted FFT, sparse graph attention, and Hebbian local plasticity — that allows direct comparison of frequency-domain and time-domain representations on the same datasets.

## Motivation

DNA methylation accumulates non-random changes with age. The dominant paradigm (Horvath-style ElasticNet) achieves median absolute error (MAE) of ~3.6 years across tissues using ~353 hand-selected CpGs. However:

1. The model is **agnostic to genomic position** — shuffling the CpG order has no effect on predictions.
2. **Spatial autocorrelation** of methylation (within CpG islands, gene bodies, replication timing domains) is empirically substantial but unexploited.
3. Recent work on aging clocks (DunedinPACE, GrimAge2) confirms diminishing returns from purely linear time-domain models.

If methylation drifts in a spatially periodic fashion — e.g., partially methylated domains expanding with age, replication timing reorganization — those signals would appear as **non-trivial frequency components** invisible to time-domain regression.

## Method

### 1. Spectral feature extraction

Given a β-value matrix `X ∈ ℝ^(n_samples × n_CpGs)` with CpGs sorted by `(chromosome, position)`:

```
x ← x − mean(x)            # per-sample centering
F ← rfft(x, dim=−1)        # real-valued FFT along CpG axis
m ← log(1 + |F|)           # log-magnitude spectrum
```

### 2. Model family (cascading complexity)

| Model | Description | Parameters |
|-------|-------------|------------|
| `SpectralAgeLinear`   | FFT → single linear layer (interpretable) | O(n_CpGs/2) |
| `SpectralAgeNet`      | FFT → learnable frequency attention → MLP | ~50K–500K |
| `SpectralForest`      | FFT magnitudes → GBM / Random Forest      | non-parametric |
| `SparseGraphNet`      | Genomic-proximity sparse graph attention  | ~100K |
| `LocalPlasticityNet`  | Slow weights `W` + fast Hebbian trace `α⊙A` (Miconi et al. 2018) | ~300K |
| `HybridSpectralAge`   | FFT + Graph + Plasticity streams → gated fusion | ~500K |

`FrequencyAttention` learns a softmax-normalized weight per frequency bin, allowing the network to suppress noise frequencies and emphasize age-informative bands.

### 3. Baseline

`ElasticNetBaseline` reproduces the Horvath protocol: 5-fold cross-validated ElasticNet on raw β-values with α=0.1, l1_ratio=0.5.

### 4. Training

- **Loss:** MSE on z-score-normalized age (denormalized at inference)
- **Optimizer:** AdamW (lr=3e-3, weight_decay=1e-4)
- **Schedule:** Cosine annealing (η_min = lr × 0.01)
- **Regularization:** Dropout 0.2–0.3, gradient clipping at norm 5.0
- **Validation:** 5-fold cross-validation with out-of-fold predictions

## Datasets

The pipeline auto-downloads three Illumina 450K whole-blood cohorts from GEO:

| GEO accession | First author | n | Age range | Tissue |
|---------------|--------------|---|-----------|--------|
| GSE40279 | Hannum (2013)   | 656  | 19–101 | Whole blood |
| GSE87571 | Johansson (2013)| 729  | 14–94  | Whole blood |
| GSE55763 | Lehne (2015)    | 2711 | 24–75  | Whole blood |

## Installation

```bash
git clone https://github.com/yusuf4lp/spectral-age.git
cd spectral-age
pip install torch numpy scipy pandas scikit-learn matplotlib requests
```

Verify installation:

```bash
python test_install.py
python test_all_models.py
```

## Usage

### Auto-download a GEO dataset and benchmark all models

```bash
python train.py --gse GSE40279 --model all --epochs 100
```

### Train a single model

```bash
python train.py --gse GSE40279 --model hybrid --epochs 200
```

`--model` options: `full | linear | forest | graph | plasticity | hybrid | all`

### Bring your own data

```bash
python train.py --beta my_beta.csv --pheno my_ages.csv --model all
```

- `my_beta.csv` — rows = samples, columns = CpG probes (cg00000029, …), values ∈ [0, 1]
- `my_ages.csv` — single column `age`, indexed by sample ID

### Quick benchmark (subsampled CpGs for fast iteration)

```bash
python run_comparison.py
```

## Preliminary results

GSE40279, n = 656, 500 CpGs (random subsample for CPU benchmark), 60 epochs, 5-fold CV:

| Model | MAE (years) | RMSE (years) | Pearson r | R² |
|-------|-------------|--------------|-----------|-----|
| ElasticNet baseline (Horvath-style) | **8.10** | 10.49 | **0.733** | 0.49 |
| SpectralForest (GBM on FFT)         | 8.97 | 11.48 | 0.630 | 0.39 |
| SpectralAgeNet (FFT + Attention)    | 11.53 | 14.28 | 0.347 | 0.06 |
| LocalPlasticityNet (Hebbian)        | 11.93 | 14.74 | -0.03 | 0.00 |

> **Caveat.** This benchmark uses a 500-CpG random subset on CPU for reproducibility on commodity hardware. Horvath's original 3.6-year MAE was obtained with 353 carefully *selected* CpGs from the full 27K array. Full-array training (473K CpGs, all 656 samples, 200+ epochs on GPU) is required for a fair comparison and is left as a follow-up.

### Frequency-age associations (GSE40279, top 4 components)

| Frequency bin | Pearson r | p-value | Variance explained |
|---------------|-----------|---------|--------------------|
| 250 (Nyquist) | +0.350 | 2.5×10⁻²⁰ | 12.2% |
| 60            | -0.323 | 1.9×10⁻¹⁷ | 10.5% |
| 27            | -0.310 | 4.2×10⁻¹⁶ | 9.6% |
| 28            | -0.304 | 1.8×10⁻¹⁵ | 9.2% |

Multiple low-frequency components show highly significant associations with chronological age (Bonferroni-corrected p < 0.001), consistent with the spatial-periodicity hypothesis.

## Project layout

```
spectral-age/
├── spectral_age/
│   ├── models.py              # SpectralAgeNet, SpectralAgeLinear, ElasticNetBaseline
│   ├── graph_models.py        # SpectralForest, SparseGraphNet, LocalPlasticityNet, HybridSpectralAge
│   ├── spectral_features.py   # FFT extraction, frequency-component selection
│   ├── geo_loader.py          # GEO series-matrix download + parsing
│   ├── preprocessing.py       # β-matrix QC, missing-value imputation, position sorting
│   └── visualization.py       # spectral landscape, model comparison, attention weights
├── train.py                   # CLI entry point
├── run_comparison.py          # 4-model benchmark script
├── test_install.py
└── test_all_models.py
```

## Reproducibility

- All training uses seed `42` (numpy, torch).
- 5-fold CV with shuffled `KFold(random_state=42)`.
- Out-of-fold predictions are concatenated for unbiased MAE/r/R² estimates.
- A trained model checkpoint is saved per run under `results/`.

## Limitations and future work

1. **Position sorting requires a probe manifest.** Currently optional; full Illumina 450K manifest integration is a TODO.
2. **CPU benchmarks subsample CpGs.** GPU training on the full 473K-CpG matrix is needed for a publication-grade comparison against Horvath/Hannum.
3. **Tissue-specific models** are not yet implemented; current results are whole-blood only.
4. **Cross-cohort generalization** (train on GSE40279, test on GSE87571) has not been evaluated.

## Citation

If this work is useful in your research, please cite:

```bibtex
@software{spectralage2025,
  author = {Keser, Yusuf},
  title  = {SpectralAge: a frequency-domain neural network for epigenetic age prediction},
  year   = {2025},
  url    = {https://github.com/yusuf4lp/spectral-age}
}
```

## References

- Horvath, S. (2013). DNA methylation age of human tissues and cell types. *Genome Biology* 14:R115.
- Hannum, G. et al. (2013). Genome-wide methylation profiles reveal quantitative views of human aging rates. *Molecular Cell* 49:359–367.
- Levine, M. E. et al. (2018). An epigenetic biomarker of aging for lifespan and healthspan (PhenoAge). *Aging* 10:573–591.
- Lu, A. T. et al. (2019). DNA methylation GrimAge strongly predicts lifespan and healthspan. *Aging* 11:303–327.
- Miconi, T., Stanley, K. O., Clune, J. (2018). Differentiable plasticity: training plastic neural networks with backpropagation. *ICML*.
- Veličković, P. et al. (2018). Graph Attention Networks. *ICLR*.

## License

MIT — see [LICENSE](LICENSE).

## Contributing

Contributions, bug reports, and methodological critiques are welcome. Open an issue or pull request.

## Contact

Yusuf Keser — kesermkkdkd@gmail.com
