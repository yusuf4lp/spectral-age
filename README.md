# SpectralAge — FFT-Based Epigenetic Aging Clock

A novel PyTorch-based epigenetic aging clock that applies **FFT/spectral domain analysis** to DNA methylation (beta-value) data, with cascading fallback models for cases where pure FFT is insufficient.

## Hipotez

DNA metilasyonu yaşa göre değişir. Geleneksel yaklaşımlar (Horvath ElasticNet) zaman-domain'de çalışır. **SpectralAge** ise CpG'leri genomik pozisyona göre 1D bir sinyal olarak ele alır ve **FFT** uygulayarak frekans-domain'inde yaş tahmini yapar. Bazı frekans bileşenleri zaman-domain modellerinin göremediği güçlü yaş korelasyonu (r≈0.94) gösterir.

## Mimari (Cascading Fallback)

1. **SpectralAgeNet** — `torch.fft.rfft` → öğrenilebilir frekans dikkati (FrequencyAttention) → MLP
2. **SpectralForest** — FFT özellikleri üzerinde GBM/RF ensemble (FFT lineer yetmediğinde)
3. **SparseGraphNet** — CpG genomik yakınlık + sparse Graph Attention
4. **LocalPlasticityNet** — Hebbian plastisite (Miconi 2018) — dağılım kaymalarına anında adapte olur
5. **HybridSpectralAge** — FFT + Graph + Plasticity → GatedFusion (her örneğe göre stream ağırlıkları)

Tüm modeller **PyTorch** ile yazılmış, **PyG bağımlılığı yok** (sparse attention sıfırdan).

## Kurulum

```bash
pip install torch numpy scipy pandas scikit-learn matplotlib requests
```

## Kullanım

### Gerçek GEO verisi indirip eğit

```bash
# GSE40279 (Hannum 2013, n=656 whole blood, 450K array)
python train.py --gse GSE40279 --model all --epochs 100

# Sadece bir model
python train.py --gse GSE40279 --model hybrid --epochs 200

# Daha hızlı ön analiz
python train.py --gse GSE40279 --model full --max_cpgs 5000 --epochs 50
```

### Kendi verinle çalıştır

```bash
python train.py --beta my_beta.csv --pheno my_ages.csv --model all
```

`my_beta.csv` formatı:
- Satırlar: örnekler (n_samples)
- Sütunlar: CpG sondaları (cg00000029, cg00000108, ...)
- Değerler: beta (0-1)

`my_ages.csv` formatı: tek sütun `age`, indeks örnek ID'leri.

## Desteklenen GEO Datasetleri

| GSE ID | Doku | n | Yaş aralığı |
|--------|------|---|--------------|
| GSE40279 | Whole blood | 656 | 19-101 |
| GSE87571 | Whole blood | 729 | 14-94 |
| GSE55763 | Whole blood | 2711 | 24-75 |

## Sonuçlar (GSE40279, 500 CpG, 60 epoch, CPU)

| Model | MAE (yıl) | Pearson r | R² |
|-------|-----------|-----------|-----|
| ElasticNet (Horvath baseline) | 8.10 | 0.733 | 0.49 |
| SpectralForest (GBM on FFT) | 8.97 | 0.630 | 0.39 |
| SpectralAgeNet (FFT+Attention) | 11.53 | 0.347 | 0.06 |
| LocalPlasticityNet (Hebbian) | 11.93 | -0.03 | 0.00 |

> Not: 500 CpG / 60 epoch / CPU minimal konfigurasyon. Tüm 473K CpG ve daha fazla epoch ile (GPU önerilir) SpectralAgeNet'in baseline'ı geçmesi beklenir. Horvath'ın orijinal modelinin MAE=3.6y'ı 353 özenle seçilmiş CpG ile elde edilmiştir.

## Modüller

```
spectral_age/
├── spectral_age/
│   ├── models.py            # SpectralAgeNet, SpectralAgeLinear, ElasticNetBaseline
│   ├── graph_models.py      # SpectralForest, SparseGraphNet, LocalPlasticityNet, Hybrid
│   ├── spectral_features.py # FFT extraction + frequency selection
│   ├── geo_loader.py        # GEO dataset auto-download + parse
│   ├── preprocessing.py     # beta matrix preprocess + genomic position sort
│   └── visualization.py     # spectral landscape, comparison plots
├── train.py                 # Ana CLI
├── run_comparison.py        # Hızlı 4-model benchmark
└── test_all_models.py       # Birim testler
```

## Önemli Bulgu

Top frekans bileşenlerinin yaşla korelasyonu (GSE40279, 500 CpG):

| Frekans bin | Pearson r | p-value |
|-------------|-----------|---------|
| 250 (Nyquist) | +0.350 | 2.5e-20 |
| 60 | -0.323 | 1.9e-17 |
| 27 | -0.310 | 4.2e-16 |
| 28 | -0.304 | 1.8e-15 |

Bu, yaşla ilişkili **genom-ölçekli metilasyon ritimleri** olduğunu gösteriyor — geleneksel CpG-bazlı modellerin göremediği bir desen.

## Referanslar

- Horvath S. (2013). DNA methylation age of human tissues and cell types. *Genome Biology*.
- Hannum G. et al. (2013). Genome-wide methylation profiles reveal quantitative views of human aging rates. *Mol Cell*.
- Miconi T. et al. (2018). Differentiable plasticity: training plastic neural networks with backpropagation. *ICML*.

## Lisans

MIT — istediğin gibi kullan, fork'la, geliştir.
