"""
GEO Dataset Downloader and Loader

Desteklenen datasetler:
  GSE40279  — Hannum 2013, whole blood, n=656, 450K array
  GSE87571  — Horvath cross-tissue (bazı çalışmalarda kullanılan)
  GSE55763  — whole blood, n=2711 (büyük cohort)

GEO'dan .soft.gz veya matrix dosyaları indirilir.
"""

import os
import gzip
import shutil
import urllib.request
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd


GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"

DATASETS = {
    "GSE40279": {
        "desc": "Hannum 2013 — whole blood, n=656, 450K",
        "series": "GSE40nnn",
        "matrix_url": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE40nnn/GSE40279/matrix/"
            "GSE40279_series_matrix.txt.gz"
        ),
        "soft_url": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE40nnn/GSE40279/soft/"
            "GSE40279_family.soft.gz"
        ),
    },
    "GSE87571": {
        "desc": "Horvath cross-tissue, blood, n=750+",
        "series": "GSE87nnn",
        "matrix_url": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE87nnn/GSE87571/matrix/"
            "GSE87571_series_matrix.txt.gz"
        ),
    },
    "GSE55763": {
        "desc": "Whole blood, n=2711 — large cohort",
        "series": "GSE55nnn",
        "matrix_url": (
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE55nnn/GSE55763/matrix/"
            "GSE55763_series_matrix.txt.gz"
        ),
    },
}


def download_geo_matrix(
    gse_id: str,
    data_dir: str = "./data",
    force_download: bool = False,
) -> Path:
    """
    GEO series matrix dosyasını indir.

    Parameters
    ----------
    gse_id : str
        "GSE40279", "GSE87571" vb.
    data_dir : str
        Dosyaların kaydedileceği klasör.
    force_download : bool
        True ise mevcut dosya üzerine yaz.

    Returns
    -------
    Path
        İndirilen .txt.gz dosyasının yolu.
    """
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    if gse_id not in DATASETS:
        raise ValueError(f"Bilinmeyen GSE ID: {gse_id}. Desteklenenler: {list(DATASETS.keys())}")

    info = DATASETS[gse_id]
    url = info["matrix_url"]
    filename = data_path / f"{gse_id}_series_matrix.txt.gz"

    if filename.exists() and not force_download:
        print(f"Mevcut dosya kullanılıyor: {filename}")
        return filename

    print(f"\nIndiriliyor: {gse_id} ({info['desc']})")
    print(f"URL: {url}")
    print(f"Hedef: {filename}")
    print("Bu işlem dataset boyutuna göre 1-30 dakika sürebilir...")

    def _progress(block_count, block_size, total_size):
        downloaded = block_count * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            mb = downloaded / 1e6
            total_mb = total_size / 1e6
            print(f"\r  {pct:.1f}%  ({mb:.1f}/{total_mb:.1f} MB)", end="", flush=True)
        else:
            print(f"\r  {downloaded/1e6:.1f} MB indirild", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, filename, reporthook=_progress)
        print(f"\nTamamlandi: {filename}")
    except Exception as e:
        if filename.exists():
            filename.unlink()
        raise RuntimeError(f"Indirme hatasi: {e}")

    return filename


def parse_geo_matrix(
    matrix_file: str,
    age_keyword: str = "age",
    max_cpgs: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    GEO series_matrix.txt.gz dosyasini ayristir.

    GEO matrix formati:
      - !Sample_geo_accession satiri: ornek IDleri
      - !Sample_characteristics_ch1 satiri: yas dahil fenotip bilgisi
      - "!series_matrix_table_begin" sonrasi: beta deger matrisi
        (satir = CpG, sutun = ornek)

    Returns
    -------
    beta : pd.DataFrame  shape (n_samples, n_cpgs)
    ages : pd.Series     indeks = ornek ID
    """
    matrix_file = str(matrix_file)
    print(f"\nGEO matrix ayristiriliyor: {matrix_file}")

    open_fn = gzip.open if matrix_file.endswith(".gz") else open

    sample_ids = []
    age_rows = []
    header_lines = []
    in_table = False
    table_lines = []

    with open_fn(matrix_file, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("!Sample_geo_accession"):
                parts = line.split("\t")
                sample_ids = [p.strip('"') for p in parts[1:]]
                continue

            if line.startswith("!Sample_characteristics_ch1"):
                lower = line.lower()
                if age_keyword in lower:
                    age_rows.append(line)
                continue

            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue

            if line.startswith("!series_matrix_table_end"):
                break

            if in_table:
                table_lines.append(line)

    print(f"  {len(sample_ids)} ornek bulundu")
    print(f"  {len(table_lines)-1} CpG satiri bulundu")

    if not sample_ids:
        raise ValueError("Ornek IDleri bulunamadi. Dosya formati farkli olabilir.")
    if not table_lines:
        raise ValueError("Beta deger tablosu bulunamadi.")

    print("  Beta matrisi olusturuluyor...")

    cpg_ids = []
    values = []

    header_done = False
    for i, line in enumerate(table_lines):
        if not header_done:
            header_done = True
            continue

        if max_cpgs and len(cpg_ids) >= max_cpgs:
            break

        parts = line.split("\t")
        if len(parts) < 2:
            continue
        cpg_id = parts[0].strip('"')
        try:
            vals = [float(v) if v not in ("", "NA", "null", "NULL") else np.nan
                    for v in parts[1:len(sample_ids)+1]]
        except ValueError:
            continue
        cpg_ids.append(cpg_id)
        values.append(vals)

    if not cpg_ids:
        raise ValueError("Hicbir gecerli CpG satiri ayristirilmadi.")

    beta_mat = np.array(values, dtype=np.float32).T
    beta = pd.DataFrame(beta_mat, index=sample_ids, columns=cpg_ids)

    print(f"  Beta matrisi: {beta.shape} (n_ornek, n_CpG)")

    ages = _parse_ages_from_header(age_rows, sample_ids)

    if ages is None or len(ages) == 0:
        raise ValueError(
            "Yas bilgisi ayristirilmadi. 'age_keyword' parametresini kontrol et. "
            f"Mevcut karakteristik satirlari:\n" + "\n".join(age_rows[:5])
        )

    common = beta.index.intersection(ages.index)
    beta = beta.loc[common]
    ages = ages[common]

    print(f"  Eslesme sonrasi: {len(common)} ornek, yas aralik {ages.min():.0f}-{ages.max():.0f}")

    if max_samples and len(beta) > max_samples:
        idx = np.random.choice(len(beta), max_samples, replace=False)
        beta = beta.iloc[idx]
        ages = ages.iloc[idx]
        print(f"  Alt ornekleme: {max_samples} ornek")

    return beta, ages


def _parse_ages_from_header(age_rows: list, sample_ids: list) -> Optional[pd.Series]:
    """
    Farkli GEO formatlarin yasini ayristir.

    Ornek formatlar:
      "!Sample_characteristics_ch1"  "age: 45"  "age: 67"  ...
      "!Sample_characteristics_ch1"  "Age: 45.2" ...
    """
    if not age_rows:
        return None

    ages = {}
    for row in age_rows:
        parts = row.split("\t")
        vals = [p.strip('"').strip() for p in parts[1:]]
        for sample_id, val in zip(sample_ids, vals):
            val_lower = val.lower()
            import re
            age_match = re.search(r'age[^:]*:\s*([\d.]+)', val_lower)
            if age_match:
                try:
                    age_num = float(age_match.group(1))
                    if 0 < age_num < 150:
                        ages[sample_id] = age_num
                except ValueError:
                    pass
            else:
                for prefix in ["age:", "age =", "age="]:
                    if val_lower.startswith(prefix):
                        try:
                            age_num = float(val[len(prefix):].strip())
                            ages[sample_id] = age_num
                        except ValueError:
                            pass
                        break
                else:
                    try:
                        age_num = float(val)
                        if 0 < age_num < 120:
                            ages[sample_id] = age_num
                    except ValueError:
                        pass

    if ages:
        return pd.Series(ages, name="age")
    return None


def load_450k_manifest(manifest_file: str) -> pd.DataFrame:
    """
    Illumina 450K array manifest dosyasini yukle.

    Manifest: https://support.illumina.com/downloads/infinium_humanmethylation450_product_files.html
    (HumanMethylation450_15017482_v1-2.csv)

    Returns pd.DataFrame: cpg_id, chromosome, position
    """
    manifest_file = str(manifest_file)
    print(f"450K manifest yukleniyor: {manifest_file}")

    if manifest_file.endswith(".gz"):
        df = pd.read_csv(manifest_file, compression="gzip", skiprows=7,
                         index_col=0, low_memory=False)
    else:
        df = pd.read_csv(manifest_file, skiprows=7, index_col=0, low_memory=False)

    df = df[["CHR", "MAPINFO"]].copy()
    df.columns = ["chromosome", "position"]
    df.index.name = "cpg_id"
    df = df[df["chromosome"].notna()]
    df["chromosome"] = df["chromosome"].astype(str)
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df = df.dropna()

    print(f"  {len(df)} CpG sitesi, {df['chromosome'].nunique()} kromozom")
    return df.reset_index()


def load_local_csv(
    beta_file: str,
    age_file: str,
    sep: str = ",",
    age_col: str = "age",
    sample_id_col: str = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Lokal CSV/TSV dosyasindan beta matrisi ve yas yukle.

    beta_file  : satir=ornek, sutun=CpG (veya tersi — otomatik algilanir)
    age_file   : 'age' (veya belirtilen) sutunu iceren CSV
    """
    print(f"Beta matrisi yukleniyor: {beta_file}")
    beta = pd.read_csv(beta_file, sep=sep, index_col=0)

    if beta.shape[0] > beta.shape[1]:
        print(f"  Transpoz aliniyor (satir>sutun: {beta.shape})")
        beta = beta.T

    print(f"  Shape: {beta.shape} (n_ornek x n_CpG)")

    print(f"Yas dosyasi yukleniyor: {age_file}")
    pheno = pd.read_csv(age_file, sep=sep, index_col=0)

    if age_col not in pheno.columns:
        candidates = [c for c in pheno.columns if "age" in c.lower()]
        if candidates:
            age_col = candidates[0]
            print(f"  Yas sutunu: '{age_col}'")
        else:
            raise ValueError(f"Yas sutunu bulunamadi. Mevcut: {list(pheno.columns)}")

    ages = pheno[age_col].dropna().astype(float)
    common = beta.index.intersection(ages.index)

    if len(common) == 0:
        raise ValueError(
            "Beta matrisi ve yas dosyasi arasinda ortak ornek ID bulunamadi."
        )

    beta = beta.loc[common]
    ages = ages[common]
    print(f"  {len(common)} ortak ornek, yas {ages.min():.0f}-{ages.max():.0f}")
    return beta, ages
