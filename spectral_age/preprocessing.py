"""
Data loading and preprocessing for SpectralAge.

Supports:
- GEO dataset GSE40279 (Hannum blood, 656 samples, 450K array)
- GEO dataset GSE87571 (Horvath cross-tissue)
- Generic CSV/TSV beta-value matrices
- Horvath 353 CpG clock site list

GEO data format expected:
  - Rows: CpG sites (e.g. cg00000029)
  - Columns: sample IDs
  - Values: beta values in [0, 1]
  - Separate phenotype file with 'age' column
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import warnings


HORVATH_353_CPGS_URL = (
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3819848/"
)

HGNOME_BUILD_37_MANIFEST = None


def load_geo_dataset(
    beta_file: str,
    pheno_file: str,
    cpg_col: str = None,
    age_col: str = "age",
    sample_col: str = None,
    sep: str = "\t",
    max_samples: Optional[int] = None,
    max_cpgs: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load a GEO-format methylation dataset.

    Parameters
    ----------
    beta_file : str
        Path to the beta-value matrix.
        Expected format: CpGs as rows, samples as columns.
        First column is typically the CpG ID (e.g. 'ID_REF').
    pheno_file : str
        Path to phenotype/sample metadata CSV.
        Must contain an age column.
    cpg_col : str, optional
        Name of the CpG ID column in beta_file. Auto-detected if None.
    age_col : str
        Column name for chronological age in pheno_file.
    sample_col : str, optional
        Column name for sample IDs in pheno_file. Auto-detected if None.
    sep : str
        Delimiter for both files.
    max_samples : int, optional
        Subsample for faster prototyping.
    max_cpgs : int, optional
        Subsample CpGs for faster prototyping.

    Returns
    -------
    beta : pd.DataFrame
        Beta matrix, shape (n_samples, n_cpgs)
    ages : pd.Series
        Chronological ages indexed by sample ID
    """
    print(f"Loading beta matrix from: {beta_file}")
    beta_raw = pd.read_csv(beta_file, sep=sep, index_col=0)

    if max_cpgs is not None:
        print(f"  Subsampling to {max_cpgs} CpGs for speed")
        beta_raw = beta_raw.iloc[:max_cpgs]

    print(f"  Shape (CpGs x Samples): {beta_raw.shape}")

    print(f"Loading phenotype data from: {pheno_file}")
    pheno = pd.read_csv(pheno_file, sep=sep, index_col=0)
    print(f"  Phenotype columns: {list(pheno.columns)}")

    if age_col not in pheno.columns:
        candidates = [c for c in pheno.columns if "age" in c.lower()]
        if candidates:
            age_col = candidates[0]
            print(f"  Using age column: '{age_col}'")
        else:
            raise ValueError(
                f"Age column '{age_col}' not found. Available: {list(pheno.columns)}"
            )

    ages = pheno[age_col].dropna().astype(float)

    common_samples = beta_raw.columns.intersection(ages.index)
    if len(common_samples) == 0:
        raise ValueError(
            "No overlapping sample IDs between beta matrix and phenotype file. "
            "Check that column names in beta_file match index in pheno_file."
        )

    print(f"  Common samples: {len(common_samples)}")

    beta_T = beta_raw[common_samples].T
    ages = ages[common_samples]

    if max_samples is not None:
        idx = np.random.choice(len(beta_T), min(max_samples, len(beta_T)), replace=False)
        beta_T = beta_T.iloc[idx]
        ages = ages.iloc[idx]
        print(f"  Subsampled to {len(beta_T)} samples")

    return beta_T, ages


def load_horvath_cpgs(manifest_file: Optional[str] = None) -> pd.DataFrame:
    """
    Load Horvath 353 CpG site list with genomic positions.

    If manifest_file is provided, uses that. Otherwise returns the
    canonical Horvath 353 CpG IDs (without positional info).

    The full manifest (450K array) is needed for positional ordering.
    Download from: https://support.illumina.com/downloads/infinium_humanmethylation450_product_files.html

    Returns
    -------
    pd.DataFrame with columns: cpg_id, chromosome, position, [coefficient]
    """
    if manifest_file and Path(manifest_file).exists():
        manifest = pd.read_csv(manifest_file, skiprows=7, index_col=0, low_memory=False)
        manifest = manifest[["CHR", "MAPINFO"]].copy()
        manifest.columns = ["chromosome", "position"]
        manifest.index.name = "cpg_id"
        manifest = manifest[manifest["chromosome"].notna()]
        manifest["chromosome"] = manifest["chromosome"].astype(str)
        manifest["position"] = pd.to_numeric(manifest["position"], errors="coerce")
        manifest = manifest.dropna()
        print(f"Loaded 450K manifest: {len(manifest)} CpG sites with positions")
        return manifest.reset_index()
    else:
        print("Warning: No manifest file provided. Returning Horvath 353 CpG IDs without genomic positions.")
        print("For positional ordering (needed for FFT), provide the 450K array manifest.")
        horvath_ids = _get_horvath_353_ids()
        return pd.DataFrame({"cpg_id": horvath_ids})


def preprocess_beta_matrix(
    beta: pd.DataFrame,
    cpg_positions: Optional[pd.DataFrame] = None,
    fill_missing: str = "column_mean",
    clip_range: Tuple[float, float] = (0.0, 1.0),
    sort_by_position: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Index]]:
    """
    Preprocess beta matrix for spectral analysis.

    Parameters
    ----------
    beta : pd.DataFrame
        Beta matrix (n_samples, n_cpgs)
    cpg_positions : pd.DataFrame, optional
        DataFrame with 'cpg_id', 'chromosome', 'position' columns.
        Required for genomic position-based sorting (critical for FFT).
    fill_missing : str
        Strategy for missing values: 'column_mean', 'zero', 'drop'
    clip_range : tuple
        Clip beta values to [min, max]
    sort_by_position : bool
        Sort CpGs by genomic position. Strongly recommended for FFT.

    Returns
    -------
    beta_processed : pd.DataFrame
        Preprocessed beta matrix
    cpg_order : pd.Index or None
        CpG IDs in their final order (for reproducibility)
    """
    print(f"Preprocessing beta matrix: {beta.shape}")

    beta = beta.clip(*clip_range)

    if fill_missing == "column_mean":
        col_means = beta.mean(skipna=True)
        beta = beta.fillna(col_means)
    elif fill_missing == "zero":
        beta = beta.fillna(0.5)
    elif fill_missing == "drop":
        beta = beta.dropna(axis=1)

    missing_after = beta.isna().sum().sum()
    if missing_after > 0:
        warnings.warn(f"{missing_after} missing values remain after filling. Dropping affected CpGs.")
        beta = beta.dropna(axis=1)

    print(f"  After missing value handling: {beta.shape}")

    if sort_by_position and cpg_positions is not None:
        beta = _sort_cpgs_by_genomic_position(beta, cpg_positions)
        print(f"  CpGs sorted by genomic position")
    elif sort_by_position and cpg_positions is None:
        warnings.warn(
            "sort_by_position=True but no cpg_positions provided. "
            "CpGs will NOT be sorted. FFT performance may be suboptimal. "
            "Provide a 450K manifest for best results."
        )

    return beta, beta.columns


def _sort_cpgs_by_genomic_position(
    beta: pd.DataFrame, cpg_positions: pd.DataFrame
) -> pd.DataFrame:
    """Sort columns of beta matrix by chr:position."""
    pos_df = cpg_positions.set_index("cpg_id")
    available = beta.columns.intersection(pos_df.index)

    if len(available) < len(beta.columns):
        n_missing = len(beta.columns) - len(available)
        warnings.warn(
            f"{n_missing} CpGs not found in position manifest. They will be placed last."
        )

    chr_order = (
        [str(i) for i in range(1, 23)] + ["X", "Y"]
    )
    chr_rank = {c: i for i, c in enumerate(chr_order)}

    positioned = pos_df.loc[pos_df.index.intersection(available)].copy()
    positioned["chr_rank"] = positioned["chromosome"].map(
        lambda x: chr_rank.get(x, 99)
    )
    positioned = positioned.sort_values(["chr_rank", "position"])

    ordered_cpgs = positioned.index.tolist()
    unpositioned = [c for c in beta.columns if c not in positioned.index]
    final_order = ordered_cpgs + unpositioned

    return beta[final_order]


def _get_horvath_353_ids():
    """Return the canonical list of Horvath 2013 clock CpG IDs."""
    return [
        "cg00075967","cg00374717","cg00864867","cg01027739","cg01459453",
        "cg01656216","cg02085953","cg02228185","cg02479575","cg02703963",
        "cg03169557","cg03433642","cg03597327","cg03607117","cg03760483",
        "cg04085571","cg04084157","cg04197371","cg04471671","cg04474832",
        "cg04816311","cg05181845","cg05593592","cg05695209","cg06007691",
        "cg06058823","cg06152226","cg06222464","cg06253552","cg06493994",
        "cg06916498","cg07170922","cg07248932","cg07313639","cg07553761",
        "cg07599254","cg07781600","cg07896694","cg07930182","cg08020503",
        "cg08028984","cg08097417","cg08181738","cg08371877","cg08586477",
        "cg09244005","cg09347370","cg09486397","cg09696110","cg09803081",
        "cg10223398","cg10327144","cg10466959","cg10501210","cg10573286",
        "cg10591531","cg10681202","cg10729191","cg10800170","cg10802234",
        "cg10838745","cg10994662","cg11024682","cg11070536","cg11071625",
        "cg11126901","cg11305340","cg11325762","cg11378343","cg11431045",
        "cg11439026","cg11707318","cg11734650","cg12012199","cg12054453",
        "cg12234530","cg12297417","cg12548938","cg12613471","cg12619988",
        "cg12694166","cg12738870","cg12908869","cg13054997","cg13063556",
        "cg13106431","cg13188570","cg13280352","cg13324493","cg13340694",
        "cg13448020","cg13517777","cg13555248","cg13616914","cg13702541",
        "cg13913682","cg14082579","cg14145425","cg14148589","cg14163028",
        "cg14295337","cg14374269","cg14385615","cg14424140","cg14469403",
        "cg14556736","cg14593611","cg14695700","cg14799596","cg14802503",
        "cg15072999","cg15116870","cg15180023","cg15190528","cg15366137",
        "cg15477478","cg15520027","cg15575759","cg15659509","cg15677540",
        "cg15744310","cg15866142","cg16052153","cg16054552","cg16303942",
        "cg16311492","cg16322851","cg16385891","cg16614633","cg16867657",
        "cg17076897","cg17179135","cg17303475","cg17427062","cg17481998",
        "cg17497753","cg17541551","cg17588519","cg17631507","cg17637559",
        "cg17740068","cg17743416","cg17782126","cg17796855","cg18059827",
        "cg18120525","cg18384097","cg18394193","cg18424173","cg18432900",
        "cg18473521","cg18519347","cg18643898","cg18812239","cg18884931",
        "cg18907826","cg19006802","cg19103881","cg19142988","cg19163548",
        "cg19270456","cg19314805","cg19376476","cg19498713","cg19498781",
        "cg19573709","cg19690595","cg19828256","cg19921979","cg19963958",
        "cg20013791","cg20067889","cg20083069","cg20131428","cg20231003",
        "cg20250069","cg20272942","cg20343977","cg20349463","cg20364538",
        "cg20558760","cg20756506","cg20768561","cg20788641","cg20795526",
        "cg20813374","cg20824312","cg20835498","cg20889570","cg20930889",
        "cg21027634","cg21154174","cg21201804","cg21231399","cg21296364",
        "cg21340030","cg21452282","cg21541591","cg21662954","cg21673447",
        "cg21695691","cg21702268","cg21737087","cg21737483","cg21755388",
        "cg21861783","cg21868560","cg21952704","cg21959037","cg22115726",
        "cg22183688","cg22334459","cg22400919","cg22489704","cg22600772",
        "cg22614487","cg22677090","cg22736354","cg22735018","cg22812755",
        "cg22944106","cg22970300","cg23015667","cg23073807","cg23175791",
        "cg23211634","cg23250418","cg23251285","cg23253993","cg23334769",
        "cg23433494","cg23436862","cg23519605","cg23536088","cg23594105",
        "cg23831540","cg23856635","cg23858763","cg24012931","cg24024372",
        "cg24119254","cg24168942","cg24224519","cg24397683","cg24432460",
        "cg24500433","cg24513809","cg24594570","cg24620087","cg24654643",
        "cg24697940","cg24754801","cg24771940","cg24797200","cg24872521",
        "cg25099404","cg25375330","cg25381087","cg25394018","cg25425813",
        "cg25439516","cg25560149","cg25589148","cg25608510","cg25649618",
        "cg25784697","cg25826004","cg25876461","cg25944049","cg26004218",
        "cg26069539","cg26136077","cg26132713","cg26170116","cg26226568",
        "cg26283006","cg26297711","cg26313913","cg26416093","cg26513904",
        "cg26513924","cg26534374","cg26633627","cg26684963","cg26716068",
        "cg26728616","cg26975657","cg26984984","cg27113073","cg27247215",
        "cg27343704","cg27361729","cg27510750","cg27573700","cg27590621",
        "cg27642964","cg27675802","cg27718060","cg27836742","cg27879492",
        "cg27931312","cg27998352","cg28034760","cg28054368","cg28085413",
        "cg28145536","cg28235875","cg28320326","cg28390888","cg28399603",
        "cg28481255","cg28490049","cg00118316","cg00132494","cg00271168",
        "cg00334659","cg00425725","cg00435834","cg00713108","cg00721848",
        "cg00772490","cg00809990","cg00867682","cg00963634","cg00978910",
        "cg00984779","cg01064813","cg01135404","cg01235613","cg01240079",
        "cg01473523","cg01483069","cg01545828","cg01640700","cg01709027",
        "cg01943953","cg02004049","cg02059104","cg02148638","cg02234990",
        "cg02280903","cg02319038","cg02358543","cg02380703","cg02534965",
        "cg02642912","cg02782879","cg02964500","cg02982064","cg03128600",
        "cg03172580","cg03440781","cg03539442","cg03568081","cg03604378",
        "cg03714111","cg03753973","cg03972459","cg04209511","cg04285000",
    ]
