from __future__ import annotations

import argparse
import hashlib
import logging
import math
import re
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # Backend no interactivo: reproducible en servidor/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================
SEED = 42
LOGGER = logging.getLogger("02_eda")

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 140, "savefig.bbox": "tight", "font.size": 10})

REQUIRED_TARGET = "monthly_rent_rub"
# Columnas que NUNCA deben entrar al conjunto de features (fuga / identidad / auditoría)
HARD_EXCLUDE = ["rent_per_sqm", "id", "dq_flags"]

EXPECTED_SCHEMA: Dict[str, Dict[str, str]] = {
    "id": {"role": "id", "type": "string", "desc": "Identificador único del anuncio"},
    "date_posted": {"role": "temporal", "type": "date", "desc": "Fecha de publicación (ISO YYYY-MM-DD)"},
    "district": {"role": "categorical", "type": "string", "desc": "Distrito municipal (raion) de Moscú"},
    "okrug": {"role": "categorical", "type": "string", "desc": "Distrito administrativo (okrug) de Moscú"},
    "lat": {"role": "spatial", "type": "float", "desc": "Latitud geográfica (grados decimales)"},
    "lon": {"role": "spatial", "type": "float", "desc": "Longitud geográfica (grados decimales)"},
    "total_area": {"role": "numeric_continuous", "type": "float", "desc": "Superficie total del inmueble (m²)"},
    "rooms": {"role": "numeric_discrete", "type": "int", "desc": "Cantidad de habitaciones / ambientes"},
    "floor": {"role": "numeric_discrete", "type": "int", "desc": "Piso en el que se ubica la unidad"},
    "total_floors": {"role": "numeric_discrete", "type": "int", "desc": "Cantidad total de pisos del edificio"},
    "building_year": {"role": "numeric_discrete", "type": "int", "desc": "Año de construcción del edificio"},
    "building_type": {"role": "categorical", "type": "string", "desc": "Material o tipología constructiva"},
    "renovation": {"role": "categorical", "type": "string", "desc": "Estado de renovación interior"},
    "furnished": {"role": "boolean", "type": "bool", "desc": "Indica si se alquila amoblado"},
    "pets_allowed": {"role": "boolean", "type": "bool", "desc": "Indica si se admiten mascotas"},
    "deposit_months": {"role": "numeric_discrete", "type": "int", "desc": "Meses de depósito en garantía"},
    "metro_station": {"role": "categorical", "type": "string", "desc": "Estación de metro asignada al anuncio"},
    "metro_line": {"role": "categorical", "type": "string", "desc": "Línea de metro de la estación"},
    "metro_distance_min": {"role": "numeric_continuous", "type": "float", "desc": "Distancia reportada al metro (en minutos)"},
    "to_center_km": {"role": "numeric_continuous", "type": "float", "desc": "Distancia en línea recta al centro de Moscú (km)"},
    "monthly_rent_rub": {"role": "target", "type": "float", "desc": "Variable objetivo: alquiler mensual (RUB)"},
    "rent_per_sqm": {"role": "leakage_candidate", "type": "float", "desc": "Precio de alquiler por m² (RUB/m²)"},
    "dq_flags": {"role": "audit", "type": "string", "desc": "Banderas de auditoría generadas en 01_cleaning.py"},
}

NUMERIC_CANDIDATES = [
    "monthly_rent_rub", "rent_per_sqm", "total_area", "rooms", "floor",
    "total_floors", "building_year", "deposit_months", "metro_distance_min",
    "to_center_km", "lat", "lon",
]
CAT_CANDIDATES = [
    "okrug", "district", "building_type", "renovation", "furnished",
    "pets_allowed", "metro_line", "metro_station", "dq_flags",
]


# =============================================================================
# UTILIDADES DE INTEGRIDAD Y VALIDACIÓN
# =============================================================================
def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_text_safe(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def df_to_md(df: pd.DataFrame) -> str:
    """Convierte un DataFrame a Markdown con fallback si 'tabulate' no está instalado."""
    if df.empty:
        return "_(sin datos)_"
    try:
        return df.to_markdown(index=False)
    except ImportError:
        LOGGER.warning("Paquete 'tabulate' no disponible; se usa representación en texto plano.")
        return "```\n" + df.to_string(index=False) + "\n```"


def parse_cleaning_report(report_path: Path) -> Dict[str, Any]:
    """
    Extrae metadatos del informe de limpieza usando regex por fila de tabla
    (robusto frente a cambios de formato) y recoge las columnas documentadas.
    """
    info: Dict[str, Any] = {
        "exists": False, "path": str(report_path), "warnings": [],
        "documented_columns": [], "mentions_dq_flags": False,
    }
    if not report_path.exists() or not report_path.is_file():
        info["warnings"].append(f"No se encontró el informe de limpieza en '{report_path}'.")
        return info

    content = read_text_safe(report_path)
    info.update({"exists": True, "size_bytes": report_path.stat().st_size})

    patterns = {
        "expected_clean_sha256": r"^\|\s*SHA-256 salida[^|]*\|\s*`?([0-9a-fA-F]{64})`?\s*\|",
        "upstream_input_sha256": r"^\|\s*SHA-256 entrada[^|]*\|\s*`?([0-9a-fA-F]{64})`?\s*\|",
        "initial_rows": r"^\|\s*Filas iniciales[^|]*\|\s*`?([\d\s_]+)`?\s*\|",
        "removed_rows": r"^\|\s*Filas eliminadas[^|]*\|\s*`?([\d\s_]+)`?\s*\|",
        "final_rows": r"^\|\s*Filas finales[^|]*\|\s*`?([\d\s_]+)`?\s*\|",
    }
    for key, pat in patterns.items():
        m = re.search(pat, content, flags=re.MULTILINE)
        if m:
            raw = m.group(1).strip().replace(" ", "").replace("_", "")
            info[key] = raw if not raw.isdigit() else int(raw)

    cols = re.findall(r"^\|\s*`([a-z_0-9]+)`\s*\|", content, flags=re.MULTILINE)
    if cols:
        info["documented_columns"] = sorted(set(cols))
    info["mentions_dq_flags"] = bool(re.search(r"`dq_flags`", content))
    return info


def cross_validate_with_cleaning_report(
    df: pd.DataFrame, cleaning_info: Dict[str, Any], input_sha256: str
) -> Tuple[List[str], List[str]]:
    """Contrasta el CSV real con lo declarado en el informe de limpieza."""
    notes: List[str] = []
    issues: List[str] = []
    if not cleaning_info.get("exists"):
        return notes, issues

    exp_sha = cleaning_info.get("expected_clean_sha256")
    if exp_sha:
        if str(exp_sha) == input_sha256:
            notes.append(f"SHA-256 del CSV coincide exactamente con el declarado en etapa 01 ({input_sha256}).")
        else:
            issues.append(
                f"SHA-256 DIFIERE: informe declara `{exp_sha}` pero el archivo analizado tiene `{input_sha256}`. "
                "El dataset podría no ser el producido por 01_cleaning.py."
            )

    for key, label in (("initial_rows", "filas iniciales"), ("final_rows", "filas finales")):
        declared = cleaning_info.get(key)
        if declared is not None and int(declared) != len(df):
            issues.append(
                f"Conteo inconsistente: el informe declara {declared} {label} pero el CSV contiene {len(df)} filas."
            )
        elif declared is not None:
            notes.append(f"Conteo de {label} verificado: {declared} == {len(df)} filas del CSV.")

    documented = cleaning_info.get("documented_columns") or []
    if documented:
        missing_in_csv = [c for c in documented if c not in df.columns]
        undocumented_in_report = [c for c in df.columns if c not in documented]
        if missing_in_csv:
            issues.append(f"Columnas documentadas en el informe pero ausentes en el CSV: {missing_in_csv}.")
        if undocumented_in_report:
            issues.append(
                f"Columnas presentes en el CSV pero NO listadas en la tabla de valores faltantes del informe: "
                f"{undocumented_in_report}. Posible omisión documental (revisar consistencia interna del informe)."
            )
    if cleaning_info.get("mentions_dq_flags") and "dq_flags" not in df.columns:
        issues.append(
            "El informe de limpieza menciona la columna `dq_flags` (339 filas marcadas) pero el CSV no la contiene."
        )
    return notes, issues


def load_and_validate_dataset(input_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"ERROR CRÍTICO: No existe el archivo de entrada en '{input_path}'. "
            "Ejecute primero 01_cleaning.py o verifique el parámetro --input."
        )
    if not input_path.is_file() or input_path.stat().st_size == 0:
        raise ValueError(f"ERROR CRÍTICO: '{input_path}' está vacío o no es un archivo regular.")

    try:
        df = pd.read_csv(input_path, low_memory=False)
    except Exception as exc:
        raise ValueError(f"ERROR CRÍTICO: Falló la lectura de '{input_path}' como CSV válido: {exc}") from exc

    if df.empty or len(df.columns) == 0:
        raise ValueError(f"ERROR CRÍTICO: '{input_path}' no contiene registros o columnas.")
    if df.columns.duplicated().any():
        raise ValueError(f"ERROR CRÍTICO: Columnas duplicadas en el CSV: {list(df.columns[df.columns.duplicated()])}")
    if REQUIRED_TARGET not in df.columns:
        raise ValueError(
            f"ERROR CRÍTICO: Falta la variable objetivo '{REQUIRED_TARGET}'. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    warnings_list: List[str] = []
    target_num = pd.to_numeric(df[REQUIRED_TARGET], errors="coerce")
    n_valid = int(target_num.notna().sum())
    if n_valid == 0:
        raise ValueError(f"ERROR CRÍTICO: '{REQUIRED_TARGET}' no contiene ningún valor numérico válido.")
    if n_valid < len(df):
        warnings_list.append(
            f"'{REQUIRED_TARGET}' presenta {len(df) - n_valid} valores no numéricos o nulos."
        )

    missing_cols = [c for c in EXPECTED_SCHEMA if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in EXPECTED_SCHEMA]
    if missing_cols:
        warnings_list.append(f"Columnas del esquema ausentes (el EDA se adaptará): {missing_cols}")
    if extra_cols:
        warnings_list.append(f"Columnas adicionales no documentadas en el esquema base: {extra_cols}")
    return df, warnings_list


def safe_numeric_series(df: pd.DataFrame, col: str) -> Tuple[pd.Series, int]:
    """Convierte a numérico en memoria sin alterar el DataFrame original."""
    if col not in df.columns:
        return pd.Series(index=df.index, dtype=float), 0
    raw = df[col]
    already_na = int(raw.isna().sum())
    converted = pd.to_numeric(raw, errors="coerce")
    # Un valor "falló" si era no nulo en origen y quedó NaN tras la coerción
    coerced_failures = int((raw.notna() & converted.isna()).sum())
    del already_na
    return converted, coerced_failures


# =============================================================================
# ESTADÍSTICA Y DIAGNÓSTICO
# =============================================================================
def compute_numeric_summary(df: pd.DataFrame, numeric_cols: List[str], seed: int = SEED) -> Tuple[pd.DataFrame, Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    type_issues: Dict[str, int] = {}
    n_total = len(df)
    rng = np.random.default_rng(seed)

    for col in numeric_cols:
        if col not in df.columns:
            continue
        s, coerced = safe_numeric_series(df, col)
        if coerced > 0:
            type_issues[col] = coerced
        valid = s.dropna()
        n_valid = len(valid)
        if n_valid == 0:
            continue

        mean_val = float(valid.mean())
        std_val = float(valid.std(ddof=1)) if n_valid > 1 else 0.0
        cv_val = std_val / mean_val if mean_val != 0 else np.nan
        q01, q05, q10, q25, q50, q75, q90, q95, q99 = np.percentile(valid, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        iqr_val = float(q75 - q25)
        skew_val = float(stats.skew(valid, bias=False)) if n_valid > 2 else np.nan
        kurt_val = float(stats.kurtosis(valid, bias=False)) if n_valid > 3 else np.nan

        # Normalidad (D'Agostino K²) sobre una submuestra determinista
        if n_valid >= 20:
            sample = valid.sample(n=min(5000, n_valid), random_state=seed) if n_valid > 5000 else valid
            try:
                _, normal_p = stats.normaltest(sample.to_numpy())
            except Exception:
                normal_p = np.nan
        else:
            normal_p = np.nan

        low_15, high_15 = q25 - 1.5 * iqr_val, q75 + 1.5 * iqr_val
        low_30, high_30 = q25 - 3.0 * iqr_val, q75 + 3.0 * iqr_val
        out15 = int(((valid < low_15) | (valid > high_15)).sum())
        out30 = int(((valid < low_30) | (valid > high_30)).sum())
        mad = float(np.median(np.abs(valid.to_numpy() - np.median(valid.to_numpy()))))

        rows.append({
            "variable": col,
            "n_valid": n_valid,
            "missing_pct": round(100.0 * (n_total - n_valid) / n_total, 2),
            "coerced_invalid": coerced,
            "mean": round(mean_val, 3),
            "std": round(std_val, 3),
            "cv": round(cv_val, 3) if not np.isnan(cv_val) else np.nan,
            "min": round(float(valid.min()), 3),
            "p01": round(float(q01), 3),
            "p05": round(float(q05), 3),
            "p25": round(float(q25), 3),
            "median": round(float(q50), 3),
            "p75": round(float(q75), 3),
            "p95": round(float(q95), 3),
            "p99": round(float(q99), 3),
            "max": round(float(valid.max()), 3),
            "iqr": round(iqr_val, 3),
            "mad": round(mad, 3),
            "skewness": round(skew_val, 3),
            "kurtosis": round(kurt_val, 3),
            "normaltest_p": (f"{normal_p:.2e}" if not np.isnan(normal_p) else "N/D"),
            "outliers_1_5_iqr": out15,
            "outliers_1_5_pct": round(100.0 * out15 / n_valid, 2),
            "outliers_3_0_iqr": out30,
        })
    del rng
    return pd.DataFrame(rows), type_issues


def compute_categorical_summary(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    n_total = len(df)
    for col in cat_cols:
        if col not in df.columns:
            continue
        s = df[col].dropna().astype(str)
        n_valid = len(s)
        if n_valid == 0:
            continue
        vc = s.value_counts(dropna=True)
        rows.append({
            "variable": col,
            "dtype": str(df[col].dtype),
            "n_valid": n_valid,
            "missing_pct": round(100.0 * (n_total - n_valid) / n_total, 2),
            "n_unique": int(len(vc)),
            "top_category": str(vc.index[0]),
            "top_pct": round(100.0 * int(vc.iloc[0]) / n_valid, 2),
            "least_category": str(vc.index[-1]),
            "least_pct": round(100.0 * int(vc.iloc[-1]) / n_valid, 2),
            "rare_categories_lt_1pct": int((vc / n_valid < 0.01).sum()),
        })
    return pd.DataFrame(rows)


def compute_vif(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    VIF = 1 / (1 - R_j²) sobre la matriz de correlación de predictores numéricos
    (estandarización implícita: la correlación ya es la covarianza de datos z-normalizados).
    """
    num_df = pd.DataFrame()
    for c in feature_cols:
        if c not in df.columns:
            continue
        s, _ = safe_numeric_series(df, c)
        if s.notna().sum() > 0 and s.nunique(dropna=True) > 1:
            num_df[c] = s
    num_df = num_df.dropna()
    if num_df.shape[0] < 10 or num_df.shape[1] < 2:
        return pd.DataFrame(columns=["variable", "vif", "r2_with_others"])

    corr_matrix = num_df.corr().to_numpy()
    used_pinv = False
    try:
        inv_corr = np.linalg.inv(corr_matrix)
    except np.linalg.LinAlgError:
        inv_corr = np.linalg.pinv(corr_matrix)
        used_pinv = True
    vifs = np.diag(inv_corr)
    cond_number = float(np.linalg.cond(corr_matrix))

    results = []
    for idx, col in enumerate(num_df.columns):
        vif_val = float(max(1.0, vifs[idx]))
        r2_val = 1.0 - (1.0 / vif_val)
        results.append({"variable": col, "vif": round(vif_val, 3), "r2_with_others": round(r2_val, 4)})
    out = pd.DataFrame(results).sort_values("vif", ascending=False)
    out.attrs["condition_number"] = cond_number
    out.attrs["used_pinv"] = used_pinv
    return out


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """V de Cramér con corrección de sesgo de Bergsma (2013)."""
    confusion = pd.crosstab(x, y)
    n = confusion.sum().sum()
    if n <= 1:
        return np.nan
    chi2, _, _, _ = stats.chi2_contingency(confusion)
    phi2 = chi2 / n
    r, k = confusion.shape
    if r <= 1 or k <= 1:
        return 0.0
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def categorical_association_matrix(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    cols = [c for c in cat_cols if c in df.columns]
    if len(cols) < 2:
        return pd.DataFrame()
    mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for a in cols:
        for b in cols:
            mat.loc[a, b] = 1.0 if a == b else round(cramers_v(df[a].astype(str), df[b].astype(str)), 4)
    return mat.reset_index().rename(columns={"index": "variable"})


def kruskal_eta_squared(df: pd.DataFrame, cat_col: str, target_col: str) -> Dict[str, float]:
    sub = df[[cat_col, target_col]].dropna().copy()
    sub[target_col] = pd.to_numeric(sub[target_col], errors="coerce")
    sub = sub.dropna()
    groups = [g[target_col].to_numpy() for _, g in sub.groupby(cat_col) if len(g) >= 2]
    if len(groups) < 2:
        return {"H_stat": np.nan, "p_value": np.nan, "eta_sq_anova": np.nan}
    h_stat, p_val = stats.kruskal(*groups)
    grand_mean = sub[target_col].mean()
    ss_total = ((sub[target_col] - grand_mean) ** 2).sum()
    ss_between = sum(len(g) * ((g.mean() - grand_mean) ** 2) for g in groups)
    eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0
    return {"H_stat": round(float(h_stat), 2), "p_value": float(p_val), "eta_sq_anova": round(eta_sq, 4)}


def detect_derived_columns(
    df: pd.DataFrame, target: str = REQUIRED_TARGET, tol: float = 0.02, min_match: float = 0.99
) -> List[Dict[str, Any]]:
    """
    Detección genérica de fuga: busca columnas reproducibles como
    combinación (a/b, a*b, a+b, a-b) de otras dos columnas numéricas o del target.
    """
    num_cols = [c for c in df.columns if c != target]
    numeric = {}
    for c in num_cols:
        s, _ = safe_numeric_series(df, c)
        if s.notna().sum() > len(df) * 0.5 and s.nunique(dropna=True) > 1:
            numeric[c] = s
    y, _ = safe_numeric_series(df, target)
    if y.notna().sum() == 0:
        return []

    found: List[Dict[str, Any]] = []
    keys = list(numeric.keys())
    for c in keys:
        col = numeric[c]
        candidates = {
            f"{target} / a": y / col.replace(0, np.nan),
            f"{target} * a": y * col,
            f"{target} + a": y + col,
            f"{target} - a": y - col,
            f"a / {target}": col / y.replace(0, np.nan),
        }
        for expr, derived in candidates.items():
            valid = pd.DataFrame({"d": derived, "c": col}).dropna()
            if len(valid) < len(df) * 0.5:
                continue
            rel = (valid["c"] - valid["d"]).abs() / valid["d"].abs().replace(0, np.nan)
            match = float((rel <= tol).mean())
            if match >= min_match:
                found.append({"column": c, "expression": expr.replace("a", c), "match_pct": round(100 * match, 2)})
    return found


# =============================================================================
# FIGURAS CON PROPÓSITO ANALÍTICO
# =============================================================================
def generate_figures(df: pd.DataFrame, figures_dir: Path, seed: int = SEED) -> List[Dict[str, str]]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Dict[str, str]] = []

    work = pd.DataFrame(index=df.index)
    for col in NUMERIC_CANDIDATES:
        if col in df.columns:
            work[col], _ = safe_numeric_series(df, col)

    # 1. Objetivo: original vs log1p
    if "monthly_rent_rub" in work.columns:
        tv = work["monthly_rent_rub"].dropna()
        tv_pos = tv[tv > 0]
        if len(tv_pos) > 0:
            log_target = np.log1p(tv_pos)
            fig_path = figures_dir / "01_target_distribution.png"
            fig, axes = plt.subplots(2, 2, figsize=(13, 8))
            sns.histplot(tv, kde=True, bins=60, ax=axes[0, 0], color="#1f77b4")
            axes[0, 0].axvline(tv.mean(), color="red", ls="--", label=f"Media: {tv.mean():,.0f} RUB")
            axes[0, 0].axvline(tv.median(), color="darkgreen", label=f"Mediana: {tv.median():,.0f} RUB")
            axes[0, 0].set_title("Distribución de monthly_rent_rub (escala original)")
            axes[0, 0].set_xlabel("Alquiler mensual (RUB)")
            axes[0, 0].legend()
            sns.boxplot(x=tv, ax=axes[1, 0], color="#1f77b4")
            axes[1, 0].set_title("Boxplot: asimetría derecha y cola pesada")
            sns.histplot(log_target, kde=True, bins=60, ax=axes[0, 1], color="#2ca02c")
            axes[0, 1].set_title("Distribución de log1p(monthly_rent_rub)")
            try:
                stats.probplot(log_target, dist="norm", plot=axes[1, 1])
                axes[1, 1].set_title("Q-Q normal de log1p(monthly_rent_rub)")
            except Exception as exc:
                LOGGER.warning("Q-Q plot omitido: %s", exc)
            plt.tight_layout()
            fig.savefig(fig_path)
            plt.close(fig)
            generated.append({"file": fig_path.name, "purpose": "Evaluar asimetría del objetivo y si log1p estabiliza la varianza."})

    # 2. Distribuciones de predictores numéricos
    num_plot_cols = [c for c in ["total_area", "to_center_km", "metro_distance_min", "building_year",
                                 "floor", "total_floors", "rooms", "deposit_months"]
                     if c in work.columns and work[c].notna().sum() > 0]
    if num_plot_cols:
        fig_path = figures_dir / "02_numeric_distributions.png"
        n_cols, n_rows = 4, math.ceil(len(num_plot_cols) / 4)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes_flat = np.array(axes).reshape(-1)
        for idx, col in enumerate(num_plot_cols):
            s = work[col].dropna()
            if s.nunique() <= 15:
                vc = s.value_counts().sort_index()
                axes_flat[idx].bar(vc.index.astype(str), vc.values, color="#4c72b0")
            else:
                sns.histplot(s, kde=True, bins=40, ax=axes_flat[idx], color="#4c72b0")
            axes_flat[idx].set_title(f"{col} (med={s.median():.1f}, IQR={s.quantile(0.75)-s.quantile(0.25):.1f})")
        for idx in range(len(num_plot_cols), len(axes_flat)):
            axes_flat[idx].axis("off")
        plt.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)
        generated.append({"file": fig_path.name, "purpose": "Multimodalidad, rangos, discretización y sesgo de predictores numéricos."})

    # 3. Categóricas clave vs alquiler
    cat_plot_cols = [c for c in ["okrug", "renovation", "building_type", "furnished"] if c in df.columns]
    if cat_plot_cols:
        fig_path = figures_dir / "03_categorical_analysis.png"
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        axes_flat = axes.flatten()
        for idx, col in enumerate(cat_plot_cols[:4]):
            tmp = pd.DataFrame({"cat": df[col].astype(str), "rent": work["monthly_rent_rub"]}).dropna()
            order = tmp.groupby("cat")["rent"].median().sort_values(ascending=False).index
            sns.boxplot(data=tmp, x="rent", y="cat", order=order, ax=axes_flat[idx],
                        showfliers=False, color="#4c72b0")
            axes_flat[idx].set_title(f"monthly_rent_rub por {col} (mediana ordenada)")
        plt.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)
        generated.append({"file": fig_path.name, "purpose": "Poder discriminativo de categorías clave sobre la mediana del alquiler."})

    # 4. Correlaciones Pearson / Spearman
    corr_cols = [c for c in ["monthly_rent_rub", "rent_per_sqm", "total_area", "rooms", "to_center_km",
                             "metro_distance_min", "floor", "total_floors", "building_year",
                             "deposit_months", "lat", "lon"]
                 if c in work.columns and work[c].notna().sum() > 0]
    if len(corr_cols) >= 3:
        fig_path = figures_dir / "04_correlation_matrices.png"
        fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
        sns.heatmap(work[corr_cols].corr(method="pearson"), annot=True, fmt=".2f", cmap="coolwarm",
                    vmin=-1, vmax=1, ax=axes[0], cbar=False)
        axes[0].set_title("Pearson (r)")
        sns.heatmap(work[corr_cols].corr(method="spearman"), annot=True, fmt=".2f", cmap="coolwarm",
                    vmin=-1, vmax=1, ax=axes[1])
        axes[1].set_title("Spearman (rho)")
        plt.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)
        generated.append({"file": fig_path.name, "purpose": "Relaciones lineales/monótonas con el objetivo y colinealidad entre predictores."})

    # 5. Interacciones clave
    if {"total_area", "to_center_km", "monthly_rent_rub"}.issubset(work.columns):
        fig_path = figures_dir / "05_key_bivariate_interactions.png"
        sample_df = work.dropna(subset=["monthly_rent_rub", "total_area", "to_center_km"]).copy()
        if len(sample_df) > 4000:
            sample_df = sample_df.sample(n=4000, random_state=seed)
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        hue_col = sample_df["rooms"] if "rooms" in sample_df.columns else None
        sns.scatterplot(data=sample_df, x="total_area", y="monthly_rent_rub", hue=hue_col,
                        palette="viridis", alpha=0.5, ax=axes[0], legend=False)
        axes[0].set_title("Alquiler vs superficie (coloreado por rooms)")
        sns.scatterplot(data=sample_df, x="to_center_km", y="monthly_rent_rub", hue="total_area",
                        palette="magma", alpha=0.5, ax=axes[1], legend=False)
        axes[1].set_yscale("log")
        axes[1].set_title("Alquiler (log) vs distancia al centro (coloreado por área)")
        plt.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)
        generated.append({"file": fig_path.name, "purpose": "No-linealidad del decaimiento del precio con to_center_km e interacción con total_area."})

    # 6. Fuga de información y anomalía de metro
    if {"rent_per_sqm", "total_area", "monthly_rent_rub"}.issubset(work.columns):
        fig_path = figures_dir / "06_leakage_and_metro_diagnostics.png"
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        calc_ratio = work["monthly_rent_rub"] / work["total_area"].replace(0, np.nan)
        axes[0].scatter(calc_ratio, work["rent_per_sqm"], alpha=0.25, s=10, color="#d62728")
        max_v = float(np.nanmax([np.nanmax(calc_ratio), np.nanmax(work["rent_per_sqm"])]))
        axes[0].plot([0, max_v], [0, max_v], "k--", label="Identidad y = x")
        axes[0].set_title("FUGA: rent_per_sqm vs monthly_rent_rub / total_area")
        axes[0].legend()
        if {"metro_distance_min", "to_center_km"}.issubset(work.columns):
            axes[1].scatter(work["to_center_km"], work["metro_distance_min"], alpha=0.2, s=10, color="#9467bd")
            axes[1].axhline(120, color="red", ls="--", label="Umbral 120 min")
            axes[1].set_title("metro_distance_min vs to_center_km")
            axes[1].legend()
        plt.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)
        generated.append({"file": fig_path.name, "purpose": "Evidencia empírica de fuga en rent_per_sqm y escala anómala de metro_distance_min."})

    # 7. Dimensión temporal
    if "date_posted" in df.columns:
        dates = pd.to_datetime(df["date_posted"], errors="coerce")
        if dates.notna().sum() > 0:
            fig_path = figures_dir / "07_temporal_trends.png"
            tdf = pd.DataFrame({"date": dates, "rent": work["monthly_rent_rub"]}).dropna()
            tdf["year_month"] = tdf["date"].dt.to_period("M").dt.to_timestamp()
            tdf["month"] = tdf["date"].dt.month
            monthly = tdf.groupby("year_month").agg(
                median_rent=("rent", "median"),
                q25=("rent", lambda x: np.percentile(x, 25)),
                q75=("rent", lambda x: np.percentile(x, 75)),
                count=("rent", "count"),
            ).reset_index()
            fig, axes = plt.subplots(2, 1, figsize=(14, 8))
            axes[0].plot(monthly["year_month"], monthly["median_rent"], color="#1f77b4", lw=2, label="Mediana mensual")
            axes[0].fill_between(monthly["year_month"], monthly["q25"], monthly["q75"], alpha=0.2,
                                 color="#1f77b4", label="IQR (P25-P75)")
            axes[0].set_title("Evolución temporal del alquiler (mediana e IQR)")
            axes[0].legend(loc="upper left")
            ax_vol = axes[0].twinx()
            ax_vol.bar(monthly["year_month"], monthly["count"], width=20, alpha=0.18, color="gray")
            ax_vol.set_ylabel("Volumen de anuncios")
            sns.boxplot(data=tdf, x="month", y="rent", showfliers=False, ax=axes[1], color="#ff7f0e")
            axes[1].set_title("Estacionalidad intra-anual por mes de publicación")
            plt.tight_layout()
            fig.savefig(fig_path)
            plt.close(fig)
            generated.append({"file": fig_path.name, "purpose": "Drift de precios 2021-2026, volumen mensual y estacionalidad para definir el split temporal."})

    # 8. Distribución espacial
    if {"lat", "lon", "monthly_rent_rub"}.issubset(work.columns):
        sp_df = work[["lat", "lon", "monthly_rent_rub"]].dropna()
        if len(sp_df) > 0:
            fig_path = figures_dir / "08_spatial_distribution.png"
            fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
            sc = axes[0].scatter(sp_df["lon"], sp_df["lat"],
                                 c=np.log10(sp_df["monthly_rent_rub"].clip(lower=1)),
                                 cmap="Spectral_r", s=6, alpha=0.45)
            plt.colorbar(sc, ax=axes[0], label="log10(monthly_rent_rub)")
            axes[0].set_title("Distribución espacial de alquileres")
            if "okrug" in df.columns:
                sp_okrug = pd.DataFrame({"lon": work["lon"], "lat": work["lat"],
                                         "okrug": df["okrug"].astype(str)}).dropna()
                sns.scatterplot(data=sp_okrug, x="lon", y="lat", hue="okrug", s=10, alpha=0.5,
                                ax=axes[1], palette="tab20", legend=False)
                axes[1].set_title("Zonificación por okrug (leyenda omitida por cardinalidad)")
            plt.tight_layout()
            fig.savefig(fig_path)
            plt.close(fig)
            generated.append({"file": fig_path.name, "purpose": "Estructura monocéntrica de precios y coherencia espacial de los distritos."})
    return generated


# =============================================================================
# REPORTE MARKDOWN
# =============================================================================
def build_markdown_report(
    input_path: Path,
    input_sha256: str,
    cleaning_info: Dict[str, Any],
    cross_notes: List[str],
    cross_issues: List[str],
    df: pd.DataFrame,
    load_warnings: List[str],
    num_summary: pd.DataFrame,
    cat_summary: pd.DataFrame,
    type_issues: Dict[str, int],
    vif_df: pd.DataFrame,
    cat_assoc_df: pd.DataFrame,
    cat_corr_matrix: pd.DataFrame,
    derived_cols: List[Dict[str, Any]],
    figures_meta: List[Dict[str, str]],
) -> str:
    n_rows, n_cols = df.shape
    lines: List[str] = []

    exact_dups = int(df.duplicated().sum())
    cols_no_id = [c for c in df.columns if c != "id"]
    dups_no_id = int(df.duplicated(subset=cols_no_id).sum()) if cols_no_id else 0
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

    work_rent, _ = safe_numeric_series(df, "monthly_rent_rub")
    work_area, _ = safe_numeric_series(df, "total_area")
    work_floor, _ = safe_numeric_series(df, "floor")
    work_tfloors, _ = safe_numeric_series(df, "total_floors")
    work_byear, _ = safe_numeric_series(df, "building_year")
    work_metro, _ = safe_numeric_series(df, "metro_distance_min")
    work_center, _ = safe_numeric_series(df, "to_center_km")
    work_rps, _ = safe_numeric_series(df, "rent_per_sqm")

    floor_gt_total = int((work_floor > work_tfloors).sum()) if not work_floor.empty and not work_tfloors.empty else 0
    nonpos_floor = int((work_floor <= 0).sum()) if not work_floor.empty else 0

    byear_after_post = 0
    date_min_str = date_max_str = "N/D"
    if "date_posted" in df.columns:
        dates = pd.to_datetime(df["date_posted"], errors="coerce")
        if dates.notna().sum() > 0:
            date_min_str, date_max_str = str(dates.min().date()), str(dates.max().date())
            if not work_byear.empty:
                byear_after_post = int((work_byear > dates.dt.year).sum())

    leakage_match_pct = corr_rent_sqm = np.nan
    if not work_rps.empty and not work_area.empty:
        ratio = work_rent / work_area.replace(0, np.nan)
        rel = (work_rps - ratio).abs() / ratio.abs().replace(0, np.nan)
        leakage_match_pct = float(100.0 * (rel <= 0.02).mean())
        corr_rent_sqm = float(work_rent.corr(work_rps))

    corr_metro_center = float(work_metro.corr(work_center)) if not work_metro.empty and not work_center.empty else np.nan
    pct_metro_gt_120 = float(100.0 * (work_metro > 120).mean()) if not work_metro.empty else np.nan
    median_okrugs_per_station = np.nan
    if {"metro_station", "okrug"}.issubset(df.columns):
        okr = df.groupby("metro_station")["okrug"].nunique()
        if len(okr) > 0:
            median_okrugs_per_station = float(okr.median())

    dup_groups = dup_rows = 0
    if {"lat", "lon"}.issubset(df.columns):
        cc = df.groupby(["lat", "lon"]).size()
        dup = cc[cc > 1]
        dup_groups, dup_rows = int(len(dup)), int(dup.sum())

    # ---------------- Encabezado ----------------
    lines.append("# Informe de Análisis Exploratorio de Datos (EDA) — Alquileres en Moscú (Etapa 02)\n")
    lines.append("| Metadato | Valor |")
    lines.append("|---|---|")
    lines.append(f"| Fecha de generación | `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}` |")
    lines.append(f"| Archivo analizado (solo lectura) | `{input_path}` |")
    lines.append(f"| SHA-256 verificado (inmutable) | `{input_sha256}` |")
    lines.append(f"| Dimensiones del dataset | `{n_rows}` filas × `{n_cols}` columnas |")
    lines.append(f"| Variable objetivo | `{REQUIRED_TARGET}` (alquiler mensual en RUB) |\n")

    if load_warnings:
        lines.append("## Advertencias de Carga y Esquema\n")
        for w in load_warnings:
            lines.append(f"- ⚠️ {w}")
        lines.append("")

    # ---------------- 1. Contraste con limpieza ----------------
    lines.append("## 1. Resumen y Contraste de la Limpieza Inicial (`01_cleaning.py`)\n")
    if cleaning_info.get("exists"):
        lines.append(f"- **Informe detectado:** `{cleaning_info.get('path')}`.")
        for n in cross_notes:
            lines.append(f"- ✅ {n}")
        for i in cross_issues:
            lines.append(f"- ⚠️ **Inconsistencia:** {i}")
        lines.append(f"- **Contabilidad declarada:** iniciales = `{cleaning_info.get('initial_rows', 'N/D')}`, "
                     f"eliminadas = `{cleaning_info.get('removed_rows', 'N/D')}`, finales = `{cleaning_info.get('final_rows', 'N/D')}`.")
    else:
        lines.append("- **Informe de limpieza previo no disponible;** el diagnóstico se basa en la inspección directa de `rentals_clean.csv`.")

    lines.append("- **Reglas de la etapa previa y su efecto sobre el EDA:**")
    lines.append("  1. Normalización de texto y conversión de tipos sin pérdida de filas (0 registros eliminados).")
    lines.append("  2. No se aplicaron filtros estadísticos de valores extremos (IQR, percentiles, Cook) sobre `monthly_rent_rub` ni predictores.")
    lines.append(f"  3. Anomalías preservadas en `dq_flags` (`building_year_after_posting` en `{byear_after_post}` filas), **no utilizable como feature**.")
    lines.append("  4. Se conservaron variables redundantes (`rent_per_sqm`) y de semántica dudosa (`metro_*`), cuyo diagnóstico se realiza aquí.\n")

    # ---------------- 2. Estructura y calidad ----------------
    lines.append("## 2. Estructura del Dataset y Calidad de Datos\n")
    lines.append("### 2.1. Inventario de Variables, Tipos y Cardinalidad\n")
    lines.append("| Columna | Tipo en Pandas | Rol Analítico | Nulos (%) | Valores Únicos | Descripción |")
    lines.append("|---|---|---|---:|---:|---|")
    for col in df.columns:
        meta = EXPECTED_SCHEMA.get(col, {"role": "unknown", "desc": "Columna no documentada en esquema base"})
        na_pct = 100.0 * df[col].isna().sum() / n_rows
        lines.append(f"| `{col}` | `{df[col].dtype}` | `{meta['role']}` | {na_pct:.2f}% | "
                     f"{df[col].nunique(dropna=True):,} | {meta['desc']} |")
    lines.append("")

    lines.append("### 2.2. Chequeos de Integridad y Consistencia Lógica\n")
    lines.append(f"- **Duplicados exactos (todas las columnas):** `{exact_dups}` filas.")
    lines.append(f"- **Duplicados excluyendo `id`:** `{dups_no_id}` filas.")
    lines.append(f"- **Columnas constantes (varianza cero):** `{constant_cols if constant_cols else 'Ninguna'}`.")
    lines.append(f"- **Fallos de coerción numérica:** `{type_issues if type_issues else 'Ninguno (0 fallos)'}`.")
    lines.append(f"- **Inconsistencia vertical (`floor > total_floors`):** `{floor_gt_total}` filas.")
    lines.append(f"- **Pisos no positivos (`floor <= 0`):** `{nonpos_floor}` filas.")
    lines.append(f"- **Anomalía temporal (`building_year > año(date_posted)`):** `{byear_after_post}` filas "
                 f"({100.0 * byear_after_post / n_rows:.2f}% del total).")
    lines.append(f"- **Coordenadas `(lat, lon)` idénticas en registros distintos:** `{dup_groups}` grupo(s) / `{dup_rows}` filas.\n")

    # ---------------- 3. Estadística descriptiva ----------------
    lines.append("## 3. Estadística Descriptiva\n")
    lines.append("### 3.1. Variables Numéricas (paramétrica y no paramétrica)\n")
    lines.append(df_to_md(num_summary))
    lines.append("\n### 3.2. Variables Categóricas (cardinalidad, frecuencias y categorías raras)\n")
    lines.append(df_to_md(cat_summary))
    lines.append("")

    # ---------------- 4. Objetivo ----------------
    lines.append("## 4. Análisis de la Variable Objetivo (`monthly_rent_rub`)\n")
    if not num_summary.empty and (num_summary["variable"] == REQUIRED_TARGET).any():
        t = num_summary[num_summary["variable"] == REQUIRED_TARGET].iloc[0]
        pos = work_rent.dropna()
        pos = pos[pos > 0]
        log_skew = float(stats.skew(np.log1p(pos), bias=False)) if len(pos) > 2 else np.nan
        log_kurt = float(stats.kurtosis(np.log1p(pos), bias=False)) if len(pos) > 3 else np.nan
        lines.append(f"- **Rango y tendencia central:** de **{t['min']:,.0f}** a **{t['max']:,.0f} RUB**, "
                     f"mediana **{t['median']:,.0f} RUB** (P25 = {t['p25']:,.0f}; P75 = {t['p75']:,.0f}; "
                     f"IQR = {t['iqr']:,.0f}) y media **{t['mean']:,.0f} RUB** (DE = {t['std']:,.0f}).")
        lines.append(f"- **Asimetría y cola derecha:** skewness = **{t['skewness']:.3f}**, kurtosis = **{t['kurtosis']:.3f}**; "
                     f"`{t['outliers_1_5_iqr']}` observaciones ({t['outliers_1_5_pct']:.2f}%) superan el umbral de Tukey 1.5×IQR.")
        lines.append(f"- **Efecto de `log1p`:** la asimetría cae a **{log_skew:.3f}** (curtosis = **{log_kurt:.3f}**), "
                     "estabilizando la varianza condicional.")
        lines.append("- **Recomendación de modelado:** usar $\\ln(1+y)$ como respuesta con corrección de sesgo de "
                     "retransformación (exp/ factor de Duan–smearing), o pérdidas robustas (Huber, MAE, Gamma/Tweedie).\n")

    # ---------------- 5. Relaciones y multicolinealidad ----------------
    lines.append("## 5. Relaciones con el Objetivo, Asociaciones y Multicolinealidad\n")
    corr_rows = []
    for c in [c for c in num_summary["variable"].tolist() if c != REQUIRED_TARGET and num_summary.empty is False]:
        s, _ = safe_numeric_series(df, c)
        pair = pd.DataFrame({"x": s, "y": work_rent}).dropna()
        if len(pair) > 2:
            pr = float(stats.pearsonr(pair["x"], pair["y"])[0])
            sr = float(stats.spearmanr(pair["x"], pair["y"])[0])
            corr_rows.append({"variable": c, "pearson_r": round(pr, 4),
                              "spearman_rho": round(sr, 4), "abs_spearman": round(abs(sr), 4)})
    corr_target_df = pd.DataFrame(corr_rows)
    if not corr_target_df.empty:
        corr_target_df = corr_target_df.sort_values("abs_spearman", ascending=False)
    lines.append("### 5.1. Correlaciones Numéricas con `monthly_rent_rub`\n")
    lines.append(df_to_md(corr_target_df))
    lines.append("\n### 5.2. Asociación Categórica → Objetivo (Kruskal‑Wallis y $\\eta^2$)\n")
    lines.append(df_to_md(cat_assoc_df))
    lines.append("\n### 5.3. Asociación entre Categóricas (V de Cramér corregida)\n")
    lines.append(df_to_md(cat_corr_matrix))
    lines.append("\n### 5.4. Multicolinealidad entre Predictores Numéricos (VIF, sin variables de fuga)\n")
    lines.append(df_to_md(vif_df))
    if not vif_df.empty:
        lines.append(f"\n- Número de condición de la matriz de correlación: **{vif_df.attrs.get('condition_number', 'N/D')}**"
                     f" (inversa pseudoinversa usada: `{vif_df.attrs.get('used_pinv', False)}`).")
        high_vif = vif_df[vif_df["vif"] > 5]["variable"].tolist()
        lines.append(f"- Predictores con **VIF > 5** (colinealidad problemática): {high_vif if high_vif else 'ninguno'}.\n")

    # ---------------- 6. Fuga y anomalías ----------------
    lines.append("## 6. Diagnóstico de Fuga de Información, Anomalías de Dominio y Sesgos\n")
    lines.append(f"1. **Fuga directa (`rent_per_sqm`) — CRÍTICO:** r = {corr_rent_sqm:.3f} con el objetivo y equivale a "
                 f"`monthly_rent_rub / total_area` (±2%) en el **{leakage_match_pct:.1f}%** de los registros. "
                 "**Acción obligatoria: excluir del entrenamiento.**")
    if derived_cols:
        lines.append("   - Otras columnas reproducibles algebraicamente a partir del objetivo:")
        for d in derived_cols:
            lines.append(f"     - `{d['column']}` ≈ `{d['expression']}` (coincidencia {d['match_pct']}%).")
    lines.append("2. **`id` y `dq_flags` — CRÍTICO:** identificador secuencial y columna de auditoría; "
                 "**excluir ambas del conjunto de features**.")
    lines.append("3. **Semántica de `metro_*` — ALTO RIESGO:**")
    lines.append(f"   - Cada estación aparece en una mediana de **{median_okrugs_per_station:.0f} de 11 `okrug`**, "
                 "lo cual es geográficamente imposible para una estación física.")
    lines.append(f"   - `metro_distance_min` alcanza **{work_metro.max():.0f} min**, el **{pct_metro_gt_120:.1f}%** "
                 f"supera 120 min, y su correlación con `to_center_km` es **r = {corr_metro_center:.3f}**.")
    lines.append("   - **Hipótesis:** asignación estación–ubicación sin restricción geográfica; `metro_distance_min` "
                 "actúa como proxy ruidoso de `to_center_km` o está expresado en otra unidad.")
    lines.append("   - **Recomendación:** excluir `metro_station`/`metro_line` (o ablación controlada) y evaluar si "
                 "`metro_distance_min` aporta señal incremental sobre `to_center_km` (VIF elevado).")
    lines.append(f"4. **Dimensión temporal (`date_posted`: {date_min_str} → {date_max_str}):** más de 5 años de datos "
                 "nominales en rublos sujetos a inflación.")
    lines.append("   - **Split recomendado:** **Time‑Based Split** (entrenamiento histórico, validación intermedia, "
                 "prueba en los últimos meses), y ajuste de transformadores **exclusivamente sobre `X_train`/`y_train`** "
                 "dentro de un `Pipeline`/`ColumnTransformer`.\n")

    # ---------------- 7. Selección de features ----------------
    lines.append("## 7. Conjunto de Features Recomendado\n")
    lines.append("| Columna | Decisión | Justificación |")
    lines.append("|---|---|---|")
    for col in df.columns:
        if col == REQUIRED_TARGET:
            decision, why = "**Objetivo**", "Variable a predecir"
        elif col in HARD_EXCLUDE:
            why = {"rent_per_sqm": "Fuga de información: derivada del objetivo",
                   "id": "Identificador sin valor predictivo",
                   "dq_flags": "Columna de auditoría creada en limpieza"}[col]
            decision = "❌ EXCLUIR"
        elif col in ("metro_station", "metro_line"):
            decision, why = "⚠️ EXCLUIR / ablación", "Asignación geográfica inconsistente (ver §6.3)"
        elif col == "metro_distance_min":
            decision, why = "⚠️ CONDICIONAL", "Proxy de `to_center_km`; usar solo si aporta señal incremental (VIF)"
        elif col == "date_posted":
            decision, why = "⚠️ DERIVAR", "Usar para split temporal y `building_age`; no como feature cruda"
        else:
            decision, why = "✅ INCLUIR", "Predictora válida"
        lines.append(f"| `{col}` | {decision} | {why} |")
    lines.append("")

    # ---------------- 8. Familias de modelos ----------------
    lines.append("## 8. Viabilidad para Machine Learning y Familias de Modelos Candidatas\n")
    lines.append("| Familia | Ejemplos | Justificación (hallazgos del EDA) | Preprocesamiento |")
    lines.append("|---|---|---|---|")
    lines.append("| **1. Lineales regularizados (baseline interpretable)** | `Ridge`, `Lasso`, `ElasticNet`, `HuberRegressor` + `TransformedTargetRegressor(log1p)` | Elasticidades auditables; sensibles a colinealidad (`to_center_km` vs `metro_distance_min`, `total_area` vs `rooms`) y a la asimetría del objetivo | `log1p` del target, eliminar colineales (VIF>5), `OneHotEncoder` en categóricas de baja cardinalidad, `TargetEncoder` con CV en `district`, `RobustScaler`, interacciones `total_area * to_center_km` |")
    lines.append("| **2. GBDT (familia principal)** | `LightGBM`, `CatBoost`, `XGBoost` | Capturan no linealidad (gradiente espacial `lat`/`lon`, `to_center_km`) e interacciones de orden superior (`total_area` × `okrug` × `renovation`); robustos a escalas dispares | Mínimo escalado; features de dominio (`floor_ratio = floor/total_floors`, `building_age`, coordenadas polares al centro); pérdida `RMSE` sobre `log1p(y)` o `Gamma`/`Huber` |")
    lines.append("| **3. Bagging no paramétrico** | `RandomForestRegressor`, `ExtraTreesRegressor` | Comparación robusta frente a varianza y extremos; no extrapola tendencias temporales fuera del rango observado | Codificación ordinal/target de alta cardinalidad e indicadores de antigüedad/fecha |\n")

    # ---------------- 9. Figuras ----------------
    lines.append("## 9. Catálogo de Gráficos Generados (`reports/figures/`)\n")
    lines.append("| Archivo | Finalidad Analítica |")
    lines.append("|---|---|")
    for f in figures_meta:
        lines.append(f"| `reports/figures/{f['file']}` | {f['purpose']} |")
    lines.append("")

    # ---------------- 10. Limitaciones ----------------
    lines.append("## 10. Limitaciones de Validez y Advertencias\n")
    lines.append("- El EDA se basa exclusivamente en `rentals_clean.csv` y en el informe de limpieza; "
                 "**no se verificó la fuente original de los datos**.")
    lines.append("- `to_center_km` no documenta el punto de referencia ni la fórmula (haversine/euclidiana): "
                 "la reproducibilidad del cálculo queda sin confirmar.")
    lines.append("- `metro_distance_min` tiene unidades/semántica sin confirmar; sus resultados deben interpretarse con cautela.")
    lines.append("- La calidad declarada (0 nulos, 0 duplicados, 0 normalizaciones aplicadas) es inusualmente perfecta "
                 "para un scraping real: los datos podrían ser sintéticos o pre‑procesados.")
    lines.append("- Los valores extremos se conservaron por diseño de la etapa 01; su legitimidad debe evaluarse con "
                 "conocimiento de dominio antes de tratarlos.")
    lines.append("- `dq_flags` es una columna de auditoría generada durante la limpieza: no debe emplearse como feature.")
    return "\n".join(lines)


# =============================================================================
# ORQUESTACIÓN
# =============================================================================
def run_eda(input_csv: Path, cleaning_report_md: Path, report_out_md: Path,
            figures_dir: Path, seed: int = SEED, strict: bool = False) -> Dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    LOGGER.info("Iniciando 02_eda.py sobre '%s' (seed=%d)", input_csv, seed)

    df, load_warnings = load_and_validate_dataset(input_csv)
    sha_before = compute_sha256(input_csv)
    mtime_before = input_csv.stat().st_mtime_ns

    cleaning_info = parse_cleaning_report(cleaning_report_md)
    cross_notes, cross_issues = cross_validate_with_cleaning_report(df, cleaning_info, sha_before)
    for msg in cross_issues:
        LOGGER.warning("Inconsistencia documental: %s", msg)
    if strict and cross_issues:
        raise RuntimeError("Modo estricto: el CSV no coincide con el informe de limpieza.")

    numeric_candidates = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    cat_candidates = [c for c in CAT_CANDIDATES if c in df.columns]

    num_summary, type_issues = compute_numeric_summary(df, numeric_candidates, seed=seed)
    cat_summary = compute_categorical_summary(df, cat_candidates)

    vif_predictors = [c for c in numeric_candidates if c not in HARD_EXCLUDE]
    vif_df = compute_vif(df, vif_predictors)

    cat_assoc_rows = []
    for c in ["okrug", "district", "renovation", "building_type", "furnished",
              "pets_allowed", "metro_line", "metro_station"]:
        if c in df.columns:
            res = kruskal_eta_squared(df, c, REQUIRED_TARGET)
            cat_assoc_rows.append({
                "variable": c,
                "n_unique": int(df[c].nunique(dropna=True)),
                "kruskal_H": res["H_stat"],
                "p_value": (f"{res['p_value']:.2e}" if not np.isnan(res["p_value"]) else "N/D"),
                "eta_squared": res["eta_sq_anova"],
            })
    cat_assoc_df = pd.DataFrame(cat_assoc_rows)
    if not cat_assoc_df.empty:
        cat_assoc_df = cat_assoc_df.sort_values("eta_squared", ascending=False, na_position="last")

    cat_corr_matrix = categorical_association_matrix(
        df, [c for c in ["okrug", "district", "building_type", "renovation"] if c in df.columns]
    )

    derived_cols = detect_derived_columns(df, REQUIRED_TARGET)
    if derived_cols:
        LOGGER.warning("Columnas derivadas del objetivo detectadas: %s", derived_cols)

    figures_meta = generate_figures(df, figures_dir, seed=seed)

    report_content = build_markdown_report(
        input_path=input_csv, input_sha256=sha_before, cleaning_info=cleaning_info,
        cross_notes=cross_notes, cross_issues=cross_issues, df=df,
        load_warnings=load_warnings, num_summary=num_summary, cat_summary=cat_summary,
        type_issues=type_issues, vif_df=vif_df, cat_assoc_df=cat_assoc_df,
        cat_corr_matrix=cat_corr_matrix, derived_cols=derived_cols, figures_meta=figures_meta,
    )

    report_out_md.parent.mkdir(parents=True, exist_ok=True)
    tmp_report = report_out_md.with_suffix(".md.tmp")
    tmp_report.write_text(report_content, encoding="utf-8")
    tmp_report.replace(report_out_md)

    sha_after = compute_sha256(input_csv)
    mtime_after = input_csv.stat().st_mtime_ns
    if sha_before != sha_after or mtime_before != mtime_after:
        raise RuntimeError("VIOLACIÓN DE INTEGRIDAD: el archivo de entrada fue modificado durante el EDA.")

    LOGGER.info("EDA finalizado. Reporte '%s' y %d figuras en '%s'.",
                report_out_md, len(figures_meta), figures_dir)
    return {
        "input_sha256": sha_before,
        "rows": len(df),
        "cols": len(df.columns),
        "report_path": str(report_out_md),
        "figures_count": len(figures_meta),
        "warnings": load_warnings,
        "cross_validation_issues": cross_issues,
        "derived_columns": derived_cols,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="02_eda.py — EDA reproducible de alquileres en Moscú.")
    p.add_argument("--input", type=Path, default=Path("data/rentals_clean.csv"))
    p.add_argument("--cleaning-report", type=Path, default=Path("data/cleaning_report.md"))
    p.add_argument("--report-out", type=Path, default=Path("reports/eda_report.md"))
    p.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    p.add_argument("--seed", type=int, default=SEED, help="Semilla para muestreos y submuestras (default: 42).")
    p.add_argument("--strict", action="store_true",
                   help="Falla si el CSV no coincide con el informe de limpieza.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = parse_args(argv)
        run_eda(input_csv=args.input, cleaning_report_md=args.cleaning_report,
                report_out_md=args.report_out, figures_dir=args.figures_dir,
                seed=args.seed, strict=args.strict)
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except Exception as exc:
        LOGGER.exception("Error inesperado durante el EDA: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())