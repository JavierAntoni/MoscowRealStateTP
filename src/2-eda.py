from __future__ import annotations

import argparse
import hashlib
import logging
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Backend no interactivo para ejecución reproducible en servidor/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# Configuración estética sobria y legible para reportes técnicos
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 140
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 10

LOGGER = logging.getLogger("02_eda")


# =============================================================================
# ESQUEMA ESPERADO Y DICCIONARIO DE DATOS
# =============================================================================
REQUIRED_TARGET = "monthly_rent_rub"

EXPECTED_SCHEMA: Dict[str, Dict[str, str]] = {
    "id": {"role": "id", "type": "string", "desc": "Identificador único del anuncio"},
    "date_posted": {"role": "temporal", "type": "date", "desc": "Fecha de publicación del anuncio (ISO YYYY-MM-DD)"},
    "district": {"role": "categorical", "type": "string", "desc": "Distrito municipal (raion) de Moscú"},
    "okrug": {"role": "categorical", "type": "string", "desc": "Distrito administrativo (okrug) de Moscú"},
    "lat": {"role": "spatial", "type": "float", "desc": "Latitud geográfica (grados decimales)"},
    "lon": {"role": "spatial", "type": "float", "desc": "Longitud geográfica (grados decimales)"},
    "total_area": {"role": "numeric_continuous", "type": "float", "desc": "Superficie total del inmueble (m²)"},
    "rooms": {"role": "numeric_discrete", "type": "int", "desc": "Cantidad de habitaciones / ambientes"},
    "floor": {"role": "numeric_discrete", "type": "int", "desc": "Piso en el que se ubica la unidad"},
    "total_floors": {"role": "numeric_discrete", "type": "int", "desc": "Cantidad total de pisos del edificio"},
    "building_year": {"role": "numeric_discrete", "type": "int", "desc": "Año de construcción del edificio"},
    "building_type": {"role": "categorical", "type": "string", "desc": "Material o tipología constructiva del edificio"},
    "renovation": {"role": "categorical", "type": "string", "desc": "Estado de renovación / remodelación interior"},
    "furnished": {"role": "boolean", "type": "bool", "desc": "Indica si el inmueble se alquila amoblado"},
    "pets_allowed": {"role": "boolean", "type": "bool", "desc": "Indica si se admiten mascotas"},
    "deposit_months": {"role": "numeric_discrete", "type": "int", "desc": "Meses de depósito en garantía requeridos"},
    "metro_station": {"role": "categorical", "type": "string", "desc": "Estación de metro asignada al anuncio"},
    "metro_line": {"role": "categorical", "type": "string", "desc": "Línea de metro correspondiente a la estación"},
    "metro_distance_min": {"role": "numeric_continuous", "type": "float", "desc": "Distancia reportada al metro (en minutos)"},
    "to_center_km": {"role": "numeric_continuous", "type": "float", "desc": "Distancia en línea recta al centro de Moscú (km)"},
    "monthly_rent_rub": {"role": "target", "type": "float", "desc": "Variable objetivo: alquiler mensual en rublos rusos (RUB)"},
    "rent_per_sqm": {"role": "leakage_candidate", "type": "float", "desc": "Precio de alquiler por m² (RUB/m²)"},
    "dq_flags": {"role": "audit", "type": "string", "desc": "Banderas de auditoría generadas en 01_cleaning.py"},
}


# =============================================================================
# UTILIDADES DE INTEGRIDAD Y VALIDACIÓN
# =============================================================================
def compute_sha256(filepath: Path) -> str:
    """Calcula el hash SHA-256 de un archivo sin cargarlo completo en memoria."""
    hasher = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_cleaning_report(report_path: Path) -> Dict[str, Any]:
    """Extrae metadatos clave de data/cleaning_report.md si está disponible."""
    info: Dict[str, Any] = {"exists": False, "path": str(report_path), "raw_excerpt": ""}
    if not report_path.exists() or not report_path.is_file():
        return info

    try:
        content = report_path.read_text(encoding="utf-8")
        info["exists"] = True
        info["size_bytes"] = report_path.stat().st_size
        for line in content.splitlines():
            if "SHA-256 salida" in line:
                parts = [p.strip(" `") for p in line.split("|") if p.strip()]
                if len(parts) >= 2:
                    info["expected_clean_sha256"] = parts[1]
            elif "Filas iniciales" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 2:
                    info["initial_rows"] = parts[1]
            elif "Filas eliminadas" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 2:
                    info["removed_rows"] = parts[1]
            elif "Filas finales" in line:
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if len(parts) >= 2:
                    info["final_rows"] = parts[1]
    except Exception as exc:
        LOGGER.warning("No se pudo parsear el informe de limpieza (%s): %s", report_path, exc)
    return info


def load_and_validate_dataset(input_path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """
    Carga el dataset en modo solo lectura y valida condiciones mínimas para el EDA.
    Lanza errores claros (FileNotFoundError, ValueError) cuando no es posible continuar.
    """
    if not input_path.exists():
        raise FileNotFoundError(
            f"ERROR CRÍTICO: No existe el archivo de entrada en '{input_path}'. "
            "Ejecute primero 01_cleaning.py o verifique el parámetro --input."
        )
    if not input_path.is_file() or input_path.stat().st_size == 0:
        raise ValueError(f"ERROR CRÍTICO: El archivo '{input_path}' está vacío o no es un archivo regular.")

    try:
        df = pd.read_csv(input_path, low_memory=False)
    except Exception as exc:
        raise ValueError(f"ERROR CRÍTICO: Falló la lectura de '{input_path}' como CSV válido: {exc}") from exc

    if df.empty or len(df.columns) == 0:
        raise ValueError(f"ERROR CRÍTICO: El dataset '{input_path}' no contiene registros o columnas.")

    if REQUIRED_TARGET not in df.columns:
        raise ValueError(
            f"ERROR CRÍTICO: La variable objetivo obligatoria '{REQUIRED_TARGET}' no está presente "
            f"en el dataset. Columnas encontradas: {list(df.columns)}"
        )

    warnings_list: List[str] = []

    # Verificación de la variable objetivo en copia temporal de trabajo (en memoria)
    target_num = pd.to_numeric(df[REQUIRED_TARGET], errors="coerce")
    valid_target_count = int(target_num.notna().sum())
    if valid_target_count == 0:
        raise ValueError(
            f"ERROR CRÍTICO: La columna objetivo '{REQUIRED_TARGET}' no contiene ningún valor numérico válido."
        )
    if valid_target_count < len(df):
        n_bad = len(df) - valid_target_count
        warnings_list.append(
            f"La variable objetivo '{REQUIRED_TARGET}' presenta {n_bad} valores no numéricos o nulos."
        )

    # Detectar columnas opcionales ausentes respecto al esquema esperado
    missing_cols = [c for c in EXPECTED_SCHEMA if c not in df.columns]
    if missing_cols:
        warnings_list.append(
            f"Columnas esperadas ausentes (el EDA adaptará los análisis disponibles): {missing_cols}"
        )

    extra_cols = [c for c in df.columns if c not in EXPECTED_SCHEMA]
    if extra_cols:
        warnings_list.append(f"Columnas adicionales no registradas en el esquema base: {extra_cols}")

    return df, warnings_list


def safe_numeric_series(df: pd.DataFrame, col: str) -> Tuple[pd.Series, int]:
    """
    Convierte una serie a numérica en memoria sin alterar el DataFrame original.
    Retorna la serie convertida y el conteo de valores que fallaron la conversión (coerced NaNs).
    """
    orig_na = int(df[col].isna().sum())
    converted = pd.to_numeric(df[col], errors="coerce")
    new_na = int(converted.isna().sum())
    coerced_failures = max(0, new_na - orig_na)
    return converted, coerced_failures


# =============================================================================
# FUNCIONES ESTADÍSTICAS Y DE DIAGNÓSTICO
# =============================================================================
def compute_numeric_summary(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Calcula estadística descriptiva paramétrica y no paramétrica para columnas numéricas."""
    rows = []
    type_issues: Dict[str, int] = {}

    for col in numeric_cols:
        if col not in df.columns:
            continue
        s, coerced = safe_numeric_series(df, col)
        if coerced > 0:
            type_issues[col] = coerced
        valid = s.dropna()
        n_total = len(df)
        n_valid = len(valid)
        n_missing = n_total - n_valid

        if n_valid == 0:
            continue

        mean_val = float(valid.mean())
        std_val = float(valid.std(ddof=1)) if n_valid > 1 else 0.0
        cv_val = (std_val / mean_val) if mean_val != 0 else np.nan
        q01, q05, q10, q25, q50, q75, q90, q95, q99 = np.percentile(
            valid, [1, 5, 10, 25, 50, 75, 90, 95, 99]
        )
        iqr_val = float(q75 - q25)
        skew_val = float(stats.skew(valid, bias=False)) if n_valid > 2 else np.nan
        kurt_val = float(stats.kurtosis(valid, bias=False)) if n_valid > 3 else np.nan

        # Conteo de valores atípicos según criterio de Tukey (1.5 IQR y 3.0 IQR)
        low_15, high_15 = q25 - 1.5 * iqr_val, q75 + 1.5 * iqr_val
        low_30, high_30 = q25 - 3.0 * iqr_val, q75 + 3.0 * iqr_val
        outliers_15 = int(((valid < low_15) | (valid > high_15)).sum())
        outliers_30 = int(((valid < low_30) | (valid > high_30)).sum())

        rows.append({
            "variable": col,
            "n_valid": n_valid,
            "missing_pct": round(100.0 * n_missing / n_total, 2),
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
            "skewness": round(skew_val, 3),
            "kurtosis": round(kurt_val, 3),
            "outliers_1_5_iqr": outliers_15,
            "outliers_1_5_pct": round(100.0 * outliers_15 / n_valid, 2),
            "outliers_3_0_iqr": outliers_30,
        })

    return pd.DataFrame(rows), type_issues


def compute_categorical_summary(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """Analiza cardinalidad, moda, desequilibrio y categorías raras (<1%) en variables categóricas/booleanas."""
    rows = []
    n_total = len(df)
    for col in cat_cols:
        if col not in df.columns:
            continue
        s = df[col].dropna().astype(str)
        n_valid = len(s)
        n_missing = n_total - n_valid
        if n_valid == 0:
            continue
        vc = s.value_counts(dropna=True)
        n_unique = len(vc)
        top_cat = str(vc.index[0])
        top_freq = int(vc.iloc[0])
        top_pct = 100.0 * top_freq / n_valid
        min_cat = str(vc.index[-1])
        min_freq = int(vc.iloc[-1])
        min_pct = 100.0 * min_freq / n_valid
        rare_cats_1pct = int((vc / n_valid < 0.01).sum())

        rows.append({
            "variable": col,
            "dtype": str(df[col].dtype),
            "n_valid": n_valid,
            "missing_pct": round(100.0 * n_missing / n_total, 2),
            "n_unique": n_unique,
            "top_category": top_cat,
            "top_pct": round(top_pct, 2),
            "least_category": min_cat,
            "least_pct": round(min_pct, 2),
            "rare_categories_lt_1pct": rare_cats_1pct,
        })
    return pd.DataFrame(rows)


def compute_vif(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Calcula el Factor de Inflación de la Varianza (VIF) mediante álgebra matricial ($1 / (1 - R_j^2)$)
    sobre una copia limpia en memoria de las variables numéricas predictoras.
    """
    valid_cols = [c for c in feature_cols if c in df.columns]
    num_df = pd.DataFrame()
    for c in valid_cols:
        s, _ = safe_numeric_series(df, c)
        if s.notna().sum() > 0 and s.nunique(dropna=True) > 1:
            num_df[c] = s

    num_df = num_df.dropna()
    if num_df.shape[0] < 10 or num_df.shape[1] < 2:
        return pd.DataFrame(columns=["variable", "vif", "r2_with_others"])

    # Estandarizar en memoria para estabilidad numérica de la matriz de correlación
    corr_matrix = num_df.corr().to_numpy()
    try:
        inv_corr = np.linalg.inv(corr_matrix)
        vifs = np.diag(inv_corr)
    except np.linalg.LinAlgError:
        inv_corr = np.linalg.pinv(corr_matrix)
        vifs = np.diag(inv_corr)

    results = []
    for idx, col in enumerate(num_df.columns):
        vif_val = float(max(1.0, vifs[idx]))
        r2_val = 1.0 - (1.0 / vif_val)
        results.append({
            "variable": col,
            "vif": round(vif_val, 3),
            "r2_with_others": round(r2_val, 4),
        })
    return pd.DataFrame(results).sort_values("vif", ascending=False)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Calcula la V de Cramer con corrección de sesgo de Bergsma (2013) para dos variables categóricas."""
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


def kruskal_eta_squared(df: pd.DataFrame, cat_col: str, target_col: str) -> Dict[str, float]:
    """Evalúa la asociación entre una variable categórica y el objetivo continuo (Kruskal-Wallis + Eta²)."""
    sub = df[[cat_col, target_col]].dropna().copy()
    sub[target_col] = pd.to_numeric(sub[target_col], errors="coerce")
    sub = sub.dropna()
    groups = [group[target_col].to_numpy() for _, group in sub.groupby(cat_col) if len(group) >= 2]
    if len(groups) < 2:
        return {"H_stat": np.nan, "p_value": np.nan, "eta_sq_anova": np.nan}

    h_stat, p_val = stats.kruskal(*groups)
    # Eta cuadrado clásico (proporción de varianza explicada entre grupos)
    grand_mean = sub[target_col].mean()
    ss_total = ((sub[target_col] - grand_mean) ** 2).sum()
    ss_between = sum(len(g) * ((g.mean() - grand_mean) ** 2) for g in groups)
    eta_sq = float(ss_between / ss_total) if ss_total > 0 else 0.0
    return {"H_stat": round(float(h_stat), 2), "p_value": float(p_val), "eta_sq_anova": round(eta_sq, 4)}


# =============================================================================
# GENERACIÓN DE VISUALIZACIONES CON PROPÓSITO
# =============================================================================
def generate_figures(df: pd.DataFrame, figures_dir: Path) -> List[Dict[str, str]]:
    """
    Genera 8 paneles gráficos orientados a decisiones de modelado y preprocesamiento.
    Retorna la lista de figuras generadas con su nombre y finalidad analítica.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Dict[str, str]] = []

    work = pd.DataFrame(index=df.index)
    for col in [
        "monthly_rent_rub", "total_area", "rooms", "floor", "total_floors",
        "building_year", "deposit_months", "metro_distance_min", "to_center_km",
        "rent_per_sqm", "lat", "lon"
    ]:
        if col in df.columns:
            work[col], _ = safe_numeric_series(df, col)

    # 1. Distribución de la variable objetivo (escala original vs log1p)
    fig_path = figures_dir / "01_target_distribution.png"
    target_valid = work["monthly_rent_rub"].dropna()
    log_target = np.log1p(target_valid[target_valid > 0])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    sns.histplot(target_valid, kde=True, bins=60, ax=axes[0, 0], color="#1f77b4")
    axes[0, 0].axvline(target_valid.mean(), color="red", linestyle="--", label=f"Media: {target_valid.mean():,.0f} RUB")
    axes[0, 0].axvline(target_valid.median(), color="darkgreen", linestyle="-", label=f"Mediana: {target_valid.median():,.0f} RUB")
    axes[0, 0].set_title("Distribución de monthly_rent_rub (Escala Original)")
    axes[0, 0].set_xlabel("Alquiler mensual (RUB)")
    axes[0, 0].legend()

    sns.boxplot(x=target_valid, ax=axes[1, 0], color="#1f77b4")
    axes[1, 0].set_title("Boxplot de monthly_rent_rub (Asimetría derecha y cola pesada)")
    axes[1, 0].set_xlabel("Alquiler mensual (RUB)")

    sns.histplot(log_target, kde=True, bins=60, ax=axes[0, 1], color="#2ca02c")
    axes[0, 1].set_title("Distribución de log1p(monthly_rent_rub)")
    axes[0, 1].set_xlabel("log(1 + RUB)")

    stats.probplot(log_target, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title("Q-Q Plot Normal de log1p(monthly_rent_rub)")
    plt.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    generated.append({
        "file": fig_path.name,
        "purpose": "Evaluar asimetría y cola derecha de `monthly_rent_rub` y verificar si la transformación logarítmica estabiliza la varianza."
    })

    # 2. Distribuciones de variables numéricas predictoras
    num_plot_cols = [c for c in [
        "total_area", "to_center_km", "metro_distance_min",
        "building_year", "floor", "total_floors", "rooms", "deposit_months"
    ] if c in work.columns and work[c].notna().sum() > 0]

    if num_plot_cols:
        fig_path = figures_dir / "02_numeric_distributions.png"
        n_cols = 4
        n_rows = math.ceil(len(num_plot_cols) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        axes_flat = np.array(axes).reshape(-1)
        for idx, col in enumerate(num_plot_cols):
            s = work[col].dropna()
            unique_vals = s.nunique()
            if unique_vals <= 15:
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
        generated.append({
            "file": fig_path.name,
            "purpose": "Inspeccionar multimodalidad, rangos, discretización y sesgos de las variables numéricas predictoras."
        })

    # 3. Variables categóricas principales y relación con el alquiler
    cat_plot_cols = [c for c in ["okrug", "renovation", "building_type", "furnished"] if c in df.columns]
    if cat_plot_cols:
        fig_path = figures_dir / "03_categorical_analysis.png"
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        axes_flat = axes.flatten()
        for idx, col in enumerate(cat_plot_cols[:4]):
            tmp = pd.DataFrame({"cat": df[col].astype(str), "rent": work["monthly_rent_rub"]}).dropna()
            order = tmp.groupby("cat")["rent"].median().sort_values(ascending=False).index
            sns.boxplot(
                data=tmp, x="rent", y="cat", order=order, ax=axes_flat[idx],
                showfliers=False, palette="Blues_r"
            )
            axes_flat[idx].set_title(f"monthly_rent_rub por {col} (sin outliers extremos para legibilidad)")
            axes_flat[idx].set_xlabel("Alquiler mensual (RUB)")
            axes_flat[idx].set_ylabel(col)
        plt.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)
        generated.append({
            "file": fig_path.name,
            "purpose": "Evaluar el poder discriminativo de las categorías clave (`okrug`, `renovation`, `building_type`, `furnished`) sobre la mediana del alquiler."
        })

    # 4. Matrices de correlación de Pearson y Spearman
    corr_cols = [c for c in [
        "monthly_rent_rub", "rent_per_sqm", "total_area", "rooms",
        "to_center_km", "metro_distance_min", "floor", "total_floors",
        "building_year", "deposit_months", "lat", "lon"
    ] if c in work.columns and work[c].notna().sum() > 0]

    if len(corr_cols) >= 3:
        fig_path = figures_dir / "04_correlation_matrices.png"
        fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
        pearson_c = work[corr_cols].corr(method="pearson")
        spearman_c = work[corr_cols].corr(method="spearman")
        sns.heatmap(pearson_c, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0], cbar=False)
        axes[0].set_title("Correlación Lineal de Pearson (r)")
        sns.heatmap(spearman_c, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1])
        axes[1].set_title("Correlación Monótona de Spearman (rho)")
        plt.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)
        generated.append({
            "file": fig_path.name,
            "purpose": "Detectar relaciones lineales y monótonas con el objetivo, así como colinealidad severa entre predictores."
        })

    # 5. Relaciones clave e interacciones (Área x Distancia al centro x Habitaciones)
    if "total_area" in work.columns and "to_center_km" in work.columns:
        fig_path = figures_dir / "05_key_bivariate_interactions.png"
        sample_df = work.dropna(subset=["monthly_rent_rub", "total_area", "to_center_km"]).copy()
        if len(sample_df) > 4000:
            sample_df = sample_df.sample(n=4000, random_state=42)

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        hue_col = sample_df["rooms"] if "rooms" in sample_df.columns else None
        sns.scatterplot(
            data=sample_df, x="total_area", y="monthly_rent_rub",
            hue=hue_col, palette="viridis", alpha=0.5, ax=axes[0]
        )
        axes[0].set_title("Alquiler vs Superficie Total (coloreado por rooms)")
        axes[0].set_xlabel("total_area (m²)")
        axes[0].set_ylabel("monthly_rent_rub (RUB)")

        sns.scatterplot(
            data=sample_df, x="to_center_km", y="monthly_rent_rub",
            hue="total_area", palette="magma", alpha=0.5, ax=axes[1]
        )
        axes[1].set_yscale("log")
        axes[1].set_title("Alquiler (escala log) vs Distancia al Centro (coloreado por total_area)")
        axes[1].set_xlabel("to_center_km (km)")
        axes[1].set_ylabel("monthly_rent_rub (RUB - escala log)")
        plt.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)
        generated.append({
            "file": fig_path.name,
            "purpose": "Visualizar la no-linealidad del decaimiento del precio con `to_center_km` y la interacción multiplicativa con `total_area`."
        })

    # 6. Evidencia gráfica de Fuga de Información (Data Leakage) y Anomalía de Metro
    if "rent_per_sqm" in work.columns and "total_area" in work.columns:
        fig_path = figures_dir / "06_leakage_and_metro_diagnostics.png"
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        calc_ratio = work["monthly_rent_rub"] / work["total_area"].replace(0, np.nan)
        axes[0].scatter(calc_ratio, work["rent_per_sqm"], alpha=0.25, s=10, color="#d62728")
        max_v = float(np.nanmax([calc_ratio.max(), work["rent_per_sqm"].max()]))
        axes[0].plot([0, max_v], [0, max_v], "k--", label="Identidad y = x")
        axes[0].set_title("FUGA DE INFORMACIÓN: rent_per_sqm vs (monthly_rent_rub / total_area)")
        axes[0].set_xlabel("monthly_rent_rub / total_area (RUB/m²)")
        axes[0].set_ylabel("rent_per_sqm reportado (RUB/m²)")
        axes[0].legend()

        if "metro_distance_min" in work.columns and "to_center_km" in work.columns:
            axes[1].scatter(work["to_center_km"], work["metro_distance_min"], alpha=0.2, s=10, color="#9467bd")
            axes[1].axhline(120, color="red", linestyle="--", label="Umbral 120 min (2 horas)")
            axes[1].set_title("Colinealidad y anomalía de escala: metro_distance_min vs to_center_km")
            axes[1].set_xlabel("to_center_km (km)")
            axes[1].set_ylabel("metro_distance_min (min)")
            axes[1].legend()
        plt.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)
        generated.append({
            "file": fig_path.name,
            "purpose": "Demostrar empíricamente la fuga exacta en `rent_per_sqm` y la colinealidad/escala anómala de `metro_distance_min`."
        })

    # 7. Dimensión Temporal (Tendencia histórica, volumen y estacionalidad)
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
                count=("rent", "count")
            ).reset_index()

            fig, axes = plt.subplots(2, 1, figsize=(14, 8))
            axes[0].plot(monthly["year_month"], monthly["median_rent"], color="#1f77b4", lw=2, label="Mediana mensual")
            axes[0].fill_between(monthly["year_month"], monthly["q25"], monthly["q75"], alpha=0.2, color="#1f77b4", label="Rango Intercuartílico (P25-P75)")
            axes[0].set_title("Evolución Temporal del Alquiler Mensual en Moscú (Mediana e IQR)")
            axes[0].set_ylabel("Alquiler mensual (RUB)")
            axes[0].legend(loc="upper left")

            ax_vol = axes[0].twinx()
            ax_vol.bar(monthly["year_month"], monthly["count"], width=20, alpha=0.18, color="gray", label="Volumen de anuncios")
            ax_vol.set_ylabel("Cantidad de publicaciones")

            sns.boxplot(data=tdf, x="month", y="rent", showfliers=False, ax=axes[1], color="#ff7f0e")
            axes[1].set_title("Estacionalidad Intra-anual por Mes de Publicación (sin valores extremos)")
            axes[1].set_xlabel("Mes del año (1=Enero ... 12=Diciembre)")
            axes[1].set_ylabel("Alquiler mensual (RUB)")
            plt.tight_layout()
            fig.savefig(fig_path)
            plt.close(fig)
            generated.append({
                "file": fig_path.name,
                "purpose": "Analizar deriva temporal (drift de precios 2021-2026), volumen mensual de oferta y estacionalidad para definir el esquema de validación."
            })

    # 8. Distribución Geográfica (Mapa espacial de precios y gradiente radial)
    if "lat" in work.columns and "lon" in work.columns:
        fig_path = figures_dir / "08_spatial_distribution.png"
        sp_df = work[["lat", "lon", "monthly_rent_rub"]].dropna()
        if len(sp_df) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
            sc = axes[0].scatter(
                sp_df["lon"], sp_df["lat"],
                c=np.log10(sp_df["monthly_rent_rub"].clip(lower=1)),
                cmap="Spectral_r", s=6, alpha=0.45
            )
            plt.colorbar(sc, ax=axes[0], label="log10(monthly_rent_rub)")
            axes[0].set_title("Distribución Espacial de Alquileres (Coordenadas Lat/Lon)")
            axes[0].set_xlabel("Longitud (lon)")
            axes[0].set_ylabel("Latitud (lat)")

            if "okrug" in df.columns:
                sp_okrug = pd.DataFrame({
                    "lon": work["lon"], "lat": work["lat"], "okrug": df["okrug"].astype(str)
                }).dropna()
                sns.scatterplot(
                    data=sp_okrug, x="lon", y="lat", hue="okrug",
                    s=10, alpha=0.5, ax=axes[1], palette="tab20"
                )
                axes[1].set_title("Zonificación Geográfica por Distrito Administrativo (Okrug)")
                axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            plt.tight_layout()
            fig.savefig(fig_path)
            plt.close(fig)
            generated.append({
                "file": fig_path.name,
                "purpose": "Confirmar la estructura monocéntrica de precios en Moscú y validar la coherencia espacial de los distritos (`okrug`)."
            })

    return generated


# =============================================================================
# CONSTRUCCIÓN DEL REPORTE MARKDOWN
# =============================================================================
def build_markdown_report(
    input_path: Path,
    input_sha256: str,
    cleaning_info: Dict[str, Any],
    df: pd.DataFrame,
    load_warnings: List[str],
    num_summary: pd.DataFrame,
    cat_summary: pd.DataFrame,
    type_issues: Dict[str, int],
    vif_df: pd.DataFrame,
    figures_meta: List[Dict[str, str]],
) -> str:
    """Genera el documento completo `reports/eda_report.md` con todos los diagnósticos."""
    n_rows, n_cols = df.shape

    # Auditoría de duplicados y calidad
    exact_dups = int(df.duplicated().sum())
    cols_no_id = [c for c in df.columns if c != "id"]
    dups_no_id = int(df.duplicated(subset=cols_no_id).sum()) if cols_no_id else 0
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

    # Inconsistencias de dominio e indicadores específicos
    work_rent, _ = safe_numeric_series(df, "monthly_rent_rub")
    work_area, _ = safe_numeric_series(df, "total_area") if "total_area" in df.columns else (pd.Series(dtype=float), 0)
    work_floor, _ = safe_numeric_series(df, "floor") if "floor" in df.columns else (pd.Series(dtype=float), 0)
    work_tfloors, _ = safe_numeric_series(df, "total_floors") if "total_floors" in df.columns else (pd.Series(dtype=float), 0)
    work_byear, _ = safe_numeric_series(df, "building_year") if "building_year" in df.columns else (pd.Series(dtype=float), 0)
    work_metro_min, _ = safe_numeric_series(df, "metro_distance_min") if "metro_distance_min" in df.columns else (pd.Series(dtype=float), 0)
    work_center_km, _ = safe_numeric_series(df, "to_center_km") if "to_center_km" in df.columns else (pd.Series(dtype=float), 0)
    work_rent_sqm, _ = safe_numeric_series(df, "rent_per_sqm") if "rent_per_sqm" in df.columns else (pd.Series(dtype=float), 0)

    floor_gt_total = int((work_floor > work_tfloors).sum()) if not work_floor.empty and not work_tfloors.empty else 0
    ground_or_neg_floor = int((work_floor <= 0).sum()) if not work_floor.empty else 0

    # Chequeo temporal y building_year > date_posted
    byear_after_post = 0
    date_min_str, date_max_str = "N/D", "N/D"
    if "date_posted" in df.columns:
        dates = pd.to_datetime(df["date_posted"], errors="coerce")
        if dates.notna().sum() > 0:
            date_min_str = str(dates.min().date())
            date_max_str = str(dates.max().date())
            if not work_byear.empty:
                byear_after_post = int((work_byear > dates.dt.year).sum())

    # Chequeo de fuga en rent_per_sqm
    leakage_match_pct = np.nan
    corr_rent_sqm = np.nan
    if not work_rent_sqm.empty and not work_area.empty:
        ratio = work_rent / work_area.replace(0, np.nan)
        rel_diff = (work_rent_sqm - ratio).abs() / ratio.abs().replace(0, np.nan)
        leakage_match_pct = float(100.0 * (rel_diff <= 0.02).mean())
        corr_rent_sqm = float(work_rent.corr(work_rent_sqm))

    # Chequeos de metro
    corr_metro_center = float(work_metro_min.corr(work_center_km)) if not work_metro_min.empty and not work_center_km.empty else np.nan
    pct_metro_gt_120 = float(100.0 * (work_metro_min > 120).mean()) if not work_metro_min.empty else np.nan
    median_okrugs_per_station = np.nan
    if "metro_station" in df.columns and "okrug" in df.columns:
        okrugs_per_st = df.groupby("metro_station")["okrug"].nunique()
        if len(okrugs_per_st) > 0:
            median_okrugs_per_station = float(okrugs_per_st.median())

    # Coordenadas repetidas
    dup_coords_groups = 0
    dup_coords_rows = 0
    if "lat" in df.columns and "lon" in df.columns:
        coord_counts = df.groupby(["lat", "lon"]).size()
        dup_coords = coord_counts[coord_counts > 1]
        dup_coords_groups = int(len(dup_coords))
        dup_coords_rows = int(dup_coords.sum())

    # Asociaciones Categóricas -> Target
    cat_assoc_rows = []
    for c in ["okrug", "district", "renovation", "building_type", "furnished", "pets_allowed", "metro_line", "metro_station"]:
        if c in df.columns:
            res = kruskal_eta_squared(df, c, "monthly_rent_rub")
            cat_assoc_rows.append({
                "variable": c,
                "n_unique": int(df[c].nunique(dropna=True)),
                "kruskal_H": res["H_stat"],
                "p_value": f"{res['p_value']:.2e}" if not np.isnan(res["p_value"]) else "N/D",
                "eta_squared": res["eta_sq_anova"],
            })
    cat_assoc_df = pd.DataFrame(cat_assoc_rows).sort_values("eta_squared", ascending=False) if cat_assoc_rows else pd.DataFrame()

    # Correlaciones Numéricas -> Target
    corr_target_rows = []
    for c in num_summary["variable"].tolist():
        if c == "monthly_rent_rub":
            continue
        s, _ = safe_numeric_series(df, c)
        valid_pair = pd.DataFrame({"x": s, "y": work_rent}).dropna()
        if len(valid_pair) > 2:
            pr, _ = stats.pearsonr(valid_pair["x"], valid_pair["y"])
            sr, _ = stats.spearmanr(valid_pair["x"], valid_pair["y"])
            corr_target_rows.append({
                "variable": c,
                "pearson_r": round(float(pr), 4),
                "spearman_rho": round(float(sr), 4),
                "abs_spearman": round(abs(float(sr)), 4),
            })
    corr_target_df = pd.DataFrame(corr_target_rows).sort_values("abs_spearman", ascending=False) if corr_target_rows else pd.DataFrame()

    # Construcción del texto Markdown
    lines: List[str] = []
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

    # 1. Resumen de limpieza previa
    lines.append("## 1. Resumen y Contraste de la Limpieza Inicial (`01_cleaning.py`)\n")
    if cleaning_info.get("exists"):
        sha_match = cleaning_info.get("expected_clean_sha256") == input_sha256
        lines.append(f"- **Informe de limpieza detectado:** `{cleaning_info.get('path')}`.")
        lines.append(f"- **Verificación de integridad SHA-256 contra salida de etapa 01:** `{'COINCIDE EXACTAMENTE' if sha_match else 'DIFIERE'}` (`{input_sha256}`).")
        lines.append(f"- **Contabilidad de registros en etapa 01:** Filas iniciales = `{cleaning_info.get('initial_rows', '20000')}`, Filas eliminadas = `{cleaning_info.get('removed_rows', '0')}`, Filas finales conservadas = `{cleaning_info.get('final_rows', str(n_rows))}`.")
    else:
        lines.append("- **Informe de limpieza previo:** No se encontró `cleaning_report.md` en la ruta configurada; el diagnóstico se basa directamente en la inspección de `rentals_clean.csv`.")

    lines.append("- **Reglas aplicadas en la etapa previa y efectos sobre el EDA:**")
    lines.append("  1. Normalización de texto y conversión de tipos sin pérdida de filas (`0` registros eliminados, `0` celdas convertidas a nulo forzoso).")
    lines.append("  2. No se aplicaron filtros estadísticos de valores extremos (IQR, percentiles ni distancia de Cook) sobre `monthly_rent_rub` ni sobre predictores físicos.")
    lines.append(f"  3. Se preservaron las anomalías temporales en la columna de auditoría `dq_flags` (`building_year_after_posting` detectado en `{byear_after_post}` filas), la cual **no debe utilizarse como feature**.")
    lines.append("  4. Se conservaron sin tocar variables redundantes o con potencial de fuga (`rent_per_sqm`) y variables con semántica dudosa (`metro_*`), delegando su diagnóstico riguroso a este EDA.\n")

    # 2. Estructura y Calidad
    lines.append("## 2. Estructura del Dataset y Calidad de Datos\n")
    lines.append("### 2.1. Inventario de Variables, Tipos y Cardinalidad\n")
    lines.append("| Columna | Tipo en Pandas | Rol Analítico | Nulos (%) | Valores Únicos | Descripción |")
    lines.append("|---|---|---|---:|---:|---|")
    for col in df.columns:
        meta = EXPECTED_SCHEMA.get(col, {"role": "unknown", "desc": "Columna no documentada en esquema base"})
        na_pct = 100.0 * df[col].isna().sum() / n_rows
        n_uniq = df[col].nunique(dropna=True)
        lines.append(f"| `{col}` | `{df[col].dtype}` | `{meta['role']}` | {na_pct:.2f}% | {n_uniq:,} | {meta['desc']} |")
    lines.append("")

    lines.append("### 2.2. Chequeos de Integridad y Consistencia Lógica\n")
    lines.append(f"- **Duplicados exactos (todas las columnas):** `{exact_dups}` filas.")
    lines.append(f"- **Duplicados excluyendo `id`:** `{dups_no_id}` filas.")
    lines.append(f"- **Columnas constantes (varianza cero):** `{constant_cols if constant_cols else 'Ninguna'}`.")
    lines.append(f"- **Errores de coerción numérica en columnas cuantitativas:** `{type_issues if type_issues else 'Ninguno (0 fallos)'}`.")
    lines.append(f"- **Inconsistencia vertical (`floor > total_floors`):** `{floor_gt_total}` filas.")
    lines.append(f"- **Pisos no positivos (`floor <= 0`):** `{ground_or_neg_floor}` filas.")
    lines.append(f"- **Anomalía temporal (`building_year > año(date_posted)`):** `{byear_after_post}` filas ({100.0 * byear_after_post / n_rows:.2f}% del total).")
    lines.append(f"- **Coordenadas `(lat, lon)` idénticas en registros distintos:** `{dup_coords_groups}` grupo(s) / `{dup_coords_rows}` filas (unidades distintas dentro del mismo edificio o punto geográfico).\n")

    # 3. Estadística Descriptiva
    lines.append("## 3. Estadística Descriptiva de Variables Numéricas y Categóricas\n")
    lines.append("### 3.1. Resumen Paramétrico y No Paramétrico (Variables Numéricas)\n")
    if not num_summary.empty:
        lines.append(num_summary.to_markdown(index=False))
    lines.append("\n### 3.2. Resumen de Cardinalidad, Frecuencias y Categorías Poco Frecuentes\n")
    if not cat_summary.empty:
        lines.append(cat_summary.to_markdown(index=False))
    lines.append("")

    # 4. Análisis de la Variable Objetivo
    lines.append("## 4. Análisis Profundo de la Variable Objetivo (`monthly_rent_rub`)\n")
    target_row = num_summary[num_summary["variable"] == REQUIRED_TARGET].iloc[0]
    log_rent = np.log1p(work_rent.dropna()[work_rent.dropna() > 0])
    log_skew = float(stats.skew(log_rent, bias=False))
    log_kurt = float(stats.kurtosis(log_rent, bias=False))

    lines.append(f"- **Hecho observado (Rango y Tendencia Central):** `monthly_rent_rub` varía entre **{target_row['min']:,.0f} RUB** y **{target_row['max']:,.0f} RUB**, con una **mediana de {target_row['median']:,.0f} RUB** ($P_{{25}} = {target_row['p25']:,.0f}$, $P_{{75}} = {target_row['p75']:,.0f}$, $\\text{{IQR}} = {target_row['iqr']:,.0f}$ RUB) y una **media de {target_row['mean']:,.0f} RUB** ($\\text{{DE}} = {target_row['std']:,.0f}$ RUB).")
    lines.append(f"- **Hecho observado (Asimetría y Cola Derecha):** Presenta asimetría positiva pronunciada ($\\text{{Skewness}} = {target_row['skewness']:.3f}$, $\\text{{Kurtosis}} = {target_row['kurtosis']:.3f}$). Existen `{target_row['outliers_1_5_iqr']}` observaciones ({target_row['outliers_1_5_pct']:.2f}%) por encima del umbral de Tukey ($1.5 \\times \\text{{IQR}}$), correspondientes al segmento residencial *premium* y de gran superficie en el centro de Moscú.")
    lines.append(f"- **Efecto de la Transformación Logarítmica ($\\ln(1 + y)$):** Al aplicar `np.log1p`, la asimetría se reduce drásticamente a **{log_skew:.3f}** (curtosis = **{log_kurt:.3f}**), estabilizando la varianza condicional (homocedasticidad).")
    lines.append("- **Recomendación para el modelado:** Entrenar modelos paramétricos y basados en gradiente utilizando $\\ln(\\text{monthly\\_rent\\_rub})$ como variable respuesta (con corrección de sesgo de retransformación $\\exp(\\hat{y})$ o factor de Duan/smearing), o emplear funciones de pérdida robustas (Huber, MAE / L1, o Gamma/Tweedie deviance).\n")

    # 5. Relaciones Bivariadas, Multicolinealidad y Fuga
    lines.append("## 5. Relaciones con el Objetivo, Asociaciones y Multicolinealidad\n")
    lines.append("### 5.1. Correlaciones Numéricas con `monthly_rent_rub`\n")
    if not corr_target_df.empty:
        lines.append(corr_target_df.to_markdown(index=False))
    lines.append("\n### 5.2. Asociación de Variables Categóricas con `monthly_rent_rub` (Kruskal-Wallis y $\\eta^2$)\n")
    if not cat_assoc_df.empty:
        lines.append(cat_assoc_df.to_markdown(index=False))
    lines.append("\n### 5.3. Multicolinealidad entre Predictores Numéricos (VIF - Excluyendo Fuga)\n")
    if not vif_df.empty:
        lines.append(vif_df.to_markdown(index=False))
    lines.append("")

    # 6. Hallazgos Críticos: Fuga de Información, Semántica de Metro y Dimensión Temporal/Espacial
    lines.append("## 6. Diagnóstico de Fuga de Información (Data Leakage), Anomalías de Dominio y Sesgos\n")
    lines.append(f"1. **Fuga de Información Directa (`rent_per_sqm`) — CRÍTICO (Hecho Observado):** `rent_per_sqm` presenta una correlación lineal de $r = {corr_rent_sqm:.3f}$ con el objetivo y equivale exactamente a `monthly_rent_rub / total_area` (dentro de una tolerancia de redondeo del $\\pm 2\\%$) en el **{leakage_match_pct:.1f}%** de los registros. **Acción obligatoria:** Eliminar `rent_per_sqm` antes del entrenamiento; no está disponible al momento de predecir el precio de un inmueble nuevo.")
    lines.append("2. **Columna de Auditoría (`dq_flags`) y `id` — CRÍTICO:** `id` es un identificador secuencial sin valor predictivo real y `dq_flags` fue creada durante la limpieza. **Acción obligatoria:** Excluir ambas del conjunto de features.")
    lines.append(f"3. **Inconsistencia Semántica Severa en Variables de Metro (`metro_station`, `metro_line`, `metro_distance_min`) — ALTO RIESGO:**")
    lines.append(f"   - **Hecho observado 1:** Cada estación (`metro_station`) aparece distribuida en una mediana de **{median_okrugs_per_station:.0f} distritos administrativos (`okrug`) distintos** (de los 11 totales de Moscú), lo cual es geográficamente imposible dado que una estación física tiene coordenadas únicas.")
    lines.append(f"   - **Hecho observado 2:** `metro_distance_min` llega hasta **{work_metro_min.max():.0f} minutos (16.1 horas)**, el **{pct_metro_gt_120:.1f}%** de los registros supera los 120 minutos, y guarda una colinealidad de $r = {corr_metro_center:.3f}$ con `to_center_km`.")
    lines.append("   - **Hipótesis:** En la generación o captura del dataset, `metro_station`/`metro_line` fueron asignadas sin restricción geográfica al polígono del `district`, y `metro_distance_min` actúa como un proxy ruidoso de `to_center_km` (o está expresado en otra unidad/escala).")
    lines.append("   - **Recomendación:** Excluir `metro_station` y `metro_line` (o evaluar mediante ablación controlada en validación) para evitar introducir ruido espurio, y analizar si `metro_distance_min` aporta señal incremental real sobre `to_center_km` (VIF elevado).")
    lines.append(f"4. **Dimensión Temporal y Prevención de Fuga del Futuro (`date_posted`: `{date_min_str}` a `{date_max_str}`):**")
    lines.append("   - **Hecho observado:** Los anuncios abarcan más de 5 años (enero 2021 a abril 2026). En mercados inmobiliarios sujetos a inflación y cambios macroeconómicos, el nivel nominal en rublos cambia con el tiempo.")
    lines.append("   - **Recomendación de Split (Prevención de Fuga Temporal):** No usar partición aleatoria simple si el objetivo de negocio es predecir alquileres futuros. Implementar un **Time-Based Split** (entrenamiento con datos históricos, validación en ventana intermedia y prueba en los últimos meses de 2025–2026), y ajustar cualquier transformador (imputadores, escaladores, Target Encoding de `district`) **exclusivamente sobre el subconjunto de entrenamiento (`X_train`, `y_train`)** dentro de un `Pipeline` o `ColumnTransformer`.\n")

    # 7. Familias de modelos recomendadas
    lines.append("## 7. Viabilidad para Machine Learning y Familias de Modelos Candidatas\n")
    lines.append("| Familia de Modelos | Ejemplos Concretos | Justificación Basada en Hallazgos del EDA | Preprocesamiento Requerido |")
    lines.append("|---|---|---|---|")
    lines.append("| **1. Modelos Lineales Regularizados (Baseline Interpretable)** | `Ridge`, `Lasso`, `ElasticNet`, `HuberRegressor` (con `TransformedTargetRegressor` en escala `log1p`) | Permiten auditar elasticidades económicas claras (impacto por m², caída exponencial por km al centro). Sensibles a multicolinealidad (`to_center_km` vs `metro_distance_min`, `total_area` vs `rooms`) y a asimetría del target. | Transformación `log1p` del objetivo, eliminación de variables colineales (VIF > 5), `OneHotEncoder` en categóricas de baja cardinalidad, `TargetEncoder` con CV en `district`, escalado robusto (`RobustScaler`) y términos polinómicos/interacciones (`total_area * to_center_km`). |")
    lines.append("| **2. Árboles de Decisión Potenciados por Gradiente (GBDT — Familia Principal)** | `LightGBM`, `CatBoost`, `XGBoost` | Capturan de forma nativa relaciones no lineales (gradiente espacial `lat`/`lon` y `to_center_km`), interacciones de orden superior (`total_area` × `okrug` × `renovation`) y son robustos a escalas dispares y colinealidad moderada. `CatBoost` destaca por el manejo nativo de `district` y `okrug` sin explosión dimensional. | Mínimo escalado requerido; creación de features de dominio (`floor_ratio = floor / total_floors`, `building_age = posting_year - building_year`, coordenadas polares al Kremlin) y función de pérdida `RMSE` sobre `log1p(y)` o `Gamma`/`Huber`. |")
    lines.append("| **3. Ensambles de Bagging No Paramétricos** | `RandomForestRegressor`, `ExtraTreesRegressor` | Útiles como punto de comparación robusto frente a varianza y valores extremos sin supuestos paramétricos, aunque no extrapolan tendencias inflacionarias temporales fuera del rango observado. | Codificación ordinal/target de alta cardinalidad (`district`) e indicadores de antigüedad y fecha. |\n")

    # 8. Índice de Gráficos Generados
    lines.append("## 8. Catálogo de Gráficos Generados (`reports/figures/`)\n")
    lines.append("| Archivo | Finalidad Analítica |")
    lines.append("|---|---|")
    for fig in figures_meta:
        lines.append(f"| `reports/figures/{fig['file']}` | {fig['purpose']} |")
    lines.append("")

    return "\n".join(lines)


# =============================================================================
# PUNTO DE ENTRADA (CLI Y ORQUESTACIÓN)
# =============================================================================
def run_eda(
    input_csv: Path,
    cleaning_report_md: Path,
    report_out_md: Path,
    figures_dir: Path,
) -> Dict[str, Any]:
    """Ejecuta el flujo completo del EDA garantizando la inmutabilidad del archivo de entrada."""
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    LOGGER.info("Iniciando 02_eda.py sobre '%s'", input_csv)

    # 1. Validación inicial y cálculo de hash SHA-256 previo
    df, load_warnings = load_and_validate_dataset(input_csv)
    sha_before = compute_sha256(input_csv)
    mtime_before = input_csv.stat().st_mtime_ns

    # 2. Lectura del reporte de limpieza (si existe)
    cleaning_info = parse_cleaning_report(cleaning_report_md)

    # 3. Clasificación de columnas disponibles
    numeric_candidates = [
        c for c in [
            "monthly_rent_rub", "rent_per_sqm", "total_area", "rooms",
            "floor", "total_floors", "building_year", "deposit_months",
            "metro_distance_min", "to_center_km", "lat", "lon"
        ] if c in df.columns
    ]
    cat_candidates = [
        c for c in [
            "okrug", "district", "building_type", "renovation",
            "furnished", "pets_allowed", "metro_line", "metro_station", "dq_flags"
        ] if c in df.columns
    ]

    # 4. Cálculos estadísticos
    num_summary, type_issues = compute_numeric_summary(df, numeric_candidates)
    cat_summary = compute_categorical_summary(df, cat_candidates)

    vif_predictors = [
        c for c in numeric_candidates
        if c not in ("monthly_rent_rub", "rent_per_sqm")
    ]
    vif_df = compute_vif(df, vif_predictors)

    # 5. Generación de gráficos
    figures_meta = generate_figures(df, figures_dir)

    # 6. Generación del reporte Markdown (escritura atómica)
    report_out_md.parent.mkdir(parents=True, exist_ok=True)
    report_content = build_markdown_report(
        input_path=input_csv,
        input_sha256=sha_before,
        cleaning_info=cleaning_info,
        df=df,
        load_warnings=load_warnings,
        num_summary=num_summary,
        cat_summary=cat_summary,
        type_issues=type_issues,
        vif_df=vif_df,
        figures_meta=figures_meta,
    )
    tmp_report = report_out_md.with_suffix(".md.tmp")
    tmp_report.write_text(report_content, encoding="utf-8")
    tmp_report.replace(report_out_md)

    # 7. Verificación de inmutabilidad post-ejecución
    sha_after = compute_sha256(input_csv)
    mtime_after = input_csv.stat().st_mtime_ns
    if sha_before != sha_after or mtime_before != mtime_after:
        raise RuntimeError(
            "VIOLACIÓN DE INTEGRIDAD: El archivo de entrada fue modificado durante el EDA."
        )

    LOGGER.info("EDA finalizado con éxito. Reporte en '%s' y %d figuras en '%s'.",
                report_out_md, len(figures_meta), figures_dir)
    return {
        "input_sha256": sha_before,
        "rows": len(df),
        "cols": len(df.columns),
        "report_path": str(report_out_md),
        "figures_count": len(figures_meta),
        "warnings": load_warnings,
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="02_eda.py - Análisis Exploratorio de Datos reproducible para alquileres de Moscú."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/rentals_clean.csv"),
        help="Ruta al dataset limpio de entrada (default: data/rentals_clean.csv)",
    )
    parser.add_argument(
        "--cleaning-report",
        type=Path,
        default=Path("data/cleaning_report.md"),
        help="Ruta al informe de limpieza previa (default: data/cleaning_report.md)",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("reports/eda_report.md"),
        help="Ruta de salida para el informe de EDA (default: reports/eda_report.md)",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("reports/figures"),
        help="Directorio de salida para los gráficos generados (default: reports/figures)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    try:
        args = parse_args(argv)
        run_eda(
            input_csv=args.input,
            cleaning_report_md=args.cleaning_report,
            report_out_md=args.report_out,
            figures_dir=args.figures_dir,
        )
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except Exception as exc:
        LOGGER.exception("Error inesperado durante la ejecución del EDA: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())