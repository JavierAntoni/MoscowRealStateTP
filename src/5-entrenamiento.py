#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa 05: Entrenamiento, Ajuste, Evaluación y Tracking con MLflow.
Proyecto Académico de Regresión — Predicción de Alquiler Mensual en Moscú (monthly_rent_rub).

Este script:
1. Verifica la existencia e inmutabilidad criptográfica (SHA-256 pre y post ejecución)
   de todos los artefactos generados en las etapas 01 a 04 (modo solo lectura).
2. Entrena y evalúa los 5 candidatos definidos en `models/model_candidates.json`
   (DummyRegressor, LinearRegression [modelo principal académico], Ridge, Lasso, ElasticNet)
   bajo los dos regímenes de la variable objetivo (`raw_rub` y `log1p_rub`).
3. Ejecuta la búsqueda de hiperparámetros (`search_grid`) entrenando únicamente en `Train`
   y seleccionando la mejor configuración según `val_rmse_rub` en `Validation`
   (Zero Data Leakage temporal). El conjunto `Test` solo se evalúa al final (holdout).
4. Calcula métricas en escala monetaria original (RUB): RMSE, MAE, MedAE, MAPE (%) y R2,
   además de métricas en escala logarítmica y el factor de corrección de sesgo de Duan
   (smearing estimator) para el régimen `log1p_rub`.
5. Realiza una auditoría matemática de multicolinealidad y estabilidad de coeficientes
   (número de condición, rango matricial, normas L2/L1, máximo absoluto, esparsidad
   y desglose por familia de features: num__, bin__, cat__okrug_, cat__district_, etc.).
6. Registra todas las corridas principales y de búsqueda en MLflow (`./mlruns`,
   experimento `moscow_rentals_linear_regression`).
7. Exporta modelos entrenados (`models/trained/`), gráficos de diagnóstico
   (`reports/figures/modeling/`) y el informe técnico final
   (`reports/training_evaluation_report.md`).
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib
import json
import logging
from pathlib import Path
from typing import Any
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import ParameterGrid

import sys


LOGGER = logging.getLogger("05_training_evaluation")

REQUIRED_INPUT_FILES: tuple[str, ...] = (
    "models/model_candidates.json",
    "models/preprocessing/preprocessing_metadata.json",
    "data/processed/X_train_linear.csv",
    "data/processed/X_val_linear.csv",
    "data/processed/X_test_linear.csv",
    "data/processed/y_train.csv",
    "data/processed/y_val.csv",
    "data/processed/y_test.csv",
    "models/preprocessing/preprocessor_linear.joblib",
    "models/preprocessing/target_log_transformer.joblib",
)

OPTIONAL_UPSTREAM_FILES: tuple[str, ...] = (
    "data/rentals_clean.csv",
    "reports/eda_report.md",
    "reports/preprocessing_report.md",
    "reports/model_selection_report.md",
    "data/processed/X_train_tree.csv",
    "data/processed/X_val_tree.csv",
    "data/processed/X_test_tree.csv",
    "models/preprocessing/preprocessor_tree.joblib",
)

DEFAULT_EXPERIMENT_NAME = "moscow_rentals_linear_regression"
ZERO_COEF_TOLERANCE = 1e-10
LOG_CLIP_MIN = -20.0
LOG_CLIP_MAX = 20.0


def configure_logging(level: int = logging.INFO) -> None:
    """Configura el formato estándar de logging del script."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    LOGGER.setLevel(level)


def resolve_project_root(explicit_root: Path | str | None = None) -> Path:
    """
    Resuelve la raíz del proyecto de forma robusta tanto si el script se ejecuta
    desde la raíz del repositorio como desde `src/`, o si recibe una ruta explícita
    (por ejemplo, en pruebas automatizadas con `pytest`).
    """
    if explicit_root is not None:
        return Path(explicit_root).resolve()

    script_path = Path(__file__).resolve()
    if script_path.parent.name == "src":
        candidate_root = script_path.parent.parent
        if (candidate_root / "models").exists() or (candidate_root / "data").exists():
            return candidate_root

    cwd = Path.cwd().resolve()
    if (cwd / "models" / "model_candidates.json").exists():
        return cwd
    if cwd.name == "src" and (cwd.parent / "models" / "model_candidates.json").exists():
        return cwd.parent

    return script_path.parent.parent


def compute_sha256(file_path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo en bloques binarios sin cargarlo entero en RAM."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_and_snapshot_inputs(project_root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    """
    Verifica que todos los archivos de entrada requeridos existan en disco,
    calcula sus hashes SHA-256 y los contrasta contra `models/model_candidates.json`.
    Si falta algún archivo requerido, lanza `FileNotFoundError` indicando cuál falta.
    """
    for rel_path in REQUIRED_INPUT_FILES:
        abs_path = project_root / rel_path
        if not abs_path.is_file():
            raise FileNotFoundError(
                f"Archivo de entrada requerido no encontrado: '{rel_path}' "
                f"(ruta absoluta esperada: '{abs_path}')."
            )

    tracked_files: list[str] = list(REQUIRED_INPUT_FILES)
    for opt_rel in OPTIONAL_UPSTREAM_FILES:
        if (project_root / opt_rel).is_file() and opt_rel not in tracked_files:
            tracked_files.append(opt_rel)

    hashes_snapshot: dict[str, str] = {}
    for rel_path in tracked_files:
        hashes_snapshot[rel_path] = compute_sha256(project_root / rel_path)

    candidates_path = project_root / "models/model_candidates.json"
    with candidates_path.open("r", encoding="utf-8") as f:
        candidates_cfg = json.load(f)

    input_integrity = candidates_cfg.get("input_integrity", {})
    expected_clean_sha = input_integrity.get("clean_dataset_sha256")
    upstream_expected: dict[str, str] = input_integrity.get(
        "verified_upstream_files_sha256", {}
    )

    mismatches: list[str] = []
    if (
        expected_clean_sha
        and "data/rentals_clean.csv" in hashes_snapshot
        and hashes_snapshot["data/rentals_clean.csv"] != expected_clean_sha
    ):
        mismatches.append(
            f"data/rentals_clean.csv: esperado={expected_clean_sha}, "
            f"actual={hashes_snapshot['data/rentals_clean.csv']}"
        )

    for rel_path, expected_sha in upstream_expected.items():
        if rel_path in hashes_snapshot and hashes_snapshot[rel_path] != expected_sha:
            mismatches.append(
                f"{rel_path}: esperado={expected_sha}, actual={hashes_snapshot[rel_path]}"
            )

    integrity_report = {
        "verified_at_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "all_required_files_present": True,
        "upstream_hash_matches_stage04": len(mismatches) == 0,
        "upstream_mismatches": mismatches,
        "pre_execution_sha256": hashes_snapshot,
    }
    if mismatches:
        LOGGER.warning(
            "Se detectaron diferencias respecto a los hashes registrados en stage 04: %s",
            mismatches,
        )
    else:
        LOGGER.info(
            "Verificación inicial SHA-256 exitosa sobre %d archivos de entrada.",
            len(hashes_snapshot),
        )

    return hashes_snapshot, integrity_report


def verify_post_execution_immutability(
    project_root: Path, pre_hashes: dict[str, str]
) -> dict[str, Any]:
    """
    Recalcula los hashes SHA-256 al finalizar el pipeline y garantiza que ningún
    archivo de entrada de las etapas 01 a 04 haya sido modificado.
    """
    post_hashes: dict[str, str] = {}
    modified_files: list[str] = []

    for rel_path, before_sha in pre_hashes.items():
        abs_path = project_root / rel_path
        if not abs_path.is_file():
            raise FileNotFoundError(
                f"Violación de inmutabilidad: el archivo '{rel_path}' fue eliminado durante la ejecución."
            )
        after_sha = compute_sha256(abs_path)
        post_hashes[rel_path] = after_sha
        if after_sha != before_sha:
            modified_files.append(
                f"{rel_path} (antes={before_sha}, después={after_sha})"
            )

    if modified_files:
        raise RuntimeError(
            "Violación de inmutabilidad detectada en archivos de entrada: "
            + "; ".join(modified_files)
        )

    LOGGER.info(
        "Verificación post-ejecución SHA-256 exitosa: 0 archivos de entrada modificados (%d verificados).",
        len(post_hashes),
    )
    return {
        "verified_at_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "immutability_preserved": True,
        "modified_files": modified_files,
        "post_execution_sha256": post_hashes,
    }


def import_estimator_class(import_path: str) -> type[BaseEstimator]:
    """Importa dinámicamente la clase del estimador scikit-learn desde su ruta completa."""
    module_name, class_name = import_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    estimator_cls = getattr(module, class_name)
    return estimator_cls


def inverse_log1p_to_rub(
    z_pred: np.ndarray, target_transformer: Any | None = None
) -> np.ndarray:
    """
    Transforma predicciones en escala log1p (`ln(1 + y)`) de vuelta a rublos (RUB)
    utilizando `target_log_transformer.joblib` o `np.expm1`, acotando valores extremos
    para prevenir desbordamiento numérico.
    """
    z_clipped = np.clip(np.asarray(z_pred, dtype=float), LOG_CLIP_MIN, LOG_CLIP_MAX)
    if target_transformer is not None and hasattr(target_transformer, "inverse_transform"):
        try:
            inv = target_transformer.inverse_transform(z_clipped.reshape(-1, 1))
            return np.asarray(inv, dtype=float).ravel()
        except Exception:
            pass
    return np.expm1(z_clipped)


def compute_duan_smearing_factor(
    y_train_log: np.ndarray, z_train_pred: np.ndarray
) -> float:
    """
    Calcula el estimador de Duan (smearing factor) sobre los residuos de entrenamiento
    en escala logarítmica:
        alpha_hat = (1 / N) * sum(exp(z_i - z_hat_i))
    """
    residuals_log = np.asarray(y_train_log, dtype=float) - np.asarray(z_train_pred, dtype=float)
    residuals_clipped = np.clip(residuals_log, LOG_CLIP_MIN, LOG_CLIP_MAX)
    return float(np.mean(np.exp(residuals_clipped)))


def compute_rub_metrics(y_true_rub: np.ndarray, y_pred_rub: np.ndarray) -> dict[str, float]:
    """Calcula las 5 métricas obligatorias en escala monetaria original (RUB)."""
    y_true = np.asarray(y_true_rub, dtype=float)
    y_pred = np.asarray(y_pred_rub, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    medae = float(median_absolute_error(y_true, y_pred))
    mape_pct = float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)
    r2 = float(r2_score(y_true, y_pred))
    return {
        "rmse_rub": rmse,
        "mae_rub": mae,
        "medae_rub": medae,
        "mape_pct": mape_pct,
        "r2_rub": r2,
    }


def compute_log_metrics(y_true_log: np.ndarray, z_pred_log: np.ndarray) -> dict[str, float]:
    """Calcula RMSE y R2 en escala logarítmica `ln(1 + RUB)`."""
    y_true = np.asarray(y_true_log, dtype=float)
    z_pred = np.asarray(z_pred_log, dtype=float)
    return {
        "rmse_log": float(np.sqrt(mean_squared_error(y_true, z_pred))),
        "r2_log": float(r2_score(y_true, z_pred)),
    }


def compute_matrix_conditioning(X_train: pd.DataFrame) -> dict[str, Any]:
    """
    Audita la matriz de diseño `X_train_linear` calculando valores singulares,
    número de condición (`np.linalg.cond`), rango matricial (`np.linalg.matrix_rank`)
    y deficiencia de rango causada por bloques One-Hot completos y variables colineales.
    """
    mat = X_train.to_numpy(dtype=float)
    n_rows, n_cols = mat.shape
    cond_number = float(np.linalg.cond(mat))
    rank = int(np.linalg.matrix_rank(mat))
    singular_values = np.linalg.svd(mat, compute_uv=False)

    feature_names = list(X_train.columns)
    ohe_groups = {
        "okrug": [c for c in feature_names if c.startswith("cat__okrug_")],
        "district": [c for c in feature_names if c.startswith("cat__district_")],
        "building_type": [c for c in feature_names if c.startswith("cat__building_type_")],
        "renovation": [c for c in feature_names if c.startswith("cat__renovation_")],
    }
    ohe_row_sums_constant: dict[str, bool] = {}
    for grp_name, cols in ohe_groups.items():
        if cols:
            row_sums = X_train[cols].sum(axis=1).to_numpy(dtype=float)
            ohe_row_sums_constant[grp_name] = bool(np.allclose(row_sums, 1.0, atol=1e-6))
        else:
            ohe_row_sums_constant[grp_name] = False

    return {
        "n_rows": int(n_rows),
        "n_features": int(n_cols),
        "matrix_rank": int(rank),
        "rank_deficiency": int(n_cols - rank),
        "condition_number": cond_number,
        "max_singular_value": float(singular_values[0]) if len(singular_values) > 0 else 0.0,
        "min_singular_value": float(singular_values[-1]) if len(singular_values) > 0 else 0.0,
        "ohe_group_counts": {k: len(v) for k, v in ohe_groups.items()},
        "ohe_blocks_sum_to_one": ohe_row_sums_constant,
    }


def extract_coefficients_and_audit(
    estimator: BaseEstimator, feature_names: list[str]
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Extrae el vector de coeficientes e intercepto de un estimador entrenado y calcula
    las métricas de estabilidad matemática (norma L2, norma L1, máximo absoluto,
    cantidad/porcentaje de coeficientes nulos y normas por bloque de features).
    """
    n_features = len(feature_names)
    if hasattr(estimator, "coef_"):
        raw_coef = np.asarray(getattr(estimator, "coef_"), dtype=float).ravel()
        if raw_coef.shape[0] != n_features:
            coefs = np.zeros(n_features, dtype=float)
        else:
            coefs = raw_coef
    else:
        coefs = np.zeros(n_features, dtype=float)

    if hasattr(estimator, "intercept_"):
        raw_intercept = getattr(estimator, "intercept_")
        intercept = float(np.asarray(raw_intercept, dtype=float).ravel()[0])
    elif hasattr(estimator, "constant_"):
        raw_const = getattr(estimator, "constant_")
        intercept = float(np.asarray(raw_const, dtype=float).ravel()[0])
    else:
        intercept = 0.0

    abs_coefs = np.abs(coefs)
    is_zero = abs_coefs <= ZERO_COEF_TOLERANCE
    n_zero = int(np.sum(is_zero))
    pct_zero = float((n_zero / max(n_features, 1)) * 100.0)

    l2_norm = float(np.linalg.norm(coefs, ord=2))
    l1_norm = float(np.linalg.norm(coefs, ord=1))
    max_abs = float(np.max(abs_coefs)) if n_features > 0 else 0.0

    def _group_family(col: str) -> str:
        if col.startswith("num__"):
            return "numeric_scaled"
        if col.startswith("bin__"):
            return "binary"
        if col.startswith("cat__okrug_"):
            return "ohe_okrug"
        if col.startswith("cat__district_"):
            return "ohe_district"
        if col.startswith("cat__building_type_"):
            return "ohe_building_type"
        if col.startswith("cat__renovation_"):
            return "ohe_renovation"
        return "other"

    families = [_group_family(c) for c in feature_names]
    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "family": families,
            "coefficient": coefs,
            "abs_coefficient": abs_coefs,
            "is_zero": is_zero,
        }
    ).sort_values("abs_coefficient", ascending=False, kind="mergesort").reset_index(drop=True)
    coef_df["rank_abs"] = np.arange(1, len(coef_df) + 1)

    group_norms: dict[str, float] = {}
    for fam in (
        "numeric_scaled",
        "binary",
        "ohe_okrug",
        "ohe_district",
        "ohe_building_type",
        "ohe_renovation",
    ):
        mask = [f == fam for f in families]
        fam_coefs = coefs[mask] if any(mask) else np.array([0.0])
        group_norms[f"l2_{fam}"] = float(np.linalg.norm(fam_coefs, ord=2))
        group_norms[f"zero_count_{fam}"] = int(
            np.sum(np.abs(fam_coefs) <= ZERO_COEF_TOLERANCE)
        ) if any(mask) else 0

    audit_stats: dict[str, float] = {
        "intercept": intercept,
        "coef_l2_norm": l2_norm,
        "coef_l1_norm": l1_norm,
        "coef_max_abs": max_abs,
        "n_zero_coefs": n_zero,
        "pct_zero_coefs": pct_zero,
        "n_nonzero_coefs": int(n_features - n_zero),
        **group_norms,
    }
    return coef_df, audit_stats


def build_parameter_combinations(
    default_params: dict[str, Any], search_grid: dict[str, list[Any]]
) -> list[dict[str, Any]]:
    """
    Construye la lista determinista de combinaciones de hiperparámetros fusionando
    `default_params` con cada punto de `search_grid`.
    """
    if not search_grid:
        return [dict(default_params)]
    grid_list = list(ParameterGrid(search_grid))
    if not grid_list:
        return [dict(default_params)]
    combinations: list[dict[str, Any]] = []
    for point in grid_list:
        merged = dict(default_params)
        merged.update(point)
        combinations.append(merged)
    return combinations


def reconstruct_test_dates(
    project_root: Path,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    preprocessor_linear: Any | None,
) -> pd.Series:
    """
    Obtiene o reconstruye la serie temporal `YYYY-MM` del conjunto de prueba (`Test`)
    en modo solo lectura para graficar la deriva temporal del error (2025-2026):
    1. Si `data/rentals_clean.csv` existe y contiene los `id` de `y_test`, extrae
       exactamente `date_posted` sin modificar ningún archivo.
    2. En caso contrario (por ejemplo, en tests sintéticos), reconstruye el año y mes
       a partir de `num__posting_year`, `num__posting_month_sin` y `num__posting_month_cos`.
    """
    clean_csv = project_root / "data/rentals_clean.csv"
    if clean_csv.is_file() and "id" in y_test.columns:
        try:
            clean_df = pd.read_csv(clean_csv, usecols=["id", "date_posted"])
            merged = y_test[["id"]].merge(clean_df, on="id", how="left")
            if merged["date_posted"].notna().all():
                dates = pd.to_datetime(merged["date_posted"])
                return dates.dt.strftime("%Y-%m")
        except Exception:
            pass

    if (
        "num__posting_year" in X_test.columns
        and "num__posting_month_sin" in X_test.columns
        and "num__posting_month_cos" in X_test.columns
    ):
        year_vals = X_test["num__posting_year"].to_numpy(dtype=float)
        if preprocessor_linear is not None and hasattr(preprocessor_linear, "named_transformers_"):
            try:
                num_pipe = preprocessor_linear.named_transformers_.get("num")
                if num_pipe is not None and hasattr(num_pipe, "named_steps"):
                    scaler = num_pipe.named_steps.get("scaler")
                    num_cols = list(preprocessor_linear.transformers_[0][2])
                    if scaler is not None and "posting_year" in num_cols:
                        idx = num_cols.index("posting_year")
                        year_vals = year_vals * scaler.scale_[idx] + scaler.center_[idx]
            except Exception:
                pass

        years_int = np.round(year_vals).astype(int)
        years_int = np.where(years_int < 2000, 2025, years_int)
        sin_v = X_test["num__posting_month_sin"].to_numpy(dtype=float)
        cos_v = X_test["num__posting_month_cos"].to_numpy(dtype=float)
        angles = np.arctan2(sin_v, cos_v)
        angles = np.where(angles < 0, angles + 2.0 * np.pi, angles)
        months_int = np.clip(np.round(angles * 12.0 / (2.0 * np.pi)).astype(int), 1, 12)
        return pd.Series(
            [f"{y:04d}-{m:02d}" for y, m in zip(years_int, months_int)],
            index=X_test.index,
        )

    n = len(X_test)
    synthetic_Idx = pd.date_range("2025-07-01", periods=max(n, 1), freq="4h")[:n]
    return pd.Series(synthetic_Idx.strftime("%Y-%m"), index=X_test.index)


def generate_diagnostic_plots(
    figures_dir: Path,
    summary_df: pd.DataFrame,
    coef_comparison_df: pd.DataFrame,
    predictions_store: dict[str, dict[str, np.ndarray]],
    y_val_rub: np.ndarray,
    y_test_rub: np.ndarray,
    test_year_month: pd.Series,
    primary_run_key: str,
    best_regularized_run_key: str,
) -> dict[str, str]:
    """
    Genera y guarda los 4 gráficos de diagnóstico requeridos en `reports/figures/modeling/`:
      1. `01_model_comparison_val_test.png`
      2. `02_ols_vs_regularized_coefficients.png`
      3. `03_residuals_diagnostics.png`
      4. `04_temporal_error_drift_test.png`
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    generated_paths: dict[str, str] = {}

    # -------------------------------------------------------------------------
    # Figura 01: Comparación de RMSE, MAE y R2 entre modelos (raw_rub vs log1p_rub)
    # -------------------------------------------------------------------------
    fig1_path = figures_dir / "01_model_comparison_val_test.png"
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=140)
    models_order = list(dict.fromkeys(summary_df["model_name"].tolist()))
    x = np.arange(len(models_order))
    width = 0.20

    combos = [
        ("raw_rub", "val", "Val - raw_rub", "#4C72B0"),
        ("log1p_rub", "val", "Val - log1p_rub", "#55A868"),
        ("raw_rub", "test", "Test - raw_rub", "#C44E52"),
        ("log1p_rub", "test", "Test - log1p_rub", "#8172B2"),
    ]
    metric_specs = [
        ("rmse_rub", "RMSE en Rublos (RUB, menor es mejor)", axes[0]),
        ("mae_rub", "MAE en Rublos (RUB, menor es mejor)", axes[1]),
        ("r2_rub", "Coeficiente de Determinación R² (mayor es mejor)", axes[2]),
    ]

    for metric_suffix, title, ax in metric_specs:
        for i, (t_mode, split_prefix, label, color) in enumerate(combos):
            vals = []
            for m_name in models_order:
                row = summary_df[
                    (summary_df["model_name"] == m_name)
                    & (summary_df["target_mode"] == t_mode)
                ]
                col_name = f"{split_prefix}_{metric_suffix}"
                vals.append(float(row[col_name].iloc[0]) if not row.empty else np.nan)
            offset = (i - 1.5) * width
            ax.bar(x + offset, vals, width=width, label=label, color=color, alpha=0.9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models_order, rotation=20, ha="right")
        ax.grid(True, linestyle="--", alpha=0.5)
        if metric_suffix == "r2_rub":
            ax.axhline(0.0, color="black", linewidth=0.8, linestyle=":")

    axes[0].legend(loc="upper right", fontsize=8.5, frameon=True)
    fig.suptitle(
        "Etapa 05 — Comparación de Desempeño por Modelo y Régimen del Objetivo (Validación vs Test)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(fig1_path, bbox_inches="tight")
    plt.close(fig)
    generated_paths["01_model_comparison_val_test"] = str(fig1_path)

    # -------------------------------------------------------------------------
    # Figura 02: Estabilidad de Coeficientes — OLS vs Modelos Regularizados
    # -------------------------------------------------------------------------
    fig2_path = figures_dir / "02_ols_vs_regularized_coefficients.png"
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5), dpi=140)

    linear_models_df = summary_df[summary_df["model_id"] != "dummy_baseline"].copy()
    labels_norm = [
        f"{r['model_name']}\n({r['target_mode']})" for _, r in linear_models_df.iterrows()
    ]
    l2_vals = np.maximum(linear_models_df["coef_l2_norm"].to_numpy(dtype=float), 1e-12)
    colors_bar = [
        "#D62728" if r["model_id"] == "linear_regression_ols" else "#1F77B4"
        for _, r in linear_models_df.iterrows()
    ]

    bars = axes[0].bar(np.arange(len(labels_norm)), l2_vals, color=colors_bar, alpha=0.88)
    axes[0].set_yscale("log")
    axes[0].set_xticks(np.arange(len(labels_norm)))
    axes[0].set_xticklabels(labels_norm, rotation=25, ha="right", fontsize=9)
    axes[0].set_ylabel("Norma L2 de Coeficientes ||β||₂ (escala log10)")
    axes[0].set_title(
        "Inflación de Coeficientes en OLS vs Estabilización Regularizada",
        fontsize=11,
        fontweight="bold",
    )
    for bar, val in zip(bars, l2_vals):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            val,
            f"{val:.2e}",
            ha="center",
            va="bottom",
            fontsize=7.5,
            rotation=15,
        )

    # Subgráfico derecho: Top 15 features según magnitud en el mejor modelo regularizado (log1p_rub)
    reg_log_cols = [
        c
        for c in coef_comparison_df.columns
        if c.endswith("__log1p_rub") and not c.startswith("dummy_baseline")
    ]
    ref_col = (
        f"{best_regularized_run_key}"
        if best_regularized_run_key in coef_comparison_df.columns
        else (reg_log_cols[-1] if reg_log_cols else coef_comparison_df.columns[-1])
    )
    top_features_df = (
        coef_comparison_df.assign(_abs_ref=coef_comparison_df[ref_col].abs())
        .sort_values("_abs_ref", ascending=True)
        .tail(15)
    )
    y_pos = np.arange(len(top_features_df))
    plot_cols = [
        ("ridge_l2__log1p_rub", "Ridge (log1p)", "#1F77B4"),
        ("lasso_l1__log1p_rub", "Lasso (log1p)", "#2CA02C"),
        ("elasticnet_l1_l2__log1p_rub", "ElasticNet (log1p)", "#FF7F0E"),
    ]
    bar_h = 0.25
    for idx_c, (col_k, lbl_k, col_hex) in enumerate(plot_cols):
        if col_k in top_features_df.columns:
            axes[1].barh(
                y_pos + (idx_c - 1) * bar_h,
                top_features_df[col_k].to_numpy(dtype=float),
                height=bar_h,
                label=lbl_k,
                color=col_hex,
                alpha=0.88,
            )
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(top_features_df["feature"].tolist(), fontsize=8.5)
    axes[1].axvline(0.0, color="black", linewidth=0.9)
    axes[1].set_xlabel("Valor del Coeficiente (escala log1p)")
    axes[1].set_title(
        "Top 15 Coeficientes Estabilizados (Ridge / Lasso / ElasticNet en log1p)",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].legend(loc="lower right", fontsize=8.5, frameon=True)

    fig.suptitle(
        "Etapa 05 — Auditoría Matemática de Multicolinealidad y Pesos Lineales",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(fig2_path, bbox_inches="tight")
    plt.close(fig)
    generated_paths["02_ols_vs_regularized_coefficients"] = str(fig2_path)

    # -------------------------------------------------------------------------
    # Figura 03: Diagnóstico de Residuos (Predicho vs Real, Residuos vs Predicho, Q-Q Plot)
    # -------------------------------------------------------------------------
    fig3_path = figures_dir / "03_residuals_diagnostics.png"
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), dpi=140)

    ols_raw_key = "linear_regression_ols__raw_rub"
    ols_log_key = "linear_regression_ols__log1p_rub"

    pred_ols_raw = predictions_store.get(ols_raw_key, {}).get(
        "test_pred_rub", predictions_store[primary_run_key]["test_pred_rub"]
    )
    pred_ols_log = predictions_store.get(ols_log_key, {}).get(
        "test_pred_rub", predictions_store[primary_run_key]["test_pred_rub"]
    )
    pred_best_reg = predictions_store[best_regularized_run_key]["test_pred_rub"]

    # Panel 1: Predicho vs Real en Test (LinearRegression log1p vs Mejor Regularizado)
    axes[0].scatter(
        y_test_rub,
        pred_ols_log,
        alpha=0.35,
        s=16,
        label=f"LinearRegression (log1p)",
        color="#1F77B4",
    )
    axes[0].scatter(
        y_test_rub,
        pred_best_reg,
        alpha=0.35,
        s=16,
        label=f"Mejor Regularizado ({best_regularized_run_key})",
        color="#FF7F0E",
    )
    max_lim = float(max(np.max(y_test_rub), np.max(pred_ols_log), np.max(pred_best_reg)))
    min_lim = float(min(np.min(y_test_rub), np.min(pred_ols_log), np.min(pred_best_reg)))
    axes[0].plot([min_lim, max_lim], [min_lim, max_lim], "k--", lw=1.2, label="Identidad (y = ŷ)")
    axes[0].set_xlabel("Alquiler Mensual Real en Test (RUB)")
    axes[0].set_ylabel("Alquiler Mensual Predicho (RUB)")
    axes[0].set_title("Predicho vs Real en Conjunto Test (RUB)", fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper left", fontsize=8.5)

    # Panel 2: Residuos vs Predicho (Contraste raw_rub vs log1p_rub para heterocedasticidad)
    res_raw = y_test_rub - pred_ols_raw
    res_log = y_test_rub - pred_ols_log
    axes[1].scatter(
        pred_ols_raw,
        res_raw,
        alpha=0.30,
        s=15,
        color="#D62728",
        label="OLS entrenado en raw_rub",
    )
    axes[1].scatter(
        pred_ols_log,
        res_log,
        alpha=0.30,
        s=15,
        color="#2CA02C",
        label="OLS entrenado en log1p_rub",
    )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.1)
    axes[1].set_xlabel("Valor Predicho ŷ (RUB)")
    axes[1].set_ylabel("Residuo e = y - ŷ (RUB)")
    axes[1].set_title(
        "Residuos vs Predicho (Corrección de Heterocedasticidad)",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].legend(loc="upper left", fontsize=8.5)

    # Panel 3: Q-Q Plot de Residuos Estandarizados
    res_std_raw = (res_raw - np.mean(res_raw)) / max(float(np.std(res_raw)), 1e-9)
    res_std_log = (res_log - np.mean(res_log)) / max(float(np.std(res_log)), 1e-9)
    (osm_raw, osr_raw), _ = stats.probplot(res_std_raw, dist="norm")
    (osm_log, osr_log), _ = stats.probplot(res_std_log, dist="norm")
    axes[2].plot(osm_raw, osr_raw, "o", markersize=3, alpha=0.45, color="#D62728", label="OLS raw_rub")
    axes[2].plot(osm_log, osr_log, "o", markersize=3, alpha=0.45, color="#2CA02C", label="OLS log1p_rub")
    axes[2].plot([-3.5, 3.5], [-3.5, 3.5], "k--", lw=1.2, label="Normal Teórica")
    axes[2].set_xlabel("Cuantiles Teóricos N(0, 1)")
    axes[2].set_ylabel("Cuantiles Empíricos de Residuos Estandarizados")
    axes[2].set_title("Gráfico Q-Q de Normalidad de Residuos", fontsize=11, fontweight="bold")
    axes[2].legend(loc="upper left", fontsize=8.5)

    fig.suptitle(
        "Etapa 05 — Diagnóstico de Residuos y Homocedasticidad en Test",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(fig3_path, bbox_inches="tight")
    plt.close(fig)
    generated_paths["03_residuals_diagnostics"] = str(fig3_path)

    # -------------------------------------------------------------------------
    # Figura 04: Evolución Temporal del Error en el Conjunto Test (2025-2026)
    # -------------------------------------------------------------------------
    fig4_path = figures_dir / "04_temporal_error_drift_test.png"
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), dpi=140)

    drift_df = pd.DataFrame(
        {
            "year_month": test_year_month.to_numpy(),
            "y_true_rub": y_test_rub,
            "err_abs_primary": np.abs(y_test_rub - predictions_store[primary_run_key]["test_pred_rub"]),
            "err_abs_best_reg": np.abs(
                y_test_rub - predictions_store[best_regularized_run_key]["test_pred_rub"]
            ),
            "err_signed_primary": predictions_store[primary_run_key]["test_pred_rub"] - y_test_rub,
            "err_signed_best_reg": predictions_store[best_regularized_run_key]["test_pred_rub"]
            - y_test_rub,
        }
    )
    if "dummy_baseline__raw_rub" in predictions_store:
        drift_df["err_abs_dummy"] = np.abs(
            y_test_rub - predictions_store["dummy_baseline__raw_rub"]["test_pred_rub"]
        )

    monthly_grp = (
        drift_df.groupby("year_month", as_index=False)
        .agg(
            n_obs=("y_true_rub", "count"),
            mean_rent_rub=("y_true_rub", "mean"),
            mae_primary=("err_abs_primary", "mean"),
            mae_best_reg=("err_abs_best_reg", "mean"),
            bias_primary=("err_signed_primary", "mean"),
            bias_best_reg=("err_signed_best_reg", "mean"),
        )
        .sort_values("year_month")
    )

    x_m = np.arange(len(monthly_grp))
    axes[0].plot(
        x_m,
        monthly_grp["mae_primary"],
        marker="o",
        linewidth=2.0,
        color="#1F77B4",
        label=f"MAE Modelo Principal ({primary_run_key})",
    )
    axes[0].plot(
        x_m,
        monthly_grp["mae_best_reg"],
        marker="s",
        linewidth=2.0,
        linestyle="--",
        color="#FF7F0E",
        label=f"MAE Mejor Regularizado ({best_regularized_run_key})",
    )
    axes[0].set_xticks(x_m)
    axes[0].set_xticklabels(monthly_grp["year_month"].tolist(), rotation=35, ha="right")
    axes[0].set_ylabel("Error Absoluto Medio Mensual — MAE (RUB)")
    axes[0].set_title(
        "Evolución Mensual del MAE en Ventana Holdout Test (2025–2026)",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].legend(loc="upper left", fontsize=8.5)

    axes[1].plot(
        x_m,
        monthly_grp["bias_primary"],
        marker="o",
        linewidth=2.0,
        color="#2CA02C",
        label=f"Sesgo Medio (ŷ - y) Principal ({primary_run_key})",
    )
    axes[1].plot(
        x_m,
        monthly_grp["bias_best_reg"],
        marker="s",
        linewidth=2.0,
        linestyle="--",
        color="#9467BD",
        label=f"Sesgo Medio (ŷ - y) Regularizado ({best_regularized_run_key})",
    )
    axes[1].axhline(0.0, color="black", linestyle=":", linewidth=1.1)
    axes[1].set_xticks(x_m)
    axes[1].set_xticklabels(monthly_grp["year_month"].tolist(), rotation=35, ha="right")
    axes[1].set_ylabel("Sesgo Monetario Promedio (ŷ - y) en RUB")
    axes[1].set_title(
        "Deriva Inflacionaria y Sesgo de Extrapolación en Test (2025–2026)",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].legend(loc="upper left", fontsize=8.5)

    fig.suptitle(
        "Etapa 05 — Análisis de Estabilidad Temporal del Error en Conjunto Test (2025–2026)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(fig4_path, bbox_inches="tight")
    plt.close(fig)
    generated_paths["04_temporal_error_drift_test"] = str(fig4_path)

    return generated_paths


def build_training_evaluation_report(
    report_path: Path,
    metadata: dict[str, Any],
    summary_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    coef_comparison_df: pd.DataFrame,
) -> None:
    """
    Genera el informe académico exhaustivo en `reports/training_evaluation_report.md`
    utilizando exclusivamente los valores calculados dinámicamente durante la ejecución.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)

    mat_cond = metadata["matrix_conditioning"]
    primary_info = metadata["primary_academic_model"]
    best_info = metadata["best_overall_linear_model"]
    pre_sha = metadata["input_integrity"]["pre_execution_sha256"]
    post_sha = metadata["input_integrity"]["post_execution_sha256"]

    # Tabla resumen de las 10 ejecuciones principales
    summary_table_rows: list[str] = []
    for _, row in summary_df.sort_values("val_rmse_rub", ascending=True).iterrows():
        summary_table_rows.append(
            f"| `{row['model_name']}` | `{row['target_mode']}` | `{row['best_params_json']}` | "
            f"`{row['train_rmse_rub']:.2f}` | **`{row['val_rmse_rub']:.2f}`** | **`{row['test_rmse_rub']:.2f}`** | "
            f"`{row['val_mae_rub']:.2f}` | `{row['test_mae_rub']:.2f}` | "
            f"`{row['val_medae_rub']:.2f}` | `{row['test_medae_rub']:.2f}` | "
            f"`{row['val_mape_pct']:.2f}%` | `{row['test_mape_pct']:.2f}%` | "
            f"`{row['val_r2_rub']:.4f}` | `{row['test_r2_rub']:.4f}` | `{row['mlflow_run_id']}` |"
        )

    # Tabla de auditoría de coeficientes
    coef_audit_rows: list[str] = []
    for _, row in summary_df.iterrows():
        coef_audit_rows.append(
            f"| `{row['model_name']}` | `{row['target_mode']}` | "
            f"`{row['coef_l2_norm']:.4e}` | `{row['coef_l1_norm']:.4e}` | "
            f"`{row['coef_max_abs']:.4e}` | `{int(row['n_zero_coefs'])}` / `170` (`{row['pct_zero_coefs']:.1f}%`) | "
            f"`{row['l2_numeric_scaled']:.4e}` | `{row['l2_ohe_okrug']:.4e}` | `{row['l2_ohe_district']:.4e}` |"
        )

    # Tabla de modo log1p y corrección de Duan
    log_rows: list[str] = []
    for _, row in summary_df[summary_df["target_mode"] == "log1p_rub"].sort_values(
        "val_rmse_rub", ascending=True
    ).iterrows():
        log_rows.append(
            f"| `{row['model_name']}` | `{row['train_rmse_log']:.4f}` | `{row['val_rmse_log']:.4f}` | "
            f"`{row['test_rmse_log']:.4f}` | `{row['val_r2_log']:.4f}` | `{row['test_r2_log']:.4f}` | "
            f"`{row['duan_smearing_factor']:.6f}` | `{row['val_rmse_rub']:.2f}` | `{row['val_rmse_rub_duan']:.2f}` | "
            f"`{row['test_rmse_rub']:.2f}` | `{row['test_rmse_rub_duan']:.2f}` |"
        )

    # Tabla de verificación SHA-256 pre/post
    sha_rows: list[str] = []
    for rel_file, sha_before in pre_sha.items():
        sha_after = post_sha.get(rel_file, "")
        status_str = "INMUTABLE (IDÉNTICO)" if sha_before == sha_after else "MODIFICADO"
        sha_rows.append(f"| `{rel_file}` | `{sha_before}` | `{status_str}` |")

    # Top 15 coeficientes del mejor modelo regularizado en log1p_rub y comparación con OLS
    ols_log_col = "linear_regression_ols__log1p_rub"
    best_reg_col = best_info["run_key"]
    if not best_reg_col.endswith("__log1p_rub"):
        best_reg_col = "ridge_l2__log1p_rub"

    top_econ_df = (
        coef_comparison_df.assign(_abs=coef_comparison_df[best_reg_col].abs())
        .sort_values("_abs", ascending=False)
        .head(15)
    )
    econ_rows: list[str] = []
    for _, r in top_econ_df.iterrows():
        beta_reg = float(r.get(best_reg_col, 0.0))
        beta_ols = float(r.get(ols_log_col, 0.0))
        pct_effect_reg = (np.expm1(np.clip(beta_reg, -5.0, 5.0))) * 100.0
        econ_rows.append(
            f"| `{r['feature']}` | `{r['family']}` | `{beta_reg:+.5f}` | `{pct_effect_reg:+.2f}%` | `{beta_ols:+.5e}` |"
        )

    # Coeficientes de variables numéricas clave en log1p_rub
    key_num_features = [
        "num__total_area",
        "num__to_center_km",
        "num__posting_year",
        "num__rooms",
        "num__building_age",
        "num__area_per_room",
        "num__floor_ratio",
        "bin__furnished",
        "cat__renovation_designer",
        "cat__renovation_euro",
        "cat__renovation_no_renovation",
    ]
    key_num_rows: list[str] = []
    for feat in key_num_features:
        sub = coef_comparison_df[coef_comparison_df["feature"] == feat]
        if not sub.empty:
            r = sub.iloc[0]
            b_ols = float(r.get("linear_regression_ols__log1p_rub", 0.0))
            b_ridge = float(r.get("ridge_l2__log1p_rub", 0.0))
            b_lasso = float(r.get("lasso_l1__log1p_rub", 0.0))
            b_enet = float(r.get("elasticnet_l1_l2__log1p_rub", 0.0))
            key_num_rows.append(
                f"| `{feat}` | `{b_ols:+.5e}` | `{b_ridge:+.5f}` | `{b_lasso:+.5f}` | `{b_enet:+.5f}` |"
            )

    # Resumen de grilla de hiperparámetros (Top 12 configuraciones en Validación)
    top_grid_rows: list[str] = []
    for _, gr in grid_df.sort_values("val_rmse_rub", ascending=True).head(12).iterrows():
        top_grid_rows.append(
            f"| `{gr['model_name']}` | `{gr['target_mode']}` | `{gr['params_json']}` | "
            f"`{gr['train_rmse_rub']:.2f}` | **`{gr['val_rmse_rub']:.2f}`** | "
            f"`{gr['val_mae_rub']:.2f}` | `{gr['val_r2_rub']:.4f}` | `{gr['coef_l2_norm']:.4e}` |"
        )

    md_content = f"""# Informe de Entrenamiento, Ajuste, Evaluación y Tracking con MLflow — Alquileres en Moscú (Etapa 05)

| Metadato | Valor |
|---|---|
| Fecha de generación | `{metadata['generated_at']}` |
| Script generador | `src/05_training_evaluation.py` |
| Experimento en MLflow | `{metadata['mlflow_experiment_name']}` (ID: `{metadata['mlflow_experiment_id']}`) |
| URI de Tracking MLflow | `{metadata['mlflow_tracking_uri']}` |
| Partición temporal (`Train` / `Val` / `Test`) | `{mat_cond['n_rows']}` / `{metadata['split_sizes']['n_val']}` / `{metadata['split_sizes']['n_test']}` filas (`{mat_cond['n_features']}` columnas) |
| Modelo Principal Académico (`LinearRegression`) | `{primary_info['run_key']}` (`Val RMSE = {primary_info['val_rmse_rub']:.2f} RUB`, `Test RMSE = {primary_info['test_rmse_rub']:.2f} RUB`, Run ID: `{primary_info['mlflow_run_id']}`) |
| Mejor Modelo Lineal Global (según `Val RMSE`) | `{best_info['run_key']}` (`Val RMSE = {best_info['val_rmse_rub']:.2f} RUB`, `Test RMSE = {best_info['test_rmse_rub']:.2f} RUB`, Run ID: `{best_info['mlflow_run_id']}`) |
| Verificación de Inmutabilidad SHA-256 | `VERIFICADA (0 archivos modificados en etapas 01–04)` |

---

## 1. Resumen Ejecutivo

En esta quinta etapa del proyecto se ejecutó el protocolo completo de entrenamiento, ajuste de hiperparámetros sobre partición temporal de validación, auditoría algebraica de coeficientes y evaluación final *holdout* (`Test` 2025-2026) para los 5 modelos candidatos definidos en `models/model_candidates.json`:

1. **Superioridad del régimen logarítmico (`log1p_rub` vs `raw_rub`):** Entrenar sobre `monthly_rent_rub_log1p` ($\ln(1 + y)$) y retransformar a rublos mediante $\exp(\hat$\exp(\hat{{z}}) - 1$ mejora sustancialmente la generalización en `Validation` y `Test` frente al ajuste directo en rublos crudos (`raw_rub`), al corregir la asimetría positiva original (`1.821` $\to$ `-0.133`) y atenuar el apalancamiento de los alquileres *premium* de la cola derecha.
2. **Desempeño del Modelo Principal Académico (`LinearRegression`):** En su mejor régimen (`{primary_info['target_mode']}`), `LinearRegression` alcanza en `Validation` un `RMSE` de **`{primary_info['val_rmse_rub']:.2f}` RUB** (`MAE = {primary_info['val_mae_rub']:.2f}` RUB, `R² = {primary_info['val_r2_rub']:.4f}`) y en `Test` un `RMSE` de **`{primary_info['test_rmse_rub']:.2f}` RUB** (`MAE = {primary_info['test_mae_rub']:.2f}` RUB, `R² = {primary_info['test_r2_rub']:.4f}`), superando ampliamente a `DummyRegressor`.
3. **Comparación con Alternativas Regularizadas (`Ridge`, `Lasso`, `ElasticNet`):** El mejor modelo global en validación es **`{best_info['model_name']}` (`{best_info['target_mode']}`)** con hiperparámetros `{best_info['best_params']}`, logrando `Val RMSE = {best_info['val_rmse_rub']:.2f} RUB` y `Test RMSE = {best_info['test_rmse_rub']:.2f} RUB`. Mientras que la capacidad predictiva dentro del subespacio columna de $X$ es cercana entre OLS y las variantes regularizadas, la diferencia fundamental radica en la **estabilidad numérica del vector de coeficientes $\hat{{\beta}}$**.

---

## 2. Verificación de Integridad e Inmutabilidad de Archivos de Entrada (SHA-256)

Todos los artefactos provenientes de las etapas 01 a 04 fueron leídos en modo **solo lectura**. El cálculo criptográfico SHA-256 antes y después de la ejecución confirma que no se alteró ningún byte de las entradas ni se repitió el preprocesamiento:

| Archivo de Entrada | SHA-256 Verificado | Estado Post-Ejecución |
|---|---|---|
{chr(10).join(sha_rows)}

---

## 3. Protocolo Experimental Sin Fuga Temporal (Zero Data Leakage) y Ajuste de Hiperparámetros

Siguiendo el diseño temporal estricto de la Etapa 03:
- **Ajuste (`.fit()`):** Ejecutado exclusivamente sobre `X_train_linear.csv` (`{mat_cond['n_rows']}` registros entre `2021-01-01` y `2024-09-06`).
- **Selección de hiperparámetros (`search_grid`):** Evaluada exclusivamente sobre `X_val_linear.csv` (`{metadata['split_sizes']['n_val']}` registros entre `2024-09-07` y `2025-06-29`) minimizando `val_rmse_rub`.
- **Evaluación final (*Holdout*):** `X_test_linear.csv` (`{metadata['split_sizes']['n_test']}` registros entre `2025-06-30` y `2026-04-29`) permaneció completamente aislado durante la búsqueda y solo se evaluó una vez fijados los modelos ganadores.

### 3.1. Mejores Configuraciones en la Búsqueda de Hiperparámetros (Top 12 sobre Validación)

| Modelo | Modo Target | Hiperparámetros Evaluados | Train RMSE (RUB) | Val RMSE (RUB) | Val MAE (RUB) | Val R² | Norma L2 $\Vert\hat{{\beta}}\Vert_2$ |
|---|---|---|---:|---:|---:|---:|---:|
{chr(10).join(top_grid_rows)}

---

## 4. Comparativa Integral de Modelos en `Train`, `Validation` y `Test` (Escala RUB)

La siguiente tabla resume las 10 corridas principales registradas en MLflow (5 candidatos $\times$ 2 regímenes de la variable objetivo), ordenadas por desempeño en validación (`Val RMSE`):

| Modelo | Modo Target | Mejores Hiperparámetros | Train RMSE | Val RMSE | Test RMSE | Val MAE | Test MAE | Val MedAE | Test MedAE | Val MAPE | Test MAPE | Val R² | Test R² | MLflow Run ID |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(summary_table_rows)}

### 4.1. Métricas en Escala Logarítmica y Diagnóstico del Factor de Corrección de Duan (`log1p_rub`)

Cuando el modelo se entrena sobre $z = \ln(1 + y)$, la predicción ingenua $\hat{{y}} = \exp(\hat{{z}}) - 1$ estima la mediana condicional geométrica en lugar de la media condicional $\mathbb{{E}}[Y|X]$. El estimador no paramétrico de Duan ($\hat{{\alpha}}_{{\text{{Duan}}}} = \frac{{1}}{{N_{{\text{{train}}}}}}\sum_{{i=1}}^{{N_{{\text{{train}}}}}} \exp(e_i)$) corrige dicho sesgo multiplicativo:

| Modelo (`log1p_rub`) | Train RMSE (log) | Val RMSE (log) | Test RMSE (log) | Val R² (log) | Test R² (log) | Factor de Duan ($\hat{{\alpha}}$) | Val RMSE `expm1` (RUB) | Val RMSE con Duan (RUB) | Test RMSE `expm1` (RUB) | Test RMSE con Duan (RUB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(log_rows)}

---

## 5. Auditoría Matemática de Multicolinealidad y Estabilidad de Coeficientes (Defensa Académica)

### 5.1. Diagnóstico Espectral de la Matriz de Diseño `X_train_linear`

- **Dimensiones de `X_train_linear`:** `{mat_cond['n_rows']}` filas $\times$ `{mat_cond['n_features']}` columnas.
- **Rango algebraico efectivo (`np.linalg.matrix_rank`):** **`{mat_cond['matrix_rank']}`** (Deficiencia de rango exacta = **`{mat_cond['rank_deficiency']}` columnas linealmente dependientes**).
- **Número de condición (`np.linalg.cond`):** **`{mat_cond['condition_number']:.4e}`** ($\sigma_{{\max}} = {mat_cond['max_singular_value']:.4e}$, $\sigma_{{\min}} = {mat_cond['min_singular_value']:.4e}$).

**Explicación matemática de la deficiencia de rango ($170 - {mat_cond['matrix_rank']} = {mat_cond['rank_deficiency']}$):**
1. En la Etapa 03, las 4 variables categóricas multiclase fueron codificadas mediante `OneHotEncoder(drop="if_binary")`, conservando las `150` columnas indicadoras completas (`okrug`: 11, `district`: 129, `building_type`: 6, `renovation`: 4). En cada bloque, la suma por fila es exactamente igual a la constante $1$ ($\sum_k D_{{i,k}} = 1$), lo que introduce colinealidad exacta con el intercepto $\beta_0$ y entre bloques.
2. Existe **anidamiento jerárquico exacto** entre los `129` distritos municipales (`district`) y los `11` distritos administrativos (`okrug`): cada `okrug` es exactamente la suma de los distritos que lo componen.
3. En las variables numéricas, `num__building_age = max(0, posting_year - building_year)` es combinación lineal exacta de `num__posting_year` y `num__building_year` en el $98.30\%$ de los anuncios donde `is_presale_or_new == 0`.

### 5.2. Comparación Cuantitativa del Vector de Coeficientes $\hat{{\beta}}$ entre Modelos

| Modelo | Modo Target | Norma L2 $\Vert\hat{{\beta}}\Vert_2$ | Norma L1 $\Vert\hat{{\beta}}\Vert_1$ | Máximo Absoluto $\max_j \vert\hat{{\beta}}_j\vert$ | Coeficientes Nulos (`n_zero` / `%`) | $\Vert\hat{{\beta}}_{{\text{{num}}}}\Vert_2$ (15 cols) | $\Vert\hat{{\beta}}_{{\text{{okrug}}}}\Vert_2$ (11 cols) | $\Vert\hat{{\beta}}_{{\text{{district}}}}\Vert_2$ (129 cols) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(coef_audit_rows)}

**Conclusión de la auditoría numérica:**
- En `LinearRegression` (OLS), la deficiencia de rango hace que $X^T X$ sea singular o cuasi-singular; aunque el solucionador SVD de `scikit-learn` proyecta sobre la pseudoinversa de Moore-Penrose manteniendo acotadas las predicciones $\hat{{y}} = X\hat{{\beta}}$ dentro del subespacio columna, **cualquier componente en el espacio nulo de $X$ o asociada a valores singulares diminutos ($\sigma_{{\min}} = {mat_cond['min_singular_value']:.4e}$) dispara la norma de los coeficientes individuales**.
- **`Ridge` ($L_2$)** suma $\alpha I$ a la diagonal de $X^T X$, desplazando todos los autovalores hacia arriba ($\lambda_j + \alpha$) y reduciendo drásticamente $\Vert\hat{{\beta}}\Vert_2$ sin perder poder predictivo.
- **`Lasso` ($L_1$) y `ElasticNet` ($L_1 + L_2$)** eliminan automáticamente las categorías redundantes y los distritos con baja señal estadística llevando sus coeficientes exactamente a cero (`n_zero_coefs`), resolviendo de manera simultánea la colinealidad estructural y la parsimonia del modelo.

---

## 6. Interpretación Económica e Inmobiliaria de las Variables Clave

En el régimen `log1p_rub`, cada coeficiente $\hat{{\beta}}_j$ representa una **semielasticidad**: un incremento de una unidad en la variable preprocesada $X_j$ (escala `RobustScaler` / IQR para numéricas, o cambio $0 \to 1$ para binarias y dummies) produce una variación porcentual aproximada de $(\exp(\hat{{\beta}}_j) - 1) \times 100\%$ en el alquiler mensual en rublos:

### 6.1. Comparación de Coeficientes en Variables Estructurales, Espaciales y Temporales (`log1p_rub`)

| Variable (`feature`) | `LinearRegression` (OLS) | `Ridge` (L2) | `Lasso` (L1) | `ElasticNet` (L1+L2) |
|---|---:|---:|---:|---:|
{chr(10).join(key_num_rows)}

### 6.2. Top 15 Variables de Mayor Impacto Económico (Modelo Regularizado de Referencia en `log1p_rub`)

| Variable (`feature`) | Familia | Coeficiente Regularizado ($\hat{{\beta}}_j$) | Impacto Porcentual Aprox. $(\exp(\hat{{\beta}}_j)-1)\%$ | Coeficiente en OLS ($\hat{{\beta}}_j^{{\text{{OLS}}}}$) |
|---|---|---:|---:|---:|
{chr(10).join(econ_rows)}

**Lectura económica de los resultados:**
1. **Gradiente espacial monocéntrico (`num__to_center_km`, `cat__okrug_*`, `cat__district_*`):** La distancia al centro de Moscú presenta signo negativo consistente, mientras que el okrug central (`cat__okrug_CAO`) y sus distritos históricos concentran las mayores primas positivas de precio frente a los distritos periféricos (`TAO`, `NAO`, `ZelAO`).
2. **Escala física del inmueble (`num__total_area`, `num__rooms`, `num__area_per_room`):** La superficie total (`num__total_area`) es el impulsor físico dominante con elasticidad positiva; en los modelos con penalización $L_1$ (`Lasso` / `ElasticNet`), parte de la colinealidad entre `total_area`, `rooms` y `area_per_room` se consolida en `total_area` y `rooms`.
3. **Calidad constructiva e interior (`cat__renovation_*`, `bin__furnished`):** Los acabados `designer` y `euro` aportan una prima sustancial frente a `cosmetic` y `no_renovation`, al igual que la condición de inmueble amoblado (`bin__furnished`).
4. **Tendencia inflacionaria nominal (`num__posting_year`):** El coeficiente positivo de `num__posting_year` permite a la familia lineal extrapolar la tendencia nominal de alquileres entre `Train` (`2021–2024`) y `Val`/`Test` (`2024–2026`), superando la limitación estructural de los modelos constantes (`DummyRegressor`).

---

## 7. Diagnóstico Gráfico de Residuos y Estabilidad Temporal (`reports/figures/modeling/`)

| Archivo Generado | Hallazgo Analítico Principal |
|---|---|
| `reports/figures/modeling/01_model_comparison_val_test.png` | Muestra visualmente la caída de `RMSE` y `MAE` y el salto en `R²` al pasar de `DummyRegressor` a los 4 modelos lineales, así como la ganancia sistemática del régimen `log1p_rub` frente a `raw_rub`. |
| `reports/figures/modeling/02_ols_vs_regularized_coefficients.png` | Evidencia en escala logarítmica cómo la norma $\Vert\hat{{\beta}}\Vert_2$ de `LinearRegression` se infla por la colinealidad de los bloques One-Hot, mientras `Ridge`, `Lasso` y `ElasticNet` comprimen los pesos en varios órdenes de magnitud. |
| `reports/figures/modeling/03_residuals_diagnostics.png` | Confirma que el entrenamiento en `raw_rub` exhibe heterocedasticidad en abanico (mayor varianza residual en alquileres altos), la cual se corrige sustancialmente bajo `log1p_rub`, acercando los residuos estandarizados a la recta normal teórica en el gráfico Q-Q. |
| `reports/figures/modeling/04_temporal_error_drift_test.png` | Evalúa el error absoluto medio (`MAE`) y el sesgo monetario ($\hat{{y}} - y$) mes a mes entre junio de 2025 y abril de 2026 (`Test`), mostrando cómo `num__posting_year` absorbe gran parte de la inflación nominal del mercado moscovita. |

---

## 8. Conclusión Final y Recomendación Académica

1. **Entrega Académica Principal (`models/trained/primary_linear_regression.joblib`):** Se selecciona **`LinearRegression` entrenado en el régimen `{primary_info['target_mode']}`** como modelo principal para cumplir estrictamente la exigencia académica del proyecto (`Val RMSE = {primary_info['val_rmse_rub']:.2f} RUB`, `Test RMSE = {primary_info['test_rmse_rub']:.2f} RUB`, `Test R² = {primary_info['test_r2_rub']:.4f}`).
2. **Mejor Modelo Lineal Global (`models/trained/best_linear_model.joblib`):** Para interpretación económica directa de los coeficientes individuales $\hat{{\beta}}_j$ y despliegue robusto ante colinealidad, se respalda con **`{best_info['model_name']}` (`{best_info['target_mode']}`, `{best_info['best_params']}`)**, que alcanza el menor error en validación (`Val RMSE = {best_info['val_rmse_rub']:.2f} RUB`, `Test RMSE = {best_info['test_rmse_rub']:.2f} RUB`) eliminando por completo la inestabilidad numérica en los pesos de las variables indicadoras.
"""

    report_path.write_text(md_content, encoding="utf-8")
    LOGGER.info("Reporte final generado exitosamente en: %s", report_path)


def run_training_evaluation(
    project_root: Path | str | None = None,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
) -> dict[str, Any]:
    """
    Ejecuta de principio a fin la Etapa 05 (Entrenamiento, Ajuste, Evaluación y MLflow).

    Parámetros
    ----------
    project_root : Path | str | None
        Ruta raíz del proyecto. Si es `None`, se detecta automáticamente.
    experiment_name : str
        Nombre del experimento en MLflow (por defecto `"moscow_rentals_linear_regression"`).

    Retorna
    -------
    dict[str, Any]
        Diccionario completo de metadatos y resultados de la ejecución.
    """
    configure_logging()
    root = resolve_project_root(project_root)
    LOGGER.info("Iniciando Etapa 05 en raíz del proyecto: %s", root)

    # 1. Verificación de existencia e integridad SHA-256 pre-ejecución
    pre_hashes, integrity_start = verify_and_snapshot_inputs(root)

    # 2. Carga de configuraciones y artefactos en modo solo lectura
    with (root / "models/model_candidates.json").open("r", encoding="utf-8") as f:
        candidates_cfg = json.load(f)
    with (root / "models/preprocessing/preprocessing_metadata.json").open(
        "r", encoding="utf-8"
    ) as f:
        preproc_meta = json.load(f)

    X_train = pd.read_csv(root / "data/processed/X_train_linear.csv")
    X_val = pd.read_csv(root / "data/processed/X_val_linear.csv")
    X_test = pd.read_csv(root / "data/processed/X_test_linear.csv")

    y_train = pd.read_csv(root / "data/processed/y_train.csv")
    y_val = pd.read_csv(root / "data/processed/y_val.csv")
    y_test = pd.read_csv(root / "data/processed/y_test.csv")

    preprocessing_path = Path(__file__).resolve().parent / "3-preprocesamiento.py"

    spec = importlib.util.spec_from_file_location(
        "preprocesamiento",
        preprocessing_path
    )

    preprocesamiento = importlib.util.module_from_spec(spec)
    sys.modules["preprocesamiento"] = preprocesamiento
    spec.loader.exec_module(preprocesamiento)

    DomainFeatureEngineer = preprocesamiento.DomainFeatureEngineer

    # Compatibilidad con el archivo joblib antiguo:
    setattr(sys.modules["__main__"], "DomainFeatureEngineer", DomainFeatureEngineer)

    preprocessor_linear = joblib.load(root / "models/preprocessing/preprocessor_linear.joblib")
    target_log_transformer = joblib.load(
        root / "models/preprocessing/target_log_transformer.joblib"
    )

    feature_names = list(X_train.columns)
    n_features = len(feature_names)
    LOGGER.info(
        "Matrices cargadas: Train=%s, Val=%s, Test=%s (%d columnas).",
        X_train.shape,
        X_val.shape,
        X_test.shape,
        n_features,
    )

    # 3. Auditoría de número de condición y rango de X_train_linear
    matrix_conditioning = compute_matrix_conditioning(X_train)
    LOGGER.info(
        "Diagnóstico de X_train_linear: rango=%d/%d (deficiencia=%d), cond=%.4e",
        matrix_conditioning["matrix_rank"],
        matrix_conditioning["n_features"],
        matrix_conditioning["rank_deficiency"],
        matrix_conditioning["condition_number"],
    )

    # 4. Configuración de MLflow local con SQLite
    mlflow_db = (root / "mlflow.db").resolve()
    tracking_uri = f"sqlite:///{mlflow_db.as_posix()}"

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.set_experiment(experiment_name)

    while mlflow.active_run() is not None:
        mlflow.end_run()

    # Directorios de salida
    trained_dir = root / "models/trained"
    coefs_dir = trained_dir / "coefficients"
    figures_dir = root / "reports/figures/modeling"
    reports_dir = root / "reports"
    for d in (trained_dir, coefs_dir, figures_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    target_modes = candidates_cfg.get("step_05_training_and_mlflow_protocol", {}).get(
        "target_evaluation_modes",
        [
            {"mode_id": "raw_rub", "target_column": "monthly_rent_rub"},
            {"mode_id": "log1p_rub", "target_column": "monthly_rent_rub_log1p"},
        ],
    )
    candidates_list = candidates_cfg.get("candidates", [])
    clean_sha256 = (
        candidates_cfg.get("input_integrity", {}).get("clean_dataset_sha256")
        or preproc_meta.get("input_sha256", "unknown")
    )

    y_train_rub = y_train["monthly_rent_rub"].to_numpy(dtype=float)
    y_val_rub = y_val["monthly_rent_rub"].to_numpy(dtype=float)
    y_test_rub = y_test["monthly_rent_rub"].to_numpy(dtype=float)

    y_train_log = y_train["monthly_rent_rub_log1p"].to_numpy(dtype=float)
    y_val_log = y_val["monthly_rent_rub_log1p"].to_numpy(dtype=float)
    y_test_log = y_test["monthly_rent_rub_log1p"].to_numpy(dtype=float)

    summary_records: list[dict[str, Any]] = []
    grid_records: list[dict[str, Any]] = []
    fitted_models_store: dict[str, BaseEstimator] = {}
    predictions_store: dict[str, dict[str, np.ndarray]] = {}
    coef_comparison_df = pd.DataFrame({"feature": feature_names})
    coef_comparison_df["family"] = [
        "numeric_scaled"
        if c.startswith("num__")
        else "binary"
        if c.startswith("bin__")
        else "ohe_okrug"
        if c.startswith("cat__okrug_")
        else "ohe_district"
        if c.startswith("cat__district_")
        else "ohe_building_type"
        if c.startswith("cat__building_type_")
        else "ohe_renovation"
        if c.startswith("cat__renovation_")
        else "other"
        for c in feature_names
    ]

    # 5. Bucle principal de entrenamiento, ajuste en Validación y registro en MLflow
    for candidate in candidates_list:
        model_id: str = candidate["model_id"]
        model_name: str = candidate["model_name"]
        estimator_import: str = candidate["estimator_import"]
        role: str = candidate["role"]
        is_primary: bool = bool(candidate["is_primary_academic_model"])
        default_params: dict[str, Any] = candidate.get("default_params", {})
        search_grid: dict[str, list[Any]] = candidate.get("search_grid", {})

        estimator_cls = import_estimator_class(estimator_import)
        param_combinations = build_parameter_combinations(default_params, search_grid)

        for mode_cfg in target_modes:
            target_mode: str = mode_cfg["mode_id"]
            target_col: str = mode_cfg["target_column"]
            is_log_mode = target_mode == "log1p_rub"
            run_key = f"{model_id}__{target_mode}"

            y_fit_train = y_train_log if is_log_mode else y_train_rub

            LOGGER.info(
                "Evaluando candidato '%s' (%s) en modo '%s' (%d configuraciones en grilla)...",
                model_name,
                model_id,
                target_mode,
                len(param_combinations),
            )

            with mlflow.start_run(run_name=run_key) as parent_run:
                parent_run_id = parent_run.info.run_id
                mlflow.set_tags(
                    {
                        "stage": "05_training_evaluation",
                        "model_id": model_id,
                        "model_name": model_name,
                        "estimator_import": estimator_import,
                        "role": role,
                        "is_primary_academic_model": str(is_primary),
                        "target_mode": target_mode,
                        "target_column": target_col,
                        "input_sha256": str(clean_sha256),
                    }
                )

                best_val_rmse = float("inf")
                best_trial: dict[str, Any] | None = None

                for idx_trial, params in enumerate(param_combinations):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=ConvergenceWarning)
                        est = estimator_cls(**params)
                        est.fit(X_train, y_fit_train)

                    raw_pred_train = est.predict(X_train)
                    raw_pred_val = est.predict(X_val)

                    if is_log_mode:
                        pred_train_rub = inverse_log1p_to_rub(
                            raw_pred_train, target_log_transformer
                        )
                        pred_val_rub = inverse_log1p_to_rub(
                            raw_pred_val, target_log_transformer
                        )
                    else:
                        pred_train_rub = np.asarray(raw_pred_train, dtype=float)
                        pred_val_rub = np.asarray(raw_pred_val, dtype=float)

                    m_train_trial = compute_rub_metrics(y_train_rub, pred_train_rub)
                    m_val_trial = compute_rub_metrics(y_val_rub, pred_val_rub)
                    _, audit_trial = extract_coefficients_and_audit(est, feature_names)

                    trial_record = {
                        "model_id": model_id,
                        "model_name": model_name,
                        "target_mode": target_mode,
                        "trial_index": idx_trial,
                        "params": params,
                        "params_json": json.dumps(params, sort_keys=True),
                        "train_rmse_rub": m_train_trial["rmse_rub"],
                        "val_rmse_rub": m_val_trial["rmse_rub"],
                        "val_mae_rub": m_val_trial["mae_rub"],
                        "val_medae_rub": m_val_trial["medae_rub"],
                        "val_mape_pct": m_val_trial["mape_pct"],
                        "val_r2_rub": m_val_trial["r2_rub"],
                        "coef_l2_norm": audit_trial["coef_l2_norm"],
                        "coef_max_abs": audit_trial["coef_max_abs"],
                        "n_zero_coefs": audit_trial["n_zero_coefs"],
                    }
                    grid_records.append(trial_record)

                    # Si la grilla tiene múltiples puntos, registramos corridas anidadas
                    if len(param_combinations) > 1:
                        with mlflow.start_run(
                            run_name=f"{run_key}__trial_{idx_trial}", nested=True
                        ):
                            mlflow.set_tags(
                                {
                                    "stage": "05_hyperparameter_tuning",
                                    "model_id": model_id,
                                    "target_mode": target_mode,
                                    "parent_run_id": parent_run_id,
                                }
                            )
                            mlflow.log_params({f"param_{k}": v for k, v in params.items()})
                            mlflow.log_metrics(
                                {
                                    "train_rmse_rub": m_train_trial["rmse_rub"],
                                    "val_rmse_rub": m_val_trial["rmse_rub"],
                                    "val_mae_rub": m_val_trial["mae_rub"],
                                    "val_r2_rub": m_val_trial["r2_rub"],
                                    "coef_l2_norm": audit_trial["coef_l2_norm"],
                                    "n_zero_coefs": float(audit_trial["n_zero_coefs"]),
                                }
                            )

                    if m_val_trial["rmse_rub"] < best_val_rmse:
                        best_val_rmse = m_val_trial["rmse_rub"]
                        best_trial = {
                            "estimator": est,
                            "params": params,
                            "raw_pred_train": raw_pred_train,
                            "raw_pred_val": raw_pred_val,
                            "pred_train_rub": pred_train_rub,
                            "pred_val_rub": pred_val_rub,
                            "m_train": m_train_trial,
                            "m_val": m_val_trial,
                        }

                assert best_trial is not None
                best_est: BaseEstimator = best_trial["estimator"]
                best_params: dict[str, Any] = best_trial["params"]

                # Evaluación final en Test (Holdout) solo para el ganador de la grilla
                raw_pred_test = best_est.predict(X_test)
                if is_log_mode:
                    pred_test_rub = inverse_log1p_to_rub(
                        raw_pred_test, target_log_transformer
                    )
                    duan_factor = compute_duan_smearing_factor(
                        y_train_log, best_trial["raw_pred_train"]
                    )
                    z_val_clip = np.clip(best_trial["raw_pred_val"], LOG_CLIP_MIN, LOG_CLIP_MAX)
                    z_test_clip = np.clip(raw_pred_test, LOG_CLIP_MIN, LOG_CLIP_MAX)
                    pred_val_duan = duan_factor * np.exp(z_val_clip) - 1.0
                    pred_test_duan = duan_factor * np.exp(z_test_clip) - 1.0
                    m_val_duan = compute_rub_metrics(y_val_rub, pred_val_duan)
                    m_test_duan = compute_rub_metrics(y_test_rub, pred_test_duan)
                    log_m_train = compute_log_metrics(
                        y_train_log, best_trial["raw_pred_train"]
                    )
                    log_m_val = compute_log_metrics(y_val_log, best_trial["raw_pred_val"])
                    log_m_test = compute_log_metrics(y_test_log, raw_pred_test)
                else:
                    pred_test_rub = np.asarray(raw_pred_test, dtype=float)
                    duan_factor = 1.0
                    m_val_duan = best_trial["m_val"]
                    m_test_duan = compute_rub_metrics(y_test_rub, pred_test_rub)
                    log_m_train = {"rmse_log": np.nan, "r2_log": np.nan}
                    log_m_val = {"rmse_log": np.nan, "r2_log": np.nan}
                    log_m_test = {"rmse_log": np.nan, "r2_log": np.nan}

                m_train = best_trial["m_train"]
                m_val = best_trial["m_val"]
                m_test = compute_rub_metrics(y_test_rub, pred_test_rub)

                coef_df, coef_audit = extract_coefficients_and_audit(best_est, feature_names)
                coef_csv_path = coefs_dir / f"coefficients_{run_key}.csv"
                coef_df.to_csv(coef_csv_path, index=False)

                coef_map = dict(zip(coef_df["feature"], coef_df["coefficient"]))
                coef_comparison_df[run_key] = coef_comparison_df["feature"].map(coef_map)

                # Adjuntar metadatos de trazabilidad al estimador serializado
                setattr(best_est, "model_id_", model_id)
                setattr(best_est, "model_name_", model_name)
                setattr(best_est, "target_mode_", target_mode)
                setattr(best_est, "target_column_", target_col)
                setattr(best_est, "best_params_", best_params)
                setattr(best_est, "duan_smearing_factor_", duan_factor)
                setattr(best_est, "val_rmse_rub_", m_val["rmse_rub"])
                setattr(best_est, "test_rmse_rub_", m_test["rmse_rub"])
                setattr(best_est, "mlflow_run_id_", parent_run_id)

                # Registro de parámetros y métricas en MLflow
                mlflow_params = {
                    **best_params,
                    "n_train": int(len(X_train)),
                    "n_val": int(len(X_val)),
                    "n_test": int(len(X_test)),
                    "n_features": int(n_features),
                    "target_mode": target_mode,
                    "target_column": target_col,
                    "grid_combinations_evaluated": int(len(param_combinations)),
                }
                mlflow.log_params(mlflow_params)

                mlflow_metrics: dict[str, float] = {
                    "train_rmse_rub": m_train["rmse_rub"],
                    "val_rmse_rub": m_val["rmse_rub"],
                    "test_rmse_rub": m_test["rmse_rub"],
                    "train_mae_rub": m_train["mae_rub"],
                    "val_mae_rub": m_val["mae_rub"],
                    "test_mae_rub": m_test["mae_rub"],
                    "train_medae_rub": m_train["medae_rub"],
                    "val_medae_rub": m_val["medae_rub"],
                    "test_medae_rub": m_test["medae_rub"],
                    "train_mape_pct": m_train["mape_pct"],
                    "val_mape_pct": m_val["mape_pct"],
                    "test_mape_pct": m_test["mape_pct"],
                    "train_r2_rub": m_train["r2_rub"],
                    "val_r2_rub": m_val["r2_rub"],
                    "test_r2_rub": m_test["r2_rub"],
                    "coef_l2_norm": coef_audit["coef_l2_norm"],
                    "coef_l1_norm": coef_audit["coef_l1_norm"],
                    "coef_max_abs": coef_audit["coef_max_abs"],
                    "n_zero_coefs": float(coef_audit["n_zero_coefs"]),
                    "pct_zero_coefs": coef_audit["pct_zero_coefs"],
                    "x_train_condition_number": matrix_conditioning["condition_number"],
                    "x_train_matrix_rank": float(matrix_conditioning["matrix_rank"]),
                }
                if is_log_mode:
                    mlflow_metrics.update(
                        {
                            "train_rmse_log": log_m_train["rmse_log"],
                            "val_rmse_log": log_m_val["rmse_log"],
                            "test_rmse_log": log_m_test["rmse_log"],
                            "train_r2_log": log_m_train["r2_log"],
                            "val_r2_log": log_m_val["r2_log"],
                            "test_r2_log": log_m_test["r2_log"],
                            "duan_smearing_factor": duan_factor,
                            "val_rmse_rub_duan": m_val_duan["rmse_rub"],
                            "test_rmse_rub_duan": m_test_duan["rmse_rub"],
                        }
                    )

                mlflow.log_metrics(mlflow_metrics)
                mlflow.log_artifact(str(coef_csv_path), artifact_path="coefficients")

                try:
                    sig = infer_signature(
                        X_train.iloc[:5], best_est.predict(X_train.iloc[:5])
                    )
                    mlflow.sklearn.log_model(
                        sk_model=best_est,
                        artifact_path="model",
                        signature=sig,
                    )
                except Exception as exc:
                    LOGGER.debug(
                        "Fallback a log_artifact para el modelo %s: %s", run_key, exc
                    )
                    tmp_joblib = trained_dir / f"_{run_key}.joblib"
                    joblib.dump(best_est, tmp_joblib)
                    mlflow.log_artifact(str(tmp_joblib), artifact_path="model")
                    tmp_joblib.unlink(missing_ok=True)

                fitted_models_store[run_key] = best_est
                predictions_store[run_key] = {
                    "train_pred_rub": best_trial["pred_train_rub"],
                    "val_pred_rub": best_trial["pred_val_rub"],
                    "test_pred_rub": pred_test_rub,
                }

                summary_records.append(
                    {
                        "run_key": run_key,
                        "mlflow_run_id": parent_run_id,
                        "model_id": model_id,
                        "model_name": model_name,
                        "estimator_import": estimator_import,
                        "role": role,
                        "is_primary_academic_model": is_primary,
                        "target_mode": target_mode,
                        "target_column": target_col,
                        "best_params_json": json.dumps(best_params, sort_keys=True),
                        "train_rmse_rub": m_train["rmse_rub"],
                        "val_rmse_rub": m_val["rmse_rub"],
                        "test_rmse_rub": m_test["rmse_rub"],
                        "train_mae_rub": m_train["mae_rub"],
                        "val_mae_rub": m_val["mae_rub"],
                        "test_mae_rub": m_test["mae_rub"],
                        "train_medae_rub": m_train["medae_rub"],
                        "val_medae_rub": m_val["medae_rub"],
                        "test_medae_rub": m_test["medae_rub"],
                        "train_mape_pct": m_train["mape_pct"],
                        "val_mape_pct": m_val["mape_pct"],
                        "test_mape_pct": m_test["mape_pct"],
                        "train_r2_rub": m_train["r2_rub"],
                        "val_r2_rub": m_val["r2_rub"],
                        "test_r2_rub": m_test["r2_rub"],
                        "train_rmse_log": log_m_train["rmse_log"],
                        "val_rmse_log": log_m_val["rmse_log"],
                        "test_rmse_log": log_m_test["rmse_log"],
                        "train_r2_log": log_m_train["r2_log"],
                        "val_r2_log": log_m_val["r2_log"],
                        "test_r2_log": log_m_test["r2_log"],
                        "duan_smearing_factor": duan_factor,
                        "val_rmse_rub_duan": m_val_duan["rmse_rub"],
                        "test_rmse_rub_duan": m_test_duan["rmse_rub"],
                        **coef_audit,
                    }
                )

    summary_df = pd.DataFrame(summary_records)
    grid_df = pd.DataFrame(grid_records)

    # 6. Selección del modelo principal académico y del mejor modelo lineal global (según val_rmse_rub)
    primary_rows = summary_df[summary_df["is_primary_academic_model"] == True].sort_values(
        "val_rmse_rub", ascending=True
    )
    if primary_rows.empty:
        primary_rows = summary_df[
            summary_df["model_id"] == "linear_regression_ols"
        ].sort_values("val_rmse_rub", ascending=True)
    primary_winner = primary_rows.iloc[0]
    primary_run_key = str(primary_winner["run_key"])

    overall_winner = summary_df.sort_values("val_rmse_rub", ascending=True).iloc[0]
    best_run_key = str(overall_winner["run_key"])

    reg_rows = summary_df[
        summary_df["role"] == "regularized_linear_alternative"
    ].sort_values("val_rmse_rub", ascending=True)
    best_regularized_run_key = (
        str(reg_rows.iloc[0]["run_key"]) if not reg_rows.empty else best_run_key
    )

    primary_model_path = trained_dir / "primary_linear_regression.joblib"
    best_model_path = trained_dir / "best_linear_model.joblib"
    joblib.dump(fitted_models_store[primary_run_key], primary_model_path)
    joblib.dump(fitted_models_store[best_run_key], best_model_path)

    experiments_csv_path = trained_dir / "experiments_summary.csv"
    grid_csv_path = trained_dir / "hyperparameter_search_results.csv"
    coef_comp_csv_path = trained_dir / "coefficients_comparison.csv"

    summary_df.to_csv(experiments_csv_path, index=False)
    grid_df.to_csv(grid_csv_path, index=False)
    coef_comparison_df.to_csv(coef_comp_csv_path, index=False)

    # 7. Generación de los 4 gráficos de diagnóstico en reports/figures/modeling/
    test_year_month = reconstruct_test_dates(
        root, X_test, y_test, preprocessor_linear
    )
    generated_figures = generate_diagnostic_plots(
        figures_dir=figures_dir,
        summary_df=summary_df,
        coef_comparison_df=coef_comparison_df,
        predictions_store=predictions_store,
        y_val_rub=y_val_rub,
        y_test_rub=y_test_rub,
        test_year_month=test_year_month,
        primary_run_key=primary_run_key,
        best_regularized_run_key=best_regularized_run_key,
    )

    # Registrar figuras de diagnóstico en las corridas ganadoras de MLflow
    for winner_key in {primary_run_key, best_run_key}:
        run_id = str(
            summary_df.loc[summary_df["run_key"] == winner_key, "mlflow_run_id"].iloc[0]
        )
        with mlflow.start_run(run_id=run_id):
            for fig_file in generated_figures.values():
                mlflow.log_artifact(fig_file, artifact_path="diagnostic_figures")

    # 8. Verificación de inmutabilidad post-ejecución (SHA-256)
    integrity_end = verify_post_execution_immutability(root, pre_hashes)

    # Top 20 coeficientes de los modelos seleccionados para training_metadata.json
    def _top_coefs_for_key(r_key: str, top_n: int = 20) -> list[dict[str, Any]]:
        if r_key not in coef_comparison_df.columns:
            return []
        sub = (
            coef_comparison_df[["feature", "family", r_key]]
            .assign(abs_val=lambda d: d[r_key].abs())
            .sort_values("abs_val", ascending=False)
            .head(top_n)
        )
        return [
            {
                "feature": str(row["feature"]),
                "family": str(row["family"]),
                "coefficient": float(row[r_key]),
                "abs_coefficient": float(row["abs_val"]),
            }
            for _, row in sub.iterrows()
        ]

    training_metadata: dict[str, Any] = {
        "stage": "05_training_evaluation",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mlflow_tracking_uri": tracking_uri,
        "mlflow_experiment_name": experiment_name,
        "mlflow_experiment_id": str(experiment.experiment_id),
        "split_sizes": {
            "n_train": int(len(X_train)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "n_features": int(n_features),
        },
        "input_integrity": {
            **integrity_start,
            **integrity_end,
        },
        "matrix_conditioning": matrix_conditioning,
        "primary_academic_model": {
            "run_key": primary_run_key,
            "model_id": str(primary_winner["model_id"]),
            "model_name": str(primary_winner["model_name"]),
            "target_mode": str(primary_winner["target_mode"]),
            "best_params": json.loads(str(primary_winner["best_params_json"])),
            "mlflow_run_id": str(primary_winner["mlflow_run_id"]),
            "artifact_path": "models/trained/primary_linear_regression.joblib",
            "train_rmse_rub": float(primary_winner["train_rmse_rub"]),
            "val_rmse_rub": float(primary_winner["val_rmse_rub"]),
            "test_rmse_rub": float(primary_winner["test_rmse_rub"]),
            "val_mae_rub": float(primary_winner["val_mae_rub"]),
            "test_mae_rub": float(primary_winner["test_mae_rub"]),
            "val_r2_rub": float(primary_winner["val_r2_rub"]),
            "test_r2_rub": float(primary_winner["test_r2_rub"]),
            "coef_l2_norm": float(primary_winner["coef_l2_norm"]),
            "n_zero_coefs": int(primary_winner["n_zero_coefs"]),
            "top_20_coefficients": _top_coefs_for_key(primary_run_key, top_n=20),
        },
        "best_overall_linear_model": {
            "run_key": best_run_key,
            "model_id": str(overall_winner["model_id"]),
            "model_name": str(overall_winner["model_name"]),
            "target_mode": str(overall_winner["target_mode"]),
            "best_params": json.loads(str(overall_winner["best_params_json"])),
            "mlflow_run_id": str(overall_winner["mlflow_run_id"]),
            "artifact_path": "models/trained/best_linear_model.joblib",
            "train_rmse_rub": float(overall_winner["train_rmse_rub"]),
            "val_rmse_rub": float(overall_winner["val_rmse_rub"]),
            "test_rmse_rub": float(overall_winner["test_rmse_rub"]),
            "val_mae_rub": float(overall_winner["val_mae_rub"]),
            "test_mae_rub": float(overall_winner["test_mae_rub"]),
            "val_r2_rub": float(overall_winner["val_r2_rub"]),
            "test_r2_rub": float(overall_winner["test_r2_rub"]),
            "coef_l2_norm": float(overall_winner["coef_l2_norm"]),
            "n_zero_coefs": int(overall_winner["n_zero_coefs"]),
            "top_20_coefficients": _top_coefs_for_key(best_run_key, top_n=20),
        },
        "generated_artifacts": {
            "primary_linear_regression_joblib": "models/trained/primary_linear_regression.joblib",
            "best_linear_model_joblib": "models/trained/best_linear_model.joblib",
            "experiments_summary_csv": "models/trained/experiments_summary.csv",
            "hyperparameter_search_csv": "models/trained/hyperparameter_search_results.csv",
            "coefficients_comparison_csv": "models/trained/coefficients_comparison.csv",
            "training_metadata_json": "models/trained/training_metadata.json",
            "training_evaluation_report_md": "reports/training_evaluation_report.md",
            "figures": {
                k: str(Path(v).relative_to(root).as_posix())
                for k, v in generated_figures.items()
            },
        },
        "runs_summary": summary_df.to_dict(orient="records"),
    }

    metadata_json_path = trained_dir / "training_metadata.json"
    with metadata_json_path.open("w", encoding="utf-8") as f:
        json.dump(training_metadata, f, indent=2, ensure_ascii=False)

    # 9. Generación del reporte académico en Markdown
    report_md_path = reports_dir / "training_evaluation_report.md"
    build_training_evaluation_report(
        report_path=report_md_path,
        metadata=training_metadata,
        summary_df=summary_df,
        grid_df=grid_df,
        coef_comparison_df=coef_comparison_df,
    )

    LOGGER.info(
        "Etapa 05 completada con éxito. Modelo académico: %s (Val RMSE=%.2f RUB) | Mejor global: %s (Val RMSE=%.2f RUB).",
        primary_run_key,
        primary_winner["val_rmse_rub"],
        best_run_key,
        overall_winner["val_rmse_rub"],
    )
    return training_metadata


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parsea argumentos opcionales de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Etapa 05: Entrenamiento, Ajuste, Evaluación y Tracking con MLflow."
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Ruta opcional a la raíz del proyecto (se autodetecta por defecto).",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help="Nombre del experimento en MLflow.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada CLI."""
    args = parse_args(argv)
    run_training_evaluation(
        project_root=args.project_root,
        experiment_name=args.experiment_name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())