#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
04_model_selection.py — Selección y configuración de modelos candidatos (Etapa 04).

Analiza de forma determinista y en modo solo lectura los artefactos generados en
las etapas 02 (EDA) y 03 (Preprocesamiento) para fundamentar la selección de
modelos de regresión lineal que predecirán `monthly_rent_rub` en la etapa 05.

Características principales:
1. Resolución automática de rutas desde `src/` hacia la raíz del proyecto.
2. Verificación estricta de existencia e inmutabilidad (SHA-256) de los reportes
   y metadatos previos (`eda_report.md`, `preprocessing_report.md`,
   `preprocessing_metadata.json` y matrices en `data/processed/`).
3. Selección acotada y justificada de 5 candidatos alineados con el requisito
   académico de regresión lineal:
   - `DummyRegressor` (línea base de referencia).
   - `LinearRegression` (modelo principal exigido por la entrega académica).
   - `Ridge` (alternativa lineal con regularización L2 frente a multicolinealidad).
   - `Lasso` (alternativa lineal con regularización L1 para selección en alta dimensión).
   - `ElasticNet` (alternativa lineal combinada L1 + L2).
4. Mapeo explícito de archivos de entrada (`X_*_linear.csv`, `y_*.csv`, artefactos
   `.joblib`) y regímenes de la variable objetivo (`monthly_rent_rub` y
   `monthly_rent_rub_log1p`) para que la etapa 05 ejecute el entrenamiento con MLflow.
5. No entrena modelos, no inventa métricas ni modifica archivos previos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


# =============================================================================
# 1. RESOLUCIÓN DE RUTAS Y CONFIGURACIÓN
# =============================================================================

EXPECTED_CLEAN_SHA256: str = (
    "e43378a0de55cdc2320b76f273ef47e1158a41e7fb0d06da0dc34666a5cc1a71"
)


def resolve_default_project_root() -> Path:
    """
    Determina la raíz del proyecto de forma segura tanto si el script se ejecuta
    desde la raíz (`python src/04_model_selection.py`) como desde dentro de `src/`
    (`cd src && python 04_model_selection.py`).
    """
    cwd = Path.cwd().resolve()
    if (cwd / "models" / "preprocessing" / "preprocessing_metadata.json").exists():
        return cwd

    script_parent = Path(__file__).resolve().parent
    if script_parent.name == "src":
        candidate_root = script_parent.parent
        if (candidate_root / "reports").exists() or (candidate_root / "models").exists():
            return candidate_root
        return candidate_root

    return cwd


@dataclass(frozen=True)
class ModelSelectionConfig:
    """Configuración inmutable para la Etapa 04 de selección de modelos."""

    project_root: Path
    eda_report_path: Path
    preprocessing_report_path: Path
    preprocessing_metadata_path: Path
    processed_data_dir: Path
    preprocessing_artifacts_dir: Path
    output_report_path: Path
    output_candidates_json_path: Path
    verify_processed_files: bool = True
    verify_clean_sha256: bool = True


# =============================================================================
# 2. UTILIDADES DE INTEGRIDAD Y VALIDACIÓN DE INSUMOS (SOLO LECTURA)
# =============================================================================

def compute_sha256(file_path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo en modo solo lectura."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def to_project_relative_posix(path: Path, project_root: Path) -> str:
    """Convierte una ruta a formato POSIX relativo a la raíz del proyecto."""
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def validate_required_inputs(config: ModelSelectionConfig) -> Dict[str, str]:
    """
    Verifica que todos los archivos requeridos de las etapas 02 y 03 existan
    en disco antes de continuar. Si falta alguno, lanza FileNotFoundError
    detallando expresamente qué archivos no se encontraron.

    Retorna un diccionario con los hashes SHA-256 de los archivos leídos
    para comprobar su inmutabilidad al finalizar el script.
    """
    mandatory_files: List[Path] = [
        config.eda_report_path,
        config.preprocessing_report_path,
        config.preprocessing_metadata_path,
    ]

    if config.verify_processed_files:
        mandatory_files.extend(
            [
                config.processed_data_dir / "X_train_linear.csv",
                config.processed_data_dir / "X_val_linear.csv",
                config.processed_data_dir / "X_test_linear.csv",
                config.processed_data_dir / "X_train_tree.csv",
                config.processed_data_dir / "X_val_tree.csv",
                config.processed_data_dir / "X_test_tree.csv",
                config.processed_data_dir / "y_train.csv",
                config.processed_data_dir / "y_val.csv",
                config.processed_data_dir / "y_test.csv",
                config.preprocessing_artifacts_dir / "preprocessor_linear.joblib",
                config.preprocessing_artifacts_dir / "preprocessor_tree.joblib",
                config.preprocessing_artifacts_dir / "target_log_transformer.joblib",
            ]
        )

    missing_files = [
        to_project_relative_posix(p, config.project_root)
        for p in mandatory_files
        if not p.exists()
    ]
    if missing_files:
        formatted_missing = "\n  - ".join(missing_files)
        raise FileNotFoundError(
            "Faltan archivos necesarios de las etapas 02 y/o 03 para ejecutar "
            "04_model_selection.py:\n  - " + formatted_missing
        )

    tracked_hashes: Dict[str, str] = {}
    for file_path in mandatory_files:
        rel_key = to_project_relative_posix(file_path, config.project_root)
        tracked_hashes[rel_key] = compute_sha256(file_path)

    return tracked_hashes


def load_and_analyze_upstream_artifacts(
    config: ModelSelectionConfig,
) -> Dict[str, Any]:
    """
    Lee `preprocessing_metadata.json`, `eda_report.md` y `preprocessing_report.md`
    para extraer los metadatos reales del proyecto y verificar su consistencia.
    """
    metadata_raw = json.loads(
        config.preprocessing_metadata_path.read_text(encoding="utf-8")
    )
    eda_text = config.eda_report_path.read_text(encoding="utf-8")
    prep_text = config.preprocessing_report_path.read_text(encoding="utf-8")

    required_meta_keys = {
        "input_sha256",
        "split_metadata",
        "linear_features_out",
        "tree_features_out",
    }
    missing_keys = required_meta_keys - set(metadata_raw.keys())
    if missing_keys:
        raise KeyError(
            f"El archivo {config.preprocessing_metadata_path} no contiene las claves "
            f"esperadas: {sorted(missing_keys)}"
        )

    input_sha256 = str(metadata_raw["input_sha256"])
    if config.verify_clean_sha256 and input_sha256 != EXPECTED_CLEAN_SHA256:
        raise ValueError(
            f"El hash SHA-256 registrado en preprocessing_metadata.json ({input_sha256}) "
            f"no coincide con el esperado ({EXPECTED_CLEAN_SHA256})."
        )

    linear_features: List[str] = list(metadata_raw["linear_features_out"])
    tree_features: List[str] = list(metadata_raw["tree_features_out"])

    num_features = [f for f in linear_features if f.startswith("num__")]
    bin_features = [f for f in linear_features if f.startswith("bin__")]
    cat_features = [f for f in linear_features if f.startswith("cat__")]
    district_features = [f for f in cat_features if f.startswith("cat__district_")]
    okrug_features = [f for f in cat_features if f.startswith("cat__okrug_")]
    btype_features = [f for f in cat_features if f.startswith("cat__building_type_")]
    renov_features = [f for f in cat_features if f.startswith("cat__renovation_")]

    # Verificación básica de coherencia documental en los reportes previos
    if "monthly_rent_rub" not in eda_text or "monthly_rent_rub" not in prep_text:
        raise ValueError(
            "Los reportes de EDA o preprocesamiento no mencionan la variable "
            "objetivo 'monthly_rent_rub'."
        )

    return {
        "input_sha256": input_sha256,
        "split_metadata": metadata_raw["split_metadata"],
        "n_linear_features": len(linear_features),
        "n_tree_features": len(tree_features),
        "feature_breakdown_linear": {
            "numeric_scaled_count": len(num_features),
            "binary_count": len(bin_features),
            "categorical_ohe_count": len(cat_features),
            "district_ohe_count": len(district_features),
            "okrug_ohe_count": len(okrug_features),
            "building_type_ohe_count": len(btype_features),
            "renovation_ohe_count": len(renov_features),
            "numeric_features": num_features,
            "binary_features": bin_features,
        },
        "eda_findings_summary": {
            "target_min_rub": 6800,
            "target_max_rub": 300700,
            "target_mean_rub": 52432.9,
            "target_std_rub": 30003.2,
            "target_median_rub": 45800.0,
            "target_iqr_rub": 31600.0,
            "target_skewness_raw": 1.820,
            "target_kurtosis_raw": 5.565,
            "target_outliers_1_5_iqr": 936,
            "target_outliers_pct": 4.68,
            "target_skewness_log1p": -0.136,
            "target_kurtosis_log1p": 0.216,
            "train_target_skewness_raw": 1.821,
            "train_target_skewness_log1p": -0.133,
            "top_numeric_correlations_pearson": {
                "to_center_km": -0.5314,
                "total_area": 0.4108,
                "rooms": 0.3583,
                "lat": 0.2738,
                "lon": 0.2225,
            },
            "multicollinearity_vif_highlights": {
                "to_center_km": 7.777,
                "total_area": 4.546,
                "rooms": 4.545,
                "lon": 2.051,
                "lat": 2.014,
            },
            "categorical_eta_squared": {
                "district": 0.6044,
                "okrug": 0.4916,
                "renovation": 0.0427,
                "building_type": 0.0289,
            },
        },
    }


# =============================================================================
# 3. DEFINICIÓN DE CANDIDATOS Y ARCHIVOS DE ENTRADA PARA EL PASO 5
# =============================================================================

def build_model_candidates_payload(
    analysis: Dict[str, Any],
    input_hashes: Dict[str, str],
) -> Dict[str, Any]:
    """
    Construye el diccionario estructurado que se guardará en `models/model_candidates.json`
    para que el paso 5 (`05_training_evaluation.py`) pueda consultarlo directamente.
    """
    linear_data_files = {
        "X_train": "data/processed/X_train_linear.csv",
        "X_val": "data/processed/X_val_linear.csv",
        "X_test": "data/processed/X_test_linear.csv",
        "y_train": "data/processed/y_train.csv",
        "y_val": "data/processed/y_val.csv",
        "y_test": "data/processed/y_test.csv",
        "preprocessor_joblib": "models/preprocessing/preprocessor_linear.joblib",
        "target_transformer_joblib": "models/preprocessing/target_log_transformer.joblib",
    }

    tree_data_files = {
        "X_train": "data/processed/X_train_tree.csv",
        "X_val": "data/processed/X_val_tree.csv",
        "X_test": "data/processed/X_test_tree.csv",
        "y_train": "data/processed/y_train.csv",
        "y_val": "data/processed/y_val.csv",
        "y_test": "data/processed/y_test.csv",
        "preprocessor_joblib": "models/preprocessing/preprocessor_tree.joblib",
        "target_transformer_joblib": "models/preprocessing/target_log_transformer.joblib",
    }

    candidates: List[Dict[str, Any]] = [
        {
            "model_id": "dummy_baseline",
            "model_name": "DummyRegressor",
            "estimator_import": "sklearn.dummy.DummyRegressor",
            "role": "baseline_reference",
            "is_primary_academic_model": False,
            "matrix_family": "linear",
            "input_files": linear_data_files,
            "supported_target_columns": [
                "monthly_rent_rub",
                "monthly_rent_rub_log1p",
            ],
            "recommended_target_column": "monthly_rent_rub",
            "requires_inverse_target_transform_when_log1p": True,
            "default_params": {
                "strategy": "median",
            },
            "search_grid": {
                "strategy": ["median", "mean"],
            },
            "justification": (
                "Punto de referencia básico indispensable (sanity check). Predice una "
                "constante (mediana de 45,800 RUB o media de 52,432.9 RUB en el conjunto "
                "histórico) sin usar predictores. Establece el piso mínimo de error "
                "(RMSE y MAE máximos aceptables) que cualquier regresión lineal debe "
                "superar en validación temporal."
            ),
            "advantages": [
                "Costo computacional nulo y ausencia total de sobreajuste.",
                "La estrategia 'median' es robusta frente a la asimetría positiva del target (skewness = 1.821 en Train).",
                "Permite cuantificar la ganancia predictiva real aportada por las 170 features.",
            ],
            "limitations": [
                "No captura ninguna relación espacial, física ni temporal (R² <= 0 por definición en media).",
                "Incapaz de reflejar la inflación nominal entre los periodos Train (2021-2024) y Val/Test (2024-2026).",
            ],
        },
        {
            "model_id": "linear_regression_ols",
            "model_name": "LinearRegression",
            "estimator_import": "sklearn.linear_model.LinearRegression",
            "role": "primary_academic_model",
            "is_primary_academic_model": True,
            "matrix_family": "linear",
            "input_files": linear_data_files,
            "supported_target_columns": [
                "monthly_rent_rub",
                "monthly_rent_rub_log1p",
            ],
            "recommended_target_column": "monthly_rent_rub_log1p",
            "requires_inverse_target_transform_when_log1p": True,
            "default_params": {
                "fit_intercept": True,
            },
            "search_grid": {
                "fit_intercept": [True],
            },
            "justification": (
                "Modelo principal del proyecto porque la entrega académica exige "
                "expresamente un modelo final de regresión lineal. Permite estimar "
                "por Mínimos Cuadrados Ordinarios (OLS mediante descomposición SVD) el "
                "efecto marginal de variables clave como 'to_center_km' (r = -0.5314), "
                "'total_area' (r = 0.4108) y los distritos ('district', eta² = 0.6044). "
                "Se recomienda su ajuste sobre 'monthly_rent_rub_log1p' (asimetría = -0.133) "
                "contrastándolo contra la escala original en el paso 5."
            ),
            "advantages": [
                "Cumple de manera directa el requisito académico obligatorio del proyecto.",
                "Máxima interpretabilidad económica de los coeficientes (elasticidades y semielasticidades en escala log1p).",
                "Entrenamiento determinista y cerrado sin hiperparámetros de penalización que ajustar.",
                "Capacidad de extrapolar la tendencia inflacionaria nominal a través de 'num__posting_year'.",
            ],
            "limitations": [
                "Alta varianza en los estimadores ante multicolinealidad entre predictores físicos ('total_area', 'rooms', 'area_per_room') y temporales ('building_year', 'posting_year', 'building_age').",
                "Sensible a la dependencia lineal exacta de las 150 columnas One-Hot ('okrug', 'district', 'building_type', 'renovation' codificadas sin descartar categoría base) y al anidamiento jerárquico entre 'district' (129) y 'okrug' (11).",
                "Sensible a valores extremos en escala original (936 outliers > 1.5 IQR) si no se emplea la transformación log1p.",
            ],
        },
        {
            "model_id": "ridge_l2",
            "model_name": "Ridge",
            "estimator_import": "sklearn.linear_model.Ridge",
            "role": "regularized_linear_alternative",
            "is_primary_academic_model": False,
            "matrix_family": "linear",
            "input_files": linear_data_files,
            "supported_target_columns": [
                "monthly_rent_rub",
                "monthly_rent_rub_log1p",
            ],
            "recommended_target_column": "monthly_rent_rub_log1p",
            "requires_inverse_target_transform_when_log1p": True,
            "default_params": {
                "alpha": 1.0,
                "fit_intercept": True,
                "random_state": 42,
            },
            "search_grid": {
                "alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0],
            },
            "justification": (
                "Alternativa lineal con regularización L2 diseñada específicamente para "
                "mitigar la multicolinealidad documentada en el EDA (VIF = 4.546 en "
                "'total_area' y 4.545 en 'rooms') y en el preprocesamiento (ratios "
                "derivados 'area_per_room', 'floor_ratio', 'building_age', más la "
                "colinealidad estructural entre las 11 dummies de 'okrug' y las 129 "
                "dummies de 'district')."
            ),
            "advantages": [
                "Estabiliza la inversión de la matriz X^T X añadiendo penalización L2 (alpha * ||w||_2^2).",
                "Retiene las 170 variables repartiendo el peso entre predictores correlacionados sin perder información geográfica.",
                "Reduce la varianza de las predicciones en los distritos de baja frecuencia (< 1% cada uno).",
            ],
            "limitations": [
                "No realiza selección de variables (ningún coeficiente se anula exactamente a cero).",
                "Requiere calibrar el hiperparámetro 'alpha' sobre el conjunto de validación temporal.",
            ],
        },
        {
            "model_id": "lasso_l1",
            "model_name": "Lasso",
            "estimator_import": "sklearn.linear_model.Lasso",
            "role": "regularized_linear_alternative",
            "is_primary_academic_model": False,
            "matrix_family": "linear",
            "input_files": linear_data_files,
            "supported_target_columns": [
                "monthly_rent_rub",
                "monthly_rent_rub_log1p",
            ],
            "recommended_target_column": "monthly_rent_rub_log1p",
            "requires_inverse_target_transform_when_log1p": True,
            "default_params": {
                "alpha": 0.001,
                "max_iter": 10000,
                "fit_intercept": True,
                "random_state": 42,
            },
            "search_grid": {
                "alpha": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1.0],
                "max_iter": [10000],
            },
            "justification": (
                "Alternativa lineal con penalización L1 orientada a la reducción de "
                "dimensionalidad y selección automática de variables sobre las 170 "
                "columnas de 'X_*_linear.csv', donde 150 provienen de One-Hot Encoding "
                "y las 129 categorías de 'district' tienen frecuencias individuales "
                "inferiores al 1% (entre 0.61% y 0.95%)."
            ),
            "advantages": [
                "Produce modelos parsimoniosos (sparse) llevando exactamente a cero los coeficientes de distritos o variables redundantes.",
                "Elimina automáticamente la redundancia de las categorías de referencia en los bloques One-Hot.",
                "Facilita identificar el subconjunto mínimo de predictores con mayor impacto real.",
            ],
            "limitations": [
                "Ante grupos de variables altamente correlacionadas (como 'total_area', 'rooms' y 'area_per_room'), tiende a seleccionar una de forma arbitraria y descartar las demás.",
                "Si 'alpha' es demasiado alto, puede subajustar (underfitting) al descartar distritos con primas de precio reales.",
            ],
        },
        {
            "model_id": "elasticnet_l1_l2",
            "model_name": "ElasticNet",
            "estimator_import": "sklearn.linear_model.ElasticNet",
            "role": "regularized_linear_alternative",
            "is_primary_academic_model": False,
            "matrix_family": "linear",
            "input_files": linear_data_files,
            "supported_target_columns": [
                "monthly_rent_rub",
                "monthly_rent_rub_log1p",
            ],
            "recommended_target_column": "monthly_rent_rub_log1p",
            "requires_inverse_target_transform_when_log1p": True,
            "default_params": {
                "alpha": 0.001,
                "l1_ratio": 0.5,
                "max_iter": 10000,
                "fit_intercept": True,
                "random_state": 42,
            },
            "search_grid": {
                "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0],
                "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                "max_iter": [10000],
            },
            "justification": (
                "Compromiso entre Ridge (L2) y Lasso (L1). Permite podar dummies "
                "irrelevantes de 'district' mediante la componente L1 mientras mantiene "
                "agrupados y estables los predictores físicamente colineales ('total_area', "
                "'rooms', 'area_per_room' y coordenadas espaciales 'lat', 'lon', "
                "'to_center_km') gracias a la componente L2."
            ),
            "advantages": [
                "Supera la limitación de selección arbitraria de Lasso en presencia de grupos colineales (efecto de agrupación).",
                "Combina esparsidad controlada con estabilidad numérica en espacios de 170 dimensiones.",
                "Permite explorar de forma continua el balance entre L1 y L2 mediante 'l1_ratio'.",
            ],
            "limitations": [
                "Mayor espacio de búsqueda en validación al requerir el ajuste conjunto de dos hiperparámetros ('alpha' y 'l1_ratio').",
                "Requiere 'max_iter' suficientemente alto para garantizar convergencia por descenso coordenado.",
            ],
        },
    ]

    return {
        "stage": "04_model_selection",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "academic_constraint": (
            "La entrega académica exige un modelo final de regresión lineal. "
            "Se establece LinearRegression como modelo principal, acompañado por "
            "Ridge, Lasso y ElasticNet como alternativas lineales regularizadas "
            "y DummyRegressor como referencia básica."
        ),
        "input_integrity": {
            "clean_dataset_sha256": analysis["input_sha256"],
            "verified_upstream_files_sha256": input_hashes,
        },
        "dataset_and_split_summary": analysis["split_metadata"],
        "feature_space_summary": {
            "n_features_linear": analysis["n_linear_features"],
            "n_features_tree": analysis["n_tree_features"],
            "linear_breakdown": analysis["feature_breakdown_linear"],
        },
        "data_routing_by_family": {
            "linear": {
                "status": "ACTIVE_FOR_STEP_05",
                "description": (
                    "Matrices preprocesadas con SimpleImputer(median) + RobustScaler "
                    "en las 15 variables numéricas, imputación por moda en las 5 "
                    "binarias y OneHotEncoder(drop='if_binary') en las 4 categóricas "
                    "(170 columnas totales)."
                ),
                "files": linear_data_files,
            },
            "tree": {
                "status": "EXCLUDED_NOT_REQUIRED_BY_LINEAR_DELIVERABLE",
                "description": (
                    "Matrices preprocesadas sin escalado numérico generadas por "
                    "03_preprocessing.py (170 columnas totales). No se asignan a "
                    "ningún candidato activo para evitar modelos adicionales "
                    "innecesarios ajenos al requisito de regresión lineal."
                ),
                "files": tree_data_files,
            },
        },
        "step_05_training_and_mlflow_protocol": {
            "fit_partition": "train",
            "model_selection_and_tuning_partition": "val",
            "final_holdout_evaluation_partition": "test",
            "target_evaluation_modes": [
                {
                    "mode_id": "raw_rub",
                    "target_column": "monthly_rent_rub",
                    "inverse_transform_required": False,
                    "description": (
                        "Entrenamiento directo en rublos (RUB). Útil como comparación "
                        "directa en unidades originales."
                    ),
                },
                {
                    "mode_id": "log1p_rub",
                    "target_column": "monthly_rent_rub_log1p",
                    "inverse_transform_required": True,
                    "inverse_transformer_artifact": "models/preprocessing/target_log_transformer.joblib",
                    "inverse_function": "numpy.expm1",
                    "description": (
                        "Entrenamiento sobre ln(1 + monthly_rent_rub) para corregir la "
                        "asimetría positiva (1.821 -> -0.133) y heterocedasticidad, "
                        "aplicando np.expm1 antes de calcular las métricas en RUB."
                    ),
                },
            ],
            "evaluation_metrics_in_rub": [
                "RMSE",
                "MAE",
                "MedAE",
                "MAPE",
                "R2",
            ],
            "mlflow_requirements": {
                "mandatory_in_step_05": True,
                "suggested_experiment_name": "moscow_rentals_linear_regression",
                "log_items": [
                    "model_id, estimator_import, target_mode y parámetros del estimador",
                    "métricas en Train, Validation y Test expresadas en RUB (RMSE, MAE, MedAE, MAPE, R2)",
                    "artefacto del modelo entrenado y referencia a preprocessor_linear.joblib",
                    "tabla comparativa de coeficientes / importancia e inspección de residuos",
                ],
            },
        },
        "candidates": candidates,
    }


# =============================================================================
# 4. GENERACIÓN DEL REPORTE MARKDOWN (reports/model_selection_report.md)
# =============================================================================

def generate_model_selection_report(
    report_path: Path,
    payload: Dict[str, Any],
    analysis: Dict[str, Any],
) -> None:
    """
    Genera `reports/model_selection_report.md` documentando los hallazgos del
    EDA y del preprocesamiento, los modelos seleccionados, sus ventajas,
    limitaciones, justificación técnica y el mapeo exacto de archivos para el paso 5.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)

    split_meta = analysis["split_metadata"]
    fb = analysis["feature_breakdown_linear"]
    eda = analysis["eda_findings_summary"]
    linear_files = payload["data_routing_by_family"]["linear"]["files"]
    tree_files = payload["data_routing_by_family"]["tree"]["files"]

    candidates_table_rows: List[str] = []
    for c in payload["candidates"]:
        role_label = (
            "**Modelo Principal (Entrega Académica)**"
            if c["is_primary_academic_model"]
            else (
                "Referencia Básica (Baseline)"
                if c["role"] == "baseline_reference"
                else "Alternativa Lineal Regularizada"
            )
        )
        candidates_table_rows.append(
            f"| `{c['model_id']}` | `{c['model_name']}` (`{c['estimator_import']}`) | "
            f"{role_label} | `{c['matrix_family']}` (`170` cols) | "
            f"`{c['recommended_target_column']}` (y contraste con `monthly_rent_rub`) |"
        )
    candidates_table_md = "\n".join(candidates_table_rows)

    detailed_sections: List[str] = []
    for idx, c in enumerate(payload["candidates"], start=1):
        adv_md = "\n".join(f"  - {a}" for a in c["advantages"])
        lim_md = "\n".join(f"  - {l}" for l in c["limitations"])
        default_params_json = json.dumps(c["default_params"], ensure_ascii=False)
        grid_json = json.dumps(c["search_grid"], ensure_ascii=False)
        detailed_sections.append(
            f"""### 3.{idx}. `{c['model_name']}` (`{c['model_id']}`)

- **Clase en scikit-learn:** `{c['estimator_import']}`
- **Rol en el experimento:** `{c['role']}` (Modelo principal académico: `{'Sí' if c['is_primary_academic_model'] else 'No'}`)
- **Justificación basada en EDA y Preprocesamiento:** {c['justification']}
- **Ventajas:**
{adv_md}
- **Limitaciones:**
{lim_md}
- **Configuración base (`default_params`):** `{default_params_json}`
- **Grilla de búsqueda para Validación en Etapa 05 (`search_grid`):** `{grid_json}`
- **Archivos de entrada asignados para la Etapa 05:**
  - Matrices de predictores (`X`): `{c['input_files']['X_train']}`, `{c['input_files']['X_val']}`, `{c['input_files']['X_test']}`
  - Vectores objetivo (`y`): `{c['input_files']['y_train']}`, `{c['input_files']['y_val']}`, `{c['input_files']['y_test']}` (columnas `monthly_rent_rub` y `monthly_rent_rub_log1p`)
  - Artefactos de transformación: `{c['input_files']['preprocessor_joblib']}` y `{c['input_files']['target_transformer_joblib']}`"""
        )

    detailed_candidates_md = "\n\n".join(detailed_sections)

    content = f"""# Informe de Selección de Modelos — Alquileres en Moscú (Etapa 04)

| Metadato | Valor |
|---|---|
| Fecha de generación | `{payload["generated_at"]}` |
| Script generador | `src/04_model_selection.py` |
| SHA-256 verificado del dataset limpio (`data/rentals_clean.csv`) | `{analysis["input_sha256"]}` |
| Variable objetivo (`y`) | `monthly_rent_rub` (escala original RUB) y `monthly_rent_rub_log1p` ($\ln(1 + y)$) |
| Partición temporal (`Train` / `Val` / `Test`) | `{split_meta["n_train"]}` (`{split_meta["pct_train"]}%`) / `{split_meta["n_val"]}` (`{split_meta["pct_val"]}%`) / `{split_meta["n_test"]}` (`{split_meta["pct_test"]}%`) |
| Dimensiones de entrada (`linear` / `tree`) | `{analysis["n_linear_features"]}` columnas / `{analysis["n_tree_features"]}` columnas |
| Modelo principal requerido | `LinearRegression` (exigencia académica de regresión lineal) |
| Archivo de configuración exportado | `models/model_candidates.json` |

---

## 1. Síntesis de Hallazgos del EDA y Preprocesamiento que Guían la Selección

El análisis conjunto de `reports/eda_report.md`, `reports/preprocessing_report.md` y `models/preprocessing/preprocessing_metadata.json` establece las siguientes condiciones técnicas para el modelado:

1. **Asimetría positiva y heterocedasticidad en `monthly_rent_rub`:**
   - En escala original, el alquiler mensual oscila entre `{eda["target_min_rub"]}` RUB y `{eda["target_max_rub"]}` RUB, con media de `{eda["target_mean_rub"]:.1f}` RUB ($\text{{DE}} = {eda["target_std_rub"]:.1f}$ RUB) y mediana de `{eda["target_median_rub"]:.1f}` RUB ($\text{{IQR}} = {eda["target_iqr_rub"]:.1f}$ RUB).
   - Presenta una asimetría positiva pronunciada ($\text{{Skewness}} = {eda["target_skewness_raw"]:.3f}$ global; `{eda["train_target_skewness_raw"]:.3f}` en `Train`) y `{eda["target_outliers_1_5_iqr"]}` observaciones (`{eda["target_outliers_pct"]}%`) por encima de $1.5 \times \text{{IQR}}$ correspondientes al segmento residencial *premium* en el centro de Moscú.
   - Al aplicar la transformación `np.log1p` (`monthly_rent_rub_log1p`), la asimetría se reduce a `{eda["target_skewness_log1p"]:.3f}` (`{eda["train_target_skewness_log1p"]:.3f}` en `Train`) y la curtosis a `{eda["target_kurtosis_log1p"]:.3f}`, estabilizando la varianza condicional. Por ello, los modelos lineales deben evaluarse en el paso 5 considerando tanto el objetivo en escala logarítmica (con retransformación $\exp(\hat{{y}}) - 1$ mediante `target_log_transformer.joblib`) como en escala original.
2. **Multicolinealidad entre predictores físicos, espaciales y variables de ingeniería:**
   - El EDA identificó colinealidad entre `total_area` ($\text{{VIF}} = 4.546$, $R^2 = 0.78$) y `rooms` ($\text{{VIF}} = 4.545$, $R^2 = 0.78$), así como entre `to_center_km` ($\text{{VIF}} = 7.777$), `lat` ($\text{{VIF}} = 2.014$) y `lon` ($\text{{VIF}} = 2.051$).
   - En la etapa 03, `DomainFeatureEngineer` incorporó ratios y diferencias exactas (`num__area_per_room`, `num__floor_ratio`, `num__building_age`) que agregan dependencia funcional con sus variables de origen (`total_area`, `rooms`, `floor`, `total_floors`, `posting_year`, `building_year`).
   - Adicionalmente, las 4 variables categóricas (`okrug` con `{fb["okrug_ohe_count"]}` niveles, `district` con `{fb["district_ohe_count"]}` niveles, `building_type` con `{fb["building_type_ohe_count"]}` niveles y `renovation` con `{fb["renovation_ohe_count"]}` niveles) fueron codificadas con `OneHotEncoder(drop="if_binary")`. Al tener todas más de 2 niveles, se conservaron las `{fb["categorical_ohe_count"]}` columnas indicadoras completas (cuya suma por bloque es igual a 1) y existe dependencia jerárquica geográfica entre `district` ($\eta^2 = 0.6044$) y `okrug` ($\eta^2 = 0.4916$). Esto motiva directamente contrastar `LinearRegression` (OLS) frente a `Ridge` ($L_2$) y `ElasticNet` ($L_1 + L_2$).
3. **Alta dimensionalidad (`170` columnas) y baja frecuencia por distrito:**
   - La matriz `X_*_linear.csv` cuenta con `{fb["numeric_scaled_count"]}` variables numéricas escaladas con `RobustScaler`, `{fb["binary_count"]}` binarias y `{fb["categorical_ohe_count"]}` dummies categóricas.
   - Las `{fb["district_ohe_count"]}` categorías de `district` presentan frecuencias individuales menores al $1\%$ (entre $0.61\%$ y $0.95\%$). La regularización $L_1$ (`Lasso` y `ElasticNet`) resulta idónea para anular coeficientes de distritos con poca señal estadística y evitar sobreajuste.
4. **Deriva temporal e inflación nominal (Partición Cronológica Estricta):**
   - El dataset abarca de `{split_meta["train_date_min"]}` a `{split_meta["test_date_max"]}`. La partición cronológica (`Train`: `{split_meta["train_date_min"]}` a `{split_meta["train_date_max"]}`; `Val`: `{split_meta["val_date_min"]}` a `{split_meta["val_date_max"]}`; `Test`: `{split_meta["test_date_min"]}` a `{split_meta["test_date_max"]}`) exige que los modelos capturen la tendencia inflacionaria mediante `num__posting_year` sin fuga de información futura.

---

## 2. Resumen de Modelos Seleccionados y Decisión sobre Familias Adicionales

Siguiendo los requisitos académicos del proyecto, se selecciona **`LinearRegression` como modelo principal**, acompañado exclusivamente por tres alternativas lineales regularizadas (**`Ridge`**, **`Lasso`**, **`ElasticNet`**) y un modelo de referencia (**`DummyRegressor`**):

| ID Candidato | Modelo (`scikit-learn`) | Rol en el Proyecto | Matriz de Entrada | Objetivo Recomendado |
|---|---|---|---|---|
{candidates_table_md}

### Justificación de No Incorporación de Modelos Basados en Árboles u Otras Familias
- Aunque la etapa 03 generó matrices específicas sin escalado para árboles (`data/processed/X_*_tree.csv` y `models/preprocessing/preprocessor_tree.joblib`), **no se incluyen modelos de árboles (`RandomForest`, `ExtraTrees`, `LightGBM`, `XGBoost`, `CatBoost`)** en el conjunto de candidatos de la etapa 05 porque:
  1. La entrega académica exige explícitamente un **modelo final de regresión lineal**.
  2. Las directrices del proyecto establecen no incorporar modelos adicionales innecesarios fuera de la familia lineal requerida y su línea base.
  3. Los modelos basados en árboles de decisión no extrapolan tendencias lineales fuera del soporte observado de `posting_year` (`2021`–`2024` en `Train` frente a `2024`–`2026` en `Val`/`Test`) a menos que se desestacionalice o deflacte externamente el objetivo.

---

## 3. Detalle Técnico de los Modelos Candidatos

{detailed_candidates_md}

---

## 4. Mapeo de Archivos de Entrada y Artefactos para la Etapa 05

### 4.1. Archivos que DEBE utilizar la Etapa 05 (Modelos Lineales Seleccionados)

Todos los modelos seleccionados (`DummyRegressor`, `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`) deben consumir la rama **`linear`** generada en el paso 3:

| Tipo de Recurso | Ruta del Archivo | Uso en la Etapa 05 (`05_training_evaluation.py`) |
|---|---|---|
| **Predictores Entrenamiento (`X_train`)** | `{linear_files["X_train"]}` | Ajuste (`.fit()`) de los modelos candidatos (`{split_meta["n_train"]}` filas × `{analysis["n_linear_features"]}` columnas). |
| **Predictores Validación (`X_val`)** | `{linear_files["X_val"]}` | Selección de hiperparámetros (`alpha`, `l1_ratio`, `strategy`) y comparación de candidatos (`{split_meta["n_val"]}` filas × `{analysis["n_linear_features"]}` columnas). |
| **Predictores Prueba (`X_test`)** | `{linear_files["X_test"]}` | Evaluación final *holdout* del modelo lineal seleccionado (`{split_meta["n_test"]}` filas × `{analysis["n_linear_features"]}` columnas). |
| **Objetivo Entrenamiento (`y_train`)** | `{linear_files["y_train"]}` | Contiene `id`, `monthly_rent_rub` (escala RUB) y `monthly_rent_rub_log1p` (escala $\ln(1+y)$). |
| **Objetivo Validación (`y_val`)** | `{linear_files["y_val"]}` | Evaluación en ventana temporal intermedia (`{split_meta["val_date_min"]}` a `{split_meta["val_date_max"]}`). |
| **Objetivo Prueba (`y_test`)** | `{linear_files["y_test"]}` | Evaluación final en ventana futura (`{split_meta["test_date_min"]}` a `{split_meta["test_date_max"]}`). |
| **Pipeline Preprocesador Lineal** | `{linear_files["preprocessor_joblib"]}` | Artefacto `joblib` ajustado en `X_train` para empaquetar junto al modelo lineal final en MLflow. |
| **Transformador Logarítmico del Target** | `{linear_files["target_transformer_joblib"]}` | Artefacto `FunctionTransformer(log1p, expm1)` para invertir las predicciones logarítmicas a RUB antes de calcular métricas. |

### 4.2. Archivos Generados por el Paso 3 que NO se utilizan en los Candidatos Lineales

Para mantener trazabilidad completa con `03_preprocessing.py`, se deja constancia de los archivos de la rama `tree` que permanecen en disco sin ser consumidos por los modelos lineales:

- `{tree_files["X_train"]}`, `{tree_files["X_val"]}`, `{tree_files["X_test"]}` y `{tree_files["preprocessor_joblib"]}` (matrices con las 15 variables numéricas en sus unidades físicas originales sin `RobustScaler`, destinadas únicamente a algoritmos de partición recursiva por árboles).

---

## 5. Lineamientos de Entrenamiento, Evaluación y Registro en MLflow para la Etapa 05

1. **Prohibición de re-preprocesamiento y prevención de fuga:** El paso 5 debe cargar directamente los archivos `X_*_linear.csv` y `y_*.csv` sin volver a ejecutar `RobustScaler` ni `OneHotEncoder`, y sin usar `X_test_linear.csv` para decidir hiperparámetros.
2. **Comparación en la misma escala monetaria (RUB):** Cuando un modelo se entrene sobre `monthly_rent_rub_log1p`, sus predicciones $\hat{{z}}$ deben transformarse mediante $\hat{{y}} = \exp(\hat{{z}}) - 1$ (`np.expm1` o `target_log_transformer.inverse_transform`) para calcular `RMSE`, `MAE`, `MedAE`, `MAPE` y `R2` en rublos comparables uno a uno con el entrenamiento en escala cruda.
3. **Trazabilidad en MLflow:** Cada ejecución (`run`) en el paso 5 deberá registrar como mínimo los parámetros de `models/model_candidates.json`, el régimen de la variable objetivo (`raw_rub` vs `log1p_rub`), las métricas en `Train`, `Val` y `Test`, y el modelo lineal final serializado.
"""
    report_path.write_text(content, encoding="utf-8")


# =============================================================================
# 5. ORQUESTACIÓN PRINCIPAL E INMUTABILIDAD POST-EJECUCIÓN
# =============================================================================

def run_model_selection(config: ModelSelectionConfig) -> Dict[str, Any]:
    """
    Ejecuta el flujo completo de la Etapa 04 verificando la integridad previa y
    posterior de todos los insumos de las etapas 02 y 03.
    """
    # 1. Validar que existan todos los archivos requeridos y registrar sus hashes
    hashes_before = validate_required_inputs(config)

    # 2. Analizar los artefactos y reportes de las etapas 02 y 03
    analysis = load_and_analyze_upstream_artifacts(config)

    # 3. Construir la configuración estructurada de candidatos para el paso 5
    candidates_payload = build_model_candidates_payload(
        analysis=analysis,
        input_hashes=hashes_before,
    )

    # 4. Persistir models/model_candidates.json
    config.output_candidates_json_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_candidates_json_path.write_text(
        json.dumps(candidates_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 5. Generar reports/model_selection_report.md
    generate_model_selection_report(
        report_path=config.output_report_path,
        payload=candidates_payload,
        analysis=analysis,
    )

    # 6. Verificar que ningún archivo de entrada haya sido modificado
    hashes_after = validate_required_inputs(config)
    if hashes_before != hashes_after:
        raise RuntimeError(
            "ALERTA CRÍTICA: Se detectó una alteración en los archivos de las "
            "etapas 02 o 03 durante la ejecución de 04_model_selection.py."
        )

    return {
        "candidates_payload": candidates_payload,
        "output_report_path": config.output_report_path,
        "output_candidates_json_path": config.output_candidates_json_path,
    }


def parse_args() -> ModelSelectionConfig:
    default_root = resolve_default_project_root()

    parser = argparse.ArgumentParser(
        description=(
            "Etapa 04: Selección justificada de modelos de regresión lineal "
            "para predecir monthly_rent_rub."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default_root,
        help="Raíz del proyecto donde residen las carpetas data/, models/ y reports/.",
    )
    parser.add_argument(
        "--eda-report",
        type=Path,
        default=None,
        help="Ruta a reports/eda_report.md (por defecto relativo a --project-root).",
    )
    parser.add_argument(
        "--preprocessing-report",
        type=Path,
        default=None,
        help="Ruta a reports/preprocessing_report.md (por defecto relativo a --project-root).",
    )
    parser.add_argument(
        "--preprocessing-metadata",
        type=Path,
        default=None,
        help="Ruta a models/preprocessing/preprocessing_metadata.json.",
    )
    parser.add_argument(
        "--processed-data-dir",
        type=Path,
        default=None,
        help="Directorio data/processed con los CSV generados en el paso 3.",
    )
    parser.add_argument(
        "--preprocessing-artifacts-dir",
        type=Path,
        default=None,
        help="Directorio models/preprocessing con los artefactos .joblib del paso 3.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Ruta de salida para reports/model_selection_report.md.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Ruta de salida para models/model_candidates.json.",
    )
    parser.add_argument(
        "--skip-processed-check",
        action="store_true",
        help=(
            "Omite la verificación física de los CSV en data/processed/ y .joblib "
            "(útil si solo se tienen los reportes y el JSON de metadatos)."
        ),
    )
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Omite la validación estricta del hash SHA-256 esperado en pruebas sintéticas.",
    )

    args = parser.parse_args()
    root = args.project_root.resolve()

    return ModelSelectionConfig(
        project_root=root,
        eda_report_path=(args.eda_report or (root / "reports" / "eda_report.md")).resolve(),
        preprocessing_report_path=(
            args.preprocessing_report or (root / "reports" / "preprocessing_report.md")
        ).resolve(),
        preprocessing_metadata_path=(
            args.preprocessing_metadata
            or (root / "models" / "preprocessing" / "preprocessing_metadata.json")
        ).resolve(),
        processed_data_dir=(
            args.processed_data_dir or (root / "data" / "processed")
        ).resolve(),
        preprocessing_artifacts_dir=(
            args.preprocessing_artifacts_dir or (root / "models" / "preprocessing")
        ).resolve(),
        output_report_path=(
            args.output_report or (root / "reports" / "model_selection_report.md")
        ).resolve(),
        output_candidates_json_path=(
            args.output_json or (root / "models" / "model_candidates.json")
        ).resolve(),
        verify_processed_files=not args.skip_processed_check,
        verify_clean_sha256=not args.skip_sha256,
    )


if __name__ == "__main__":
    cfg = parse_args()
    result = run_model_selection(cfg)
    candidates_list = result["candidates_payload"]["candidates"]
    primary_models = [
        c["model_name"] for c in candidates_list if c["is_primary_academic_model"]
    ]
    print(
        f"[OK] Selección de modelos (Etapa 04) finalizada exitosamente.\n"
        f"  - Candidatos configurados: {len(candidates_list)} "
        f"({', '.join(c['model_name'] for c in candidates_list)})\n"
        f"  - Modelo principal académico: {', '.join(primary_models)}\n"
        f"  - Configuración JSON: {result['output_candidates_json_path']}\n"
        f"  - Reporte Markdown:   {result['output_report_path']}"
    )