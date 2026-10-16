#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_preprocessing.py — Preparación de datos para Machine Learning (Etapa 03).

Prepara el dataset limpio de alquileres de Moscú (`data/rentals_clean.csv`)
para su posterior modelado predictivo de `monthly_rent_rub`, respetando
estrictamente los hallazgos de `cleaning_report.md` y `eda_report.md`.

Características principales:
1. Verificación criptográfica SHA-256 de solo lectura sobre `data/rentals_clean.csv`.
2. Exclusión justificada de variables con fuga de información (`rent_per_sqm`),
   artefactos de auditoría (`dq_flags`), identificadores (`id`) y variables con
   corrupción semántica severa (`metro_station`, `metro_line`, `metro_distance_min`).
3. División temporal estricta (Train 70% / Val 15% / Test 15%) ordenada por
   `date_posted` sin romper días en los bordes para evitar fuga del futuro.
4. Ingeniería de variables de dominio sin fuga (`DomainFeatureEngineer`) integrada
   dentro del Pipeline de scikit-learn.
5. Arquitectura dual de preprocesamiento (`ColumnTransformer` + `Pipeline`):
   - `linear`: Imputación por mediana + `RobustScaler` en numéricas, booleanas en {0,1}
     y `OneHotEncoder(drop='if_binary')` en categóricas.
   - `tree`: Imputación por mediana SIN escalado en numéricas (preserva unidades
     físicas para GBDT/Bagging), booleanas en {0,1} y `OneHotEncoder` en categóricas.
6. Ajuste (`fit`) exclusivo sobre `X_train` y serialización reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler


# =============================================================================
# 1. CONSTANTES Y CONFIGURACIÓN DEL DOMINIO
# =============================================================================

EXPECTED_CLEAN_SHA256: str = (
    "e43378a0de55cdc2320b76f273ef47e1158a41e7fb0d06da0dc34666a5cc1a71"
)

TARGET_COL: str = "monthly_rent_rub"
ID_COL: str = "id"
DATE_COL: str = "date_posted"

# Variables excluidas con justificación documentada en el EDA:
LEAKAGE_COLS: List[str] = [
    "rent_per_sqm",  # Fuga directa: monthly_rent_rub / total_area en 100% de filas (r=0.8933)
]
AUDIT_AND_ID_COLS: List[str] = [
    "id",            # Identificador único secuencial (20,000 valores únicos)
    "dq_flags",      # Bandera de auditoría creada en 01_cleaning.py (98.31% nulos)
]
CORRUPTED_METRO_COLS: List[str] = [
    "metro_station",       # Inconsistencia geográfica: 11/11 okrugs por estación
    "metro_line",          # Derivada de metro_station con asignación incoherente
    "metro_distance_min",  # Colineal con to_center_km (r=0.904, VIF=5.498) y 45% > 120 min
]

# Variables crudas de entrada requeridas por el preprocesador (antes de ingeniería):
RAW_NUMERIC_COLS: List[str] = [
    "total_area",
    "rooms",
    "floor",
    "total_floors",
    "building_year",
    "deposit_months",
    "to_center_km",
    "lat",
    "lon",
]

RAW_BOOLEAN_COLS: List[str] = [
    "furnished",
    "pets_allowed",
]

RAW_CATEGORICAL_COLS: List[str] = [
    "okrug",
    "district",
    "building_type",
    "renovation",
]

# Variables numéricas generadas por DomainFeatureEngineer:
ENGINEERED_NUMERIC_COLS: List[str] = [
    "posting_year",
    "posting_month_sin",
    "posting_month_cos",
    "building_age",
    "area_per_room",
    "floor_ratio",
]

ENGINEERED_BINARY_COLS: List[str] = [
    "is_presale_or_new",
    "is_first_floor",
    "is_top_floor",
]


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuración inmutable del pipeline de preprocesamiento."""

    input_csv: Path = Path("data/rentals_clean.csv")
    output_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("models/preprocessing")
    report_path: Path = Path("reports/preprocessing_report.md")
    verify_sha256: bool = True
    include_metro_distance: bool = False
    include_district: bool = True
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"Las proporciones de partición deben sumar 1.0; se recibió {total:.4f}"
            )


# =============================================================================
# 2. UTILIDADES DE INTEGRIDAD Y CARGA DE DATOS
# =============================================================================

def compute_sha256(file_path: Path) -> str:
    """Calcula el hash SHA-256 de un archivo en modo solo lectura."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_and_validate_clean_data(
    csv_path: Path,
    verify_hash: bool = True,
    expected_sha256: str = EXPECTED_CLEAN_SHA256,
) -> Tuple[pd.DataFrame, str]:
    """
    Carga `data/rentals_clean.csv` sin modificarlo y verifica su integridad.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de entrada: {csv_path}")

    observed_sha256 = compute_sha256(csv_path)
    if verify_hash and observed_sha256 != expected_sha256:
        raise ValueError(
            f"Fallo de integridad SHA-256 en {csv_path}.\n"
            f"Esperado:  {expected_sha256}\n"
            f"Observado: {observed_sha256}"
        )

    df = pd.read_csv(csv_path, low_memory=False)
    required_cols = (
        {TARGET_COL, ID_COL, DATE_COL}
        | set(RAW_NUMERIC_COLS)
        | set(RAW_BOOLEAN_COLS)
        | set(RAW_CATEGORICAL_COLS)
    )
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Faltan columnas requeridas en el dataset limpio: {sorted(missing)}")

    return df, observed_sha256


# =============================================================================
# 3. SEPARACIÓN X / y Y EXCLUSIÓN DE FUGA DE INFORMACIÓN
# =============================================================================

def separate_features_and_target(
    df: pd.DataFrame,
    include_metro_distance: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, List[str]]]:
    """
    Separa la matriz predictora `X` y el vector objetivo `y` (`monthly_rent_rub`),
    excluyendo explícitamente las columnas de fuga, auditoría e inconsistencia.
    """
    if TARGET_COL not in df.columns:
        raise KeyError(f"La variable objetivo '{TARGET_COL}' no existe en el DataFrame.")

    y = df[TARGET_COL].astype(np.float64).copy()

    excluded_leakage = [c for c in LEAKAGE_COLS if c in df.columns]
    excluded_audit_id = [c for c in AUDIT_AND_ID_COLS if c in df.columns]

    metro_to_drop = list(CORRUPTED_METRO_COLS)
    if include_metro_distance and "metro_distance_min" in metro_to_drop:
        metro_to_drop.remove("metro_distance_min")
    excluded_metro = [c for c in metro_to_drop if c in df.columns]

    all_excluded = [TARGET_COL] + excluded_leakage + excluded_audit_id + excluded_metro
    X = df.drop(columns=all_excluded, errors="ignore").copy()

    exclusion_summary = {
        "target": [TARGET_COL],
        "leakage": excluded_leakage,
        "audit_and_id": excluded_audit_id,
        "corrupted_metro": excluded_metro,
        "retained_raw_features": list(X.columns),
    }
    return X, y, exclusion_summary


# =============================================================================
# 4. PARTICIÓN TEMPORAL (TRAIN / VALIDATION / TEST) SIN FUGA DEL FUTURO
# =============================================================================

@dataclass
class TemporalSplitResult:
    """Contenedor del resultado de la partición temporal."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    ids_train: pd.Series
    ids_val: pd.Series
    ids_test: pd.Series
    split_metadata: Dict[str, object] = field(default_factory=dict)


def temporal_train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    include_metro_distance: bool = False,
) -> TemporalSplitResult:
    """
    Divide el dataset cronológicamente según `date_posted` en Train, Validation y Test.

    Justificación:
    - `date_posted` abarca 2021-01-01 a 2026-04-29 con inflación nominal acumulada
      (Figura 07). Un split aleatorio filtraría precios futuros hacia el pasado.
    - Para evitar que anuncios publicados exactamente el mismo día queden repartidos
      entre dos particiones en la frontera de corte, el umbral se ajusta al cierre del
      día completo más cercano al cuantil objetivo.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Las proporciones train_ratio + val_ratio + test_ratio deben sumar 1.0.")

    df_sorted = df.copy()
    parsed_dates = pd.to_datetime(df_sorted[DATE_COL], format="%Y-%m-%d", errors="raise")
    df_sorted["_parsed_date"] = parsed_dates

    # Orden determinista por fecha de publicación e identificador único
    sort_cols = ["_parsed_date", ID_COL] if ID_COL in df_sorted.columns else ["_parsed_date"]
    df_sorted = df_sorted.sort_values(by=sort_cols, ascending=True).reset_index(drop=True)

    n_total = len(df_sorted)
    raw_train_end = int(round(n_total * train_ratio))
    raw_val_end = int(round(n_total * (train_ratio + val_ratio)))

    dates_array = df_sorted["_parsed_date"].to_numpy()

    # Ajuste de frontera para no partir un mismo día (date_posted) entre dos subconjuntos
    train_boundary_date = dates_array[max(0, raw_train_end - 1)]
    train_end_idx = int(np.searchsorted(dates_array, train_boundary_date, side="right"))

    val_boundary_date = dates_array[max(train_end_idx, raw_val_end - 1)]
    val_end_idx = int(np.searchsorted(dates_array, val_boundary_date, side="right"))

    df_sorted = df_sorted.drop(columns=["_parsed_date"])

    ids_all = (
        df_sorted[ID_COL].copy()
        if ID_COL in df_sorted.columns
        else pd.Series(df_sorted.index.astype(str), name=ID_COL)
    )

    X_all, y_all, exclusion_info = separate_features_and_target(
        df_sorted, include_metro_distance=include_metro_distance
    )

    X_train = X_all.iloc[:train_end_idx].reset_index(drop=True)
    y_train = y_all.iloc[:train_end_idx].reset_index(drop=True)
    ids_train = ids_all.iloc[:train_end_idx].reset_index(drop=True)

    X_val = X_all.iloc[train_end_idx:val_end_idx].reset_index(drop=True)
    y_val = y_all.iloc[train_end_idx:val_end_idx].reset_index(drop=True)
    ids_val = ids_all.iloc[train_end_idx:val_end_idx].reset_index(drop=True)

    X_test = X_all.iloc[val_end_idx:].reset_index(drop=True)
    y_test = y_all.iloc[val_end_idx:].reset_index(drop=True)
    ids_test = ids_all.iloc[val_end_idx:].reset_index(drop=True)

    # Auditoría de pares de coordenadas idénticas entre splits
    train_coords = set(zip(X_train["lat"].round(6), X_train["lon"].round(6)))
    val_coords = set(zip(X_val["lat"].round(6), X_val["lon"].round(6)))
    test_coords = set(zip(X_test["lat"].round(6), X_test["lon"].round(6)))

    split_metadata = {
        "n_total": n_total,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "pct_train": round(len(X_train) / n_total * 100, 2),
        "pct_val": round(len(X_val) / n_total * 100, 2),
        "pct_test": round(len(X_test) / n_total * 100, 2),
        "train_date_min": str(X_train[DATE_COL].min()),
        "train_date_max": str(X_train[DATE_COL].max()),
        "val_date_min": str(X_val[DATE_COL].min()),
        "val_date_max": str(X_val[DATE_COL].max()),
        "test_date_min": str(X_test[DATE_COL].min()),
        "test_date_max": str(X_test[DATE_COL].max()),
        "shared_coords_train_val": len(train_coords & val_coords),
        "shared_coords_train_test": len(train_coords & test_coords),
        "shared_coords_val_test": len(val_coords & test_coords),
        "exclusion_summary": exclusion_info,
    }

    return TemporalSplitResult(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        ids_train=ids_train,
        ids_val=ids_val,
        ids_test=ids_test,
        split_metadata=split_metadata,
    )


# =============================================================================
# 5. TRANSFORMADORES PERSONALIZADOS DE DOMINIO (COMPATIBLES CON SKLEARN)
# =============================================================================

class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformador sin estado (stateless) que extrae variables temporales y de
    dominio inmobiliario a partir de las columnas crudas de `X`.

    Operaciones realizadas fila a fila (sin riesgo de fuga entre muestras):
    1. De `date_posted`: extrae `posting_year`, `posting_month_sin`, `posting_month_cos`
       y descarta la cadena ISO `date_posted`.
    2. Anomalía `building_year > posting_year` (339 filas en EDA):
       crea `is_presale_or_new` (0/1) y `building_age = max(0, posting_year - building_year)`.
    3. Relaciones físicas del inmueble:
       - `area_per_room = total_area / max(1, rooms)`
       - `floor_ratio = floor / max(1, total_floors)`
       - `is_first_floor = (floor == 1)`
       - `is_top_floor = (floor == total_floors)`
    4. Normalización de booleanas (`furnished`, `pets_allowed`) a float (0.0 / 1.0)
       preservando valores faltantes (`np.nan`) para el imputador posterior.
    """

    def __init__(self, drop_date_col: bool = True) -> None:
        self.drop_date_col = drop_date_col

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "DomainFeatureEngineer":
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=getattr(self, "feature_names_in_", None))
        else:
            X_df = X.copy()

        # 1. Descomposición temporal segura desde date_posted
        if DATE_COL in X_df.columns:
            dt = pd.to_datetime(X_df[DATE_COL], format="%Y-%m-%d", errors="coerce")
            posting_year = dt.dt.year.astype(float)
            posting_month = dt.dt.month.astype(float)
        else:
            posting_year = pd.Series(np.nan, index=X_df.index, dtype=float)
            posting_month = pd.Series(np.nan, index=X_df.index, dtype=float)

        X_df["posting_year"] = posting_year
        X_df["posting_month_sin"] = np.sin(2.0 * np.pi * posting_month / 12.0)
        X_df["posting_month_cos"] = np.cos(2.0 * np.pi * posting_month / 12.0)

        # 2. Edad del edificio y tratamiento explícito de preventa (building_year > posting_year)
        b_year = pd.to_numeric(X_df["building_year"], errors="coerce")
        raw_age = posting_year - b_year
        X_df["is_presale_or_new"] = np.where(
            raw_age.isna(), np.nan, (raw_age < 0).astype(float)
        )
        X_df["building_age"] = np.where(raw_age.isna(), np.nan, np.maximum(0.0, raw_age))

        # 3. Ratios estructurales e indicadores de piso
        area = pd.to_numeric(X_df["total_area"], errors="coerce")
        rooms = pd.to_numeric(X_df["rooms"], errors="coerce")
        floor = pd.to_numeric(X_df["floor"], errors="coerce")
        total_floors = pd.to_numeric(X_df["total_floors"], errors="coerce")

        safe_rooms = np.where((rooms.isna()) | (rooms <= 0), np.nan, rooms)
        safe_total_floors = np.where(
            (total_floors.isna()) | (total_floors <= 0), np.nan, total_floors
        )

        X_df["area_per_room"] = area / safe_rooms
        X_df["floor_ratio"] = floor / safe_total_floors
        X_df["is_first_floor"] = np.where(floor.isna(), np.nan, (floor == 1).astype(float))
        X_df["is_top_floor"] = np.where(
            floor.isna() | total_floors.isna(),
            np.nan,
            (floor == total_floors).astype(float),
        )

        # 4. Conversión segura de booleanos a {0.0, 1.0} preservando NaNs
        for b_col in RAW_BOOLEAN_COLS:
            if b_col in X_df.columns:
                X_df[b_col] = self._coerce_boolean_series(X_df[b_col])

        if self.drop_date_col and DATE_COL in X_df.columns:
            X_df = X_df.drop(columns=[DATE_COL])

        return X_df

    @staticmethod
    def _coerce_boolean_series(series: pd.Series) -> pd.Series:
        mapping = {
            True: 1.0,
            False: 0.0,
            1: 1.0,
            0: 0.0,
            "True": 1.0,
            "False": 0.0,
            "true": 1.0,
            "false": 0.0,
            "1": 1.0,
            "0": 0.0,
        }
        return series.map(mapping).astype(float)


# =============================================================================
# 6. CONSTRUCCIÓN MODULAR DE PIPELINES SEGÚN FAMILIA DE MODELOS
# =============================================================================

def build_preprocessing_pipeline(
    model_family: str = "linear",
    include_metro_distance: bool = False,
    include_district: bool = True,
) -> Pipeline:
    """
    Construye el Pipeline de preprocesamiento adaptado a la familia de modelos:

    - `model_family="linear"` (Modelos Lineales Regularizados: Ridge, Lasso, ElasticNet, Huber):
      * Numéricas continuas y discretas: `SimpleImputer(strategy="median")` + `RobustScaler()`.
        Justificación: sensibilidad a la escala en penalizaciones L1/L2 y presencia de
        asimetría/outliers en `to_center_km` (skew=2.12) y `floor` (skew=1.07).
      * Binarias / Booleanas: `SimpleImputer(strategy="most_frequent")` sin escalar (ya en {0, 1}).
      * Categóricas: `SimpleImputer(strategy="most_frequent")` +
        `OneHotEncoder(handle_unknown="ignore", drop="if_binary", sparse_output=False)`.

    - `model_family="tree"` (Árboles de Decisión: LightGBM, XGBoost, CatBoost, RandomForest):
      * Numéricas continuas y discretas: `SimpleImputer(strategy="median")` SIN escalado.
        Justificación: las particiones por umbral en árboles son invariantes a transformaciones
        monótonas; no escalar preserva la interpretabilidad directa en unidades físicas.
      * Binarias / Booleanas: `SimpleImputer(strategy="most_frequent")` sin escalar.
      * Categóricas: `SimpleImputer(strategy="most_frequent")` +
        `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.
    """
    family = model_family.strip().lower()
    if family not in {"linear", "tree"}:
        raise ValueError(
            f"model_family debe ser 'linear' o 'tree'; se recibió '{model_family}'."
        )

    numeric_cols = list(RAW_NUMERIC_COLS) + list(ENGINEERED_NUMERIC_COLS)
    if include_metro_distance and "metro_distance_min" not in numeric_cols:
        numeric_cols.append("metro_distance_min")

    binary_cols = list(RAW_BOOLEAN_COLS) + list(ENGINEERED_BINARY_COLS)

    categorical_cols = [c for c in RAW_CATEGORICAL_COLS if include_district or c != "district"]

    # Rama numérica diferenciada según la familia del modelo
    if family == "linear":
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler(with_centering=True, with_scaling=True)),
            ]
        )
        ohe_drop = "if_binary"
    else:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        ohe_drop = None

    # Rama binaria (booleanas y banderas de dominio en {0, 1})
    binary_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    # Rama categórica
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop=ohe_drop,
                    sparse_output=False,
                    dtype=np.float64,
                ),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("bin", binary_transformer, binary_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    column_transformer.set_output(transform="pandas")

    full_pipeline = Pipeline(
        steps=[
            ("domain_features", DomainFeatureEngineer(drop_date_col=True)),
            ("preprocessor", column_transformer),
        ]
    )
    return full_pipeline


def build_target_log_transformer() -> FunctionTransformer:
    """
    Crea el transformador invertible `log1p` / `expm1` para `monthly_rent_rub`.
    Reduce la asimetría del target de 1.820 a -0.136 según el EDA (Figura 01).
    """
    return FunctionTransformer(
        func=np.log1p,
        inverse_func=np.expm1,
        validate=False,
        check_inverse=True,
    )


# =============================================================================
# 7. GENERACIÓN AUTOMÁTICA DEL REPORTE DE PREPROCESAMIENTO
# =============================================================================

def generate_preprocessing_report(
    report_path: Path,
    input_sha256: str,
    split_result: TemporalSplitResult,
    linear_pipeline: Pipeline,
    tree_pipeline: Pipeline,
    X_train_linear: pd.DataFrame,
    X_train_tree: pd.DataFrame,
) -> None:
    """Genera el archivo `reports/preprocessing_report.md` con trazabilidad completa."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    meta = split_result.split_metadata
    excl = meta["exclusion_summary"]

    num_cols_linear = linear_pipeline.named_steps["preprocessor"].transformers_[0][2]
    bin_cols_linear = linear_pipeline.named_steps["preprocessor"].transformers_[1][2]
    cat_cols_linear = linear_pipeline.named_steps["preprocessor"].transformers_[2][2]

    y_train_skew_raw = float(pd.Series(split_result.y_train).skew())
    y_train_skew_log = float(pd.Series(np.log1p(split_result.y_train)).skew())

    content = f"""# Informe de Preprocesamiento para Machine Learning — Alquileres en Moscú (Etapa 03)

| Metadato | Valor |
|---|---|
| Fecha de generación | `{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}` |
| Archivo de entrada (solo lectura) | `data/rentals_clean.csv` |
| SHA-256 verificado (inmutable) | `{input_sha256}` |
| Variable objetivo (`y`) | `{TARGET_COL}` (alquiler mensual en RUB) |
| Filas totales procesadas | `{meta["n_total"]}` |
| Dimensiones post-preprocesamiento (`linear`) | `{X_train_linear.shape[1]}` columnas |
| Dimensiones post-preprocesamiento (`tree`) | `{X_train_tree.shape[1]}` columnas |

## 1. Integridad de Entrada y Separación de Variables (`X` / `y`)

- **Verificación de inmutabilidad:** El archivo `data/rentals_clean.csv` se leyó en modo solo lectura comprobando que su hash SHA-256 coincide exactamente con la salida de `01_cleaning.py` y `02_eda.py` (`{input_sha256}`).
- **Variable objetivo (`y`):** `{TARGET_COL}` fue separada antes de cualquier transformación. En el conjunto de entrenamiento presenta una asimetría de `{y_train_skew_raw:.3f}` en escala original y de `{y_train_skew_log:.3f}` bajo transformación `np.log1p`.

## 2. Variables Excluidas y Justificación Anti-Fuga (Data Leakage)

| Columna Excluida | Categoría de Exclusión | Justificación Basada en el EDA |
|---|---|---|
| `{", ".join(excl["leakage"])}` | **Fuga directa del objetivo** | Equivale exactamente a `monthly_rent_rub / total_area` ($\pm 2\%$) en el 100.0% de los registros ($r = 0.8933$, Figura 06). No existe al momento de predecir un inmueble nuevo. |
| `id` | **Identificador único** | Secuencia alfanumérica (`R000001`..`R020000`) con cardinalidad 100% sin valor predictivo; se conserva por separado solo para trazabilidad de filas. |
| `dq_flags` | **Artefacto de auditoría** | Creada en `01_cleaning.py` (98.31% nula). Tanto el informe de limpieza como el EDA prohíben su uso como predictor. |
| `metro_station`, `metro_line` | **Corrupción semántica severa** | Cada estación aparece distribuida en una mediana de 11 de los 11 `okrug` de Moscú, lo cual es geográficamente imposible e introduce ruido espurio. |
| `metro_distance_min` | **Colinealidad severa y anomalía de escala** | Presenta $r = 0.904$ ($\text{{VIF}} = 5.498$) con `to_center_km`, alcanza 966 minutos y supera 120 minutos en el 45.0% de las filas (Figura 06). |

## 3. Estrategia de División Temporal (Train / Validation / Test)

Dado que `date_posted` abarca desde `2021-01-01` hasta `2026-04-29` y la Figura 07 demuestra una tendencia inflacionaria nominal sostenida, se aplicó una **división cronológica estricta (70% / 15% / 15%)** ajustada a días completos para evitar fuga temporal del futuro hacia el pasado:

| Partición | Filas (`n`) | Proporción (%) | Fecha Inicial (`min`) | Fecha Final (`max`) |
|---|---:|---:|---|---|
| **Entrenamiento (`Train`)** | `{meta["n_train"]}` | `{meta["pct_train"]:.2f}%` | `{meta["train_date_min"]}` | `{meta["train_date_max"]}` |
| **Validación (`Validation`)** | `{meta["n_val"]}` | `{meta["pct_val"]:.2f}%` | `{meta["val_date_min"]}` | `{meta["val_date_max"]}` |
| **Prueba (`Test`)** | `{meta["n_test"]}` | `{meta["pct_test"]:.2f}%` | `{meta["test_date_min"]}` | `{meta["test_date_max"]}` |

- **Control de registros relacionados:** Se verificaron las coordenadas exactas `(lat, lon)` entre particiones (`Train ∩ Val = {meta["shared_coords_train_val"]}`, `Train ∩ Test = {meta["shared_coords_train_test"]}`, `Val ∩ Test = {meta["shared_coords_val_test"]}`).

## 4. Ingeniería de Variables y Transformaciones por Familia de Modelos

1. **Ingeniería de dominio sin estado (`DomainFeatureEngineer`):**
   - Extracción temporal desde `date_posted`: `posting_year`, `posting_month_sin`, `posting_month_cos` (eliminando luego la cadena `date_posted`).
   - Resolución de la anomalía `building_year > posting_year` (339 filas): creación de `is_presale_or_new` (0/1) y `building_age = max(0, posting_year - building_year)`.
   - Ratios estructurales: `area_per_room = total_area / rooms`, `floor_ratio = floor / total_floors`, `is_first_floor` e `is_top_floor`.
2. **Columnas por rama dentro del `ColumnTransformer`:**
   - **Numéricas ({len(num_cols_linear)}):** `{", ".join(num_cols_linear)}`
   - **Binarias ({len(bin_cols_linear)}):** `{", ".join(bin_cols_linear)}`
   - **Categóricas ({len(cat_cols_linear)}):** `{", ".join(cat_cols_linear)}`
3. **Escalado diferenciado según el modelo:**
   - **Pipeline `linear` (`Ridge`, `Lasso`, `ElasticNet`, `HuberRegressor`):** Imputación por mediana + **`RobustScaler`** en las {len(num_cols_linear)} variables numéricas (mitiga la asimetría y cola pesada de `to_center_km` y `floor` observadas en la Figura 02), imputación por moda en binarias (`{{0, 1}}` sin escalar) y `OneHotEncoder(handle_unknown="ignore", drop="if_binary")` en categóricas.
   - **Pipeline `tree` (`LightGBM`, `XGBoost`, `CatBoost`, `RandomForest`, `ExtraTrees`):** Imputación por mediana **sin escalado** en las {len(num_cols_linear)} variables numéricas (los árboles son invariantes a escalados monótonos y conservan las unidades físicas originales), imputación por moda en binarias y `OneHotEncoder(handle_unknown="ignore")` en categóricas.
4. **Garantía anti-fuga en el ajuste:** Todos los estimadores (`SimpleImputer`, `RobustScaler`, `OneHotEncoder`) se ajustaron **exclusivamente sobre `X_train`**, aplicando únicamente `.transform()` sobre `X_val` y `X_test`.
"""
    report_path.write_text(content, encoding="utf-8")


# =============================================================================
# 8. ORQUESTACIÓN PRINCIPAL Y PERSISTENCIA DE ARTEFACTOS
# =============================================================================

def run_preprocessing(config: PreprocessingConfig) -> Dict[str, object]:
    """
    Ejecuta el flujo completo de preprocesamiento sin entrenar modelos ni
    alterar los archivos de entrada.
    """
    # 1. Carga y verificación SHA-256 antes del procesamiento
    df_clean, sha_before = load_and_validate_clean_data(
        config.input_csv, verify_hash=config.verify_sha256
    )

    # 2. Partición temporal Train / Val / Test
    split_res = temporal_train_val_test_split(
        df_clean,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        include_metro_distance=config.include_metro_distance,
    )

    # 3. Construcción y ajuste EXCLUSIVO con X_train para ambas familias de modelos
    linear_pipeline = build_preprocessing_pipeline(
        model_family="linear",
        include_metro_distance=config.include_metro_distance,
        include_district=config.include_district,
    )
    tree_pipeline = build_preprocessing_pipeline(
        model_family="tree",
        include_metro_distance=config.include_metro_distance,
        include_district=config.include_district,
    )
    target_log_transformer = build_target_log_transformer()

    X_train_linear = linear_pipeline.fit_transform(split_res.X_train, split_res.y_train)
    X_val_linear = linear_pipeline.transform(split_res.X_val)
    X_test_linear = linear_pipeline.transform(split_res.X_test)

    X_train_tree = tree_pipeline.fit_transform(split_res.X_train, split_res.y_train)
    X_val_tree = tree_pipeline.transform(split_res.X_val)
    X_test_tree = tree_pipeline.transform(split_res.X_test)

    y_train_log = pd.Series(
        target_log_transformer.fit_transform(split_res.y_train.to_numpy()),
        name=f"{TARGET_COL}_log1p",
    )
    y_val_log = pd.Series(
        target_log_transformer.transform(split_res.y_val.to_numpy()),
        name=f"{TARGET_COL}_log1p",
    )
    y_test_log = pd.Series(
        target_log_transformer.transform(split_res.y_test.to_numpy()),
        name=f"{TARGET_COL}_log1p",
    )

    # 4. Guardado de datos procesados y pipelines serializados
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)

    X_train_linear.to_csv(config.output_dir / "X_train_linear.csv", index=False)
    X_val_linear.to_csv(config.output_dir / "X_val_linear.csv", index=False)
    X_test_linear.to_csv(config.output_dir / "X_test_linear.csv", index=False)

    X_train_tree.to_csv(config.output_dir / "X_train_tree.csv", index=False)
    X_val_tree.to_csv(config.output_dir / "X_val_tree.csv", index=False)
    X_test_tree.to_csv(config.output_dir / "X_test_tree.csv", index=False)

    for split_name, y_raw, y_lg, ids_s in [
        ("train", split_res.y_train, y_train_log, split_res.ids_train),
        ("val", split_res.y_val, y_val_log, split_res.ids_val),
        ("test", split_res.y_test, y_test_log, split_res.ids_test),
    ]:
        target_df = pd.DataFrame(
            {
                ID_COL: ids_s,
                TARGET_COL: y_raw,
                f"{TARGET_COL}_log1p": y_lg,
            }
        )
        target_df.to_csv(config.output_dir / f"y_{split_name}.csv", index=False)

    joblib.dump(linear_pipeline, config.artifacts_dir / "preprocessor_linear.joblib")
    joblib.dump(tree_pipeline, config.artifacts_dir / "preprocessor_tree.joblib")
    joblib.dump(target_log_transformer, config.artifacts_dir / "target_log_transformer.joblib")

    metadata_payload = {
        "input_sha256": sha_before,
        "split_metadata": split_res.split_metadata,
        "linear_features_out": list(X_train_linear.columns),
        "tree_features_out": list(X_train_tree.columns),
    }
    (config.artifacts_dir / "preprocessing_metadata.json").write_text(
        json.dumps(metadata_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 5. Generación del reporte Markdown
    generate_preprocessing_report(
        report_path=config.report_path,
        input_sha256=sha_before,
        split_result=split_res,
        linear_pipeline=linear_pipeline,
        tree_pipeline=tree_pipeline,
        X_train_linear=X_train_linear,
        X_train_tree=X_train_tree,
    )

    # 6. Verificación de inmutabilidad post-ejecución sobre el archivo original
    sha_after = compute_sha256(config.input_csv)
    if sha_before != sha_after:
        raise RuntimeError(
            "ALERTA CRÍTICA: El hash de `data/rentals_clean.csv` cambió durante la ejecución."
        )

    return {
        "split": split_res,
        "linear_pipeline": linear_pipeline,
        "tree_pipeline": tree_pipeline,
        "X_train_linear": X_train_linear,
        "X_val_linear": X_val_linear,
        "X_test_linear": X_test_linear,
        "X_train_tree": X_train_tree,
        "X_val_tree": X_val_tree,
        "X_test_tree": X_test_tree,
    }


def parse_args() -> PreprocessingConfig:
    parser = argparse.ArgumentParser(
        description="Etapa 03: Preprocesamiento modular de alquileres en Moscú."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/rentals_clean.csv"),
        help="Ruta al dataset limpio de entrada.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directorio de salida para las matrices preprocesadas.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("models/preprocessing"),
        help="Directorio para guardar los pipelines serializados (.joblib).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/preprocessing_report.md"),
        help="Ruta del reporte Markdown de preprocesamiento.",
    )
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Omite la verificación estricta del hash SHA-256 (útil para tests sintéticos).",
    )
    parser.add_argument(
        "--include-metro-distance",
        action="store_true",
        help="Incluye metro_distance_min para experimentos de ablación.",
    )
    args = parser.parse_args()
    return PreprocessingConfig(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        artifacts_dir=args.artifacts_dir,
        report_path=args.report_path,
        verify_sha256=not args.skip_sha256,
        include_metro_distance=args.include_metro_distance,
    )


if __name__ == "__main__":
    cfg = parse_args()
    results = run_preprocessing(cfg)
    print(
        f"[OK] Preprocesamiento finalizado exitosamente.\n"
        f"  - Train: {len(results['X_train_linear'])} filas\n"
        f"  - Val:   {len(results['X_val_linear'])} filas\n"
        f"  - Test:  {len(results['X_test_linear'])} filas\n"
        f"  - Features (linear): {results['X_train_linear'].shape[1]}\n"
        f"  - Features (tree):   {results['X_train_tree'].shape[1]}\n"
        f"  - Reporte: {cfg.report_path}"
    )