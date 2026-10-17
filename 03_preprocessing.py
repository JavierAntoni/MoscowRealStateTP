"""
03_preprocessing.py — Etapa 03: Preprocesamiento y construcción de features
Proyecto: Alquileres en Moscú (regresión lineal, objetivo monthly_rent_rub)

Entrada : data/rentals_clean.csv          (salida de 01_cleaning.py, solo lectura)
Salida  : data/processed/splits/*.parquet  (X/y por split temporal)
          artifacts/preprocessor_03.joblib (ColumnTransformer ajustado SOLO con train)
          reports/preprocessing_report.json (auditoría reproducible)

Requisitos: python>=3.10, pandas>=2.0, scikit-learn>=1.3, joblib, pyarrow (opcional)

Cambios v2 (mejoras tras diagnóstico de residuos; ver reports/ablation_report.md):
  1. Deriva temporal  -> t_years (años desde 2021-01-01) con SplineTransformer y
                         extrapolation="constant": captura la subida 2021-2024 y, fuera
                         del rango de train, mantiene el último nivel (no extrapola
                         una tendencia lineal que sobreestimaría).
  2. No linealidad    -> splines en total_area (curvatura de la superficie).
  3. Señal de metro   -> metro_distance_min ENTRA por defecto, en escala log + spline.
                         El criterio del EDA (§7) era "usar si aporta señal incremental":
                         en bruto r=0.90 con to_center_km (VIF 5.56), en log solo 0.62, y
                         reduce el RMSE de test de ~0.126 a ~0.103. metro_station/line
                         siguen excluidas (no aportan, semántica inconsistente).
  4. Se probaron y DESCARTARON (sin ganancia medida): splines/polares/interacciones de
     lat-lon, Target Encoding por celdas de ~1-2 km, interacciones área x ubicación,
     renovación x ubicación, tiempo x ubicación, Box-Cox agresivo.

Cambios v3 (multicolinealidad; ver reports/multicollinearity_report.md):
  1. Una variable con spline ya no entra además como término lineal (la base B-spline + intercepto
     contiene la recta): eliminaba 2 dependencias exactas (total_area y log_metro).
  2. metro: UNA sola representación (spline de metro_distance_min con nudos por cuantiles).
     Se quitan log_metro y log_metro_minus_log_center; floor y area_per_room (redundantes, medido en valid).
  3. Cada bloque spline se ortogonaliza con PCA blanqueada (rotación ajustada solo en train: mismo espacio
     de funciones, sin pérdida de información) -> desaparece el VIF interno de las bases B-spline.
  4. --legacy-features reproduce el diseño anterior solo para comparar.

Cambios v4 (cola izquierda del QQ-plot; ver diagnóstico en el reporte JSON, "district_effect"):
  Causa medida: la cola (z hasta -4.8) NO viene de metro_distance_min anómalo (el 67 % de las filas
  supera 50 min y es coherente con la distancia al centro) sino de un efecto de distrito mal codificado:
  el TargetEncoder clásico codifica la media de log(renta) del distrito, que mezcla composición
  (superficie, tipo de edificio, ubicación) con el efecto propio del distrito; el modelo lineal no puede
  separarlas y sobrepredice sistemáticamente distritos como Chertanovo Yuzhnoye (sesgo -0.18 en log,
  ya presente en train). Corrección: el TargetEncoder de `district` se ajusta con el objetivo
  AJUSTADO por covariables (log-renta menos la parte explicada por las demás features, estimada
  conjuntamente con el distrito y con validación cruzada dentro de train). Solo cambia el objetivo
  con el que se ajusta ese encoder: no se elimina ninguna fila, no se toca valid/test, el modelo sigue
  prediciendo log1p(renta), y el preprocesador resultante es un ColumnTransformer estándar de sklearn
  (mismos 57 nombres de columnas, sin clases propias). --no-district-adjust reproduce la v3.

Cambios v5 (deriva temporal en el modelo final):
  valid y test caen enteros DESPUÉS del último día de train; con extrapolation="constant" el spline
  temporal congela el nivel de mercado del corte y aparece un sesgo positivo que crece ~0.0016
  log/mes (valid +0.013, test +0.023). No se corrige con otra extrapolación (probadas, empeoran);
  se corrige reajustando con datos recientes. Por eso este script escribe, ADEMÁS de los artefactos
  de siempre (sin cambios), artifacts/preprocessor_03_trainvalid.joblib: el mismo preprocesador
  (mismos 57 nombres de columna) ajustado con train+valid. 05_training_mlflow.py lo usa con
  `--fit-on train_valid`. Los splits y X_*/y_* no cambian; test no interviene en ningún ajuste.
  --no-trainvalid-preprocessor lo omite.

Uso: python 03_preprocessing.py [--strict-hash] [--baseline] [--exclude-metro-distance]
         [--no-district-adjust] [--no-trainvalid-preprocessor] [--train-frac 0.80 --valid-frac 0.10 | --fixed-dates]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler,
    SplineTransformer,
    StandardScaler,
    TargetEncoder,
)

RANDOM_STATE = 42
INPUT_SHA256 = "e43378a0de55cdc2320b76f273ef47e1158a41e7fb0d06da0dc34666a5cc1a71"  # EDA etapa 02

# --------------------------------------------------------------------------- #
# Configuración                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    input_csv: Path = Path("data/rentals_clean.csv")
    out_dir: Path = Path("data/processed/splits")
    artifact_dir: Path = Path("artifacts")
    report_dir: Path = Path("reports")

    # Split temporal (date_posted: 2021-01-01 -> 2026-04-29), SIEMPRE cronológico.
    # split_mode="proportional": cortes por fracción de filas ordenadas por fecha
    #   (por defecto 80/10/10; las filas de un mismo día no se separan entre splits).
    # split_mode="fixed": cortes por fecha fija (split v1 ~85/9/6).
    split_mode: str = "proportional"
    train_frac: float = 0.80
    valid_frac: float = 0.10        # test = 1 - train_frac - valid_frac
    train_end: str = "2025-06-30"   # solo para split_mode="fixed"
    valid_end: str = "2025-12-31"   # solo para split_mode="fixed"; test = posterior

    # Objetivo
    target: str = "monthly_rent_rub"

    # Columnas prohibidas (fuga / identificadores / auditoría) — guard-rail
    forbidden: tuple = ("id", "dq_flags", "rent_per_sqm")

    # metro_distance_min: ahora ON por defecto (señal incremental demostrada en log/spline)
    include_metro_distance: bool = True
    # True reproduce el diseño anterior (v2, con dependencias exactas) solo para comparar antes/después
    legacy_features: bool = False
    # Deriva temporal: "spline" (recomendado) | "linear" (control negativo) | None
    time_mode: str | None = "spline"
    # Spline sobre total_area
    area_spline: bool = True
    # v4: ajustar el TargetEncoder de district con el objetivo ajustado por covariables (ver docstring)
    district_adjust: bool = True
    district_adjust_alpha: float = 1.0   # Ridge del ajuste conjunto (covariables + distrito)
    district_adjust_folds: int = 5


# Columnas por rol (según EDA etapa 02, §7)
NUMERIC_COLS = [
    "total_area", "rooms", "floor", "total_floors", "deposit_months",
    "lat", "lon",
]
ENGINEERED_NUMERIC = ["building_age", "floor_ratio", "area_per_room"]
SPLINE_COLS = ["to_center_km"]                     # decaimiento no lineal del precio
TIME_ORIGIN = pd.Timestamp("2021-01-01")           # inicio del dataset (índice temporal)
TIME_COL = "t_years"
TIME_SPLINE_KNOTS, TIME_SPLINE_DEGREE = 8, 2       # elegido sobre valid (ver ablation_report)
# v3 (colinealidad): una variable representada por spline NO se repite como término lineal (la base
# B-spline + intercepto ya contiene la recta). Nudos por cuantiles: la señal del metro está en 0-30 min,
# zona poco densa donde los nudos uniformes no dan resolución. Elegido sobre valid (ver reporte).
SPLINE_SPECS = {"to_center_km": ("quantile", 8), "total_area": ("quantile", 8),
                "metro_distance_min": ("quantile", 12)}
ONEHOT_COLS = ["building_type", "renovation"]      # 6 y 4 categorías
TARGET_ENC_COLS = ["district"]                     # 129 categorías
BOOL_AS_INT = ["furnished", "pets_allowed"]
# okrug: excluido por redundancia con district (V de Cramér = 0.997)


# --------------------------------------------------------------------------- #
# Utilidades                                                                  #
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        if path.suffix == ".parquet":
            df.to_parquet(tmp_path, index=True)
        else:
            df.to_csv(tmp_path, index=True)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, suffix=".tmp", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(obj, tmp, indent=2, ensure_ascii=False, default=str)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


# --------------------------------------------------------------------------- #
# 1. Carga y validación de entrada                                            #
# --------------------------------------------------------------------------- #
def load_data(cfg: Config, strict_hash: bool) -> pd.DataFrame:
    digest = sha256_file(cfg.input_csv)
    if digest != INPUT_SHA256:
        msg = f"SHA-256 de entrada distinto al declarado: {digest}"
        if strict_hash:
            raise RuntimeError(msg)
        print(f"[AVISO] {msg} — se continúa (sin --strict-hash).")
    df = pd.read_csv(cfg.input_csv, parse_dates=["date_posted"])
    assert df["date_posted"].notna().all(), "date_posted con NaT tras el parseo"
    missing = set(cfg.forbidden) - set(df.columns)
    assert not missing, f"Columnas esperadas ausentes en la entrada: {missing}"
    return df


# --------------------------------------------------------------------------- #
# 2. Ingeniería de features                                                   #
# --------------------------------------------------------------------------- #
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    year_posted = out["date_posted"].dt.year

    # Edad del edificio; los 339 casos building_year > año(date_posted) -> NaN + flag
    age = year_posted - out["building_year"]
    anomaly = age < 0
    out["building_age"] = age.mask(anomaly).astype(float)
    out["building_age_missing"] = anomaly.astype(np.int8)

    out["floor_ratio"] = out["floor"] / out["total_floors"]
    out["area_per_room"] = out["total_area"] / out["rooms"]

    # Índice temporal continuo (años desde el inicio del dataset)
    out[TIME_COL] = (out["date_posted"] - TIME_ORIGIN).dt.days / 365.25

    # Metro: escala log y diferencia log respecto a to_center_km (aporta señal incremental
    # aunque en bruto correlacionen 0.90; en log la correlación baja a ~0.62)
    out["log_metro"] = np.log1p(out["metro_distance_min"])
    out["log_metro_minus_log_center"] = out["log_metro"] - np.log1p(out["to_center_km"])

    for c in BOOL_AS_INT:
        out[c] = out[c].astype(np.int8)
    return out


def resolve_cutoffs(df: pd.DataFrame, cfg: Config) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Devuelve (train_end, valid_end): train = date<=train_end, valid = (train_end, valid_end],
    test = date>valid_end. Con 'proportional' los cortes caen en el último día completo que
    deja >= la fracción pedida de filas en cada bloque, sin partir un mismo día."""
    if cfg.split_mode == "fixed":
        return pd.Timestamp(cfg.train_end), pd.Timestamp(cfg.valid_end)
    assert cfg.split_mode == "proportional", f"split_mode desconocido: {cfg.split_mode}"
    assert 0 < cfg.train_frac and 0 < cfg.valid_frac and cfg.train_frac + cfg.valid_frac < 1, \
        "Fracciones inválidas: train_frac + valid_frac debe ser < 1"
    dates = df["date_posted"].sort_values().reset_index(drop=True)
    n = len(dates)
    train_end = dates.iloc[int(round(n * cfg.train_frac)) - 1]
    valid_end = dates.iloc[int(round(n * (cfg.train_frac + cfg.valid_frac))) - 1]
    return train_end, valid_end


def temporal_split(df: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    train_end, valid_end = resolve_cutoffs(df, cfg)
    train = df[df["date_posted"] <= train_end]
    valid = df[(df["date_posted"] > train_end) & (df["date_posted"] <= valid_end)]
    test = df[df["date_posted"] > valid_end]
    for name, part in (("train", train), ("valid", valid), ("test", test)):
        assert len(part) > 0, f"Split {name} vacío — revisa los cortes temporales"
    # Garantía anti-fuga temporal: ningún split puede solaparse con el siguiente
    assert train["date_posted"].max() < valid["date_posted"].min()
    assert valid["date_posted"].max() < test["date_posted"].min()
    return {"train": train, "valid": valid, "test": test}


# --------------------------------------------------------------------------- #
# 3. Preprocesador (fit SOLO sobre train)                                     #
# --------------------------------------------------------------------------- #
def build_feature_lists(include_metro_distance: bool = True, time_mode: str | None = "spline",
                       area_spline: bool = True, legacy: bool = False) -> dict[str, list]:
    if legacy:   # diseño v2: términos lineales Y splines de la misma variable (dependencias exactas)
        numeric = NUMERIC_COLS + ENGINEERED_NUMERIC + BOOL_AS_INT
        spline = list(SPLINE_COLS)
        if area_spline:
            spline.append("total_area")
        if include_metro_distance:
            numeric += ["log_metro", "log_metro_minus_log_center"]
            spline += ["metro_distance_min", "log_metro"]
        if time_mode == "linear":
            numeric.append(TIME_COL)
        return {"numeric": numeric, "spline": spline, "onehot": ONEHOT_COLS, "target_enc": TARGET_ENC_COLS,
                "time_spline": [TIME_COL] if time_mode == "spline" else [], "legacy": True}

    # Diseño v3. Variables descartadas por redundancia (misma información, medido en valid):
    #   floor (= floor_ratio * total_floors), area_per_room (= total_area / rooms),
    #   log_metro y log_metro_minus_log_center (transformaciones de la variable ya en spline).
    numeric = ["rooms", "total_floors", "deposit_months", "lat", "lon",
               "building_age", "floor_ratio"] + BOOL_AS_INT
    spline = ["to_center_km"]
    if area_spline:
        spline.append("total_area")
    else:
        numeric.insert(0, "total_area")          # sin spline sí entra como término lineal
    if include_metro_distance:
        spline.append("metro_distance_min")
    if time_mode == "linear":
        numeric.append(TIME_COL)
    return {"numeric": numeric, "spline": spline, "onehot": ONEHOT_COLS, "target_enc": TARGET_ENC_COLS,
            "time_spline": [TIME_COL] if time_mode == "spline" else [], "legacy": False}


def _spline_block_v3(cols: list[str]) -> ColumnTransformer:
    """Un bloque por variable: B-spline (include_bias=False, sin la columna redundante con el
    intercepto) + PCA blanqueado. La PCA es una ROTACIÓN del mismo espacio de funciones (no pierde
    información; ajustada solo con train) que vuelve ortogonales las columnas del bloque y elimina el
    VIF interno de las bases B-spline, que son inevitablemente solapadas."""
    inner = []
    for c in cols:
        kind, nk = SPLINE_SPECS[c]
        inner.append((c, Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("spline", SplineTransformer(n_knots=nk, degree=3, include_bias=False, knots=kind)),
            ("orth", PCA(whiten=True, random_state=RANDOM_STATE)),
        ]), [c]))
    return ColumnTransformer(inner, remainder="drop")


def build_preprocessor(features: dict[str, list]) -> ColumnTransformer:
    if not features.get("legacy", False):
        return _build_preprocessor_v3(features)
    return _build_preprocessor_legacy(features)


def _build_preprocessor_v3(features: dict[str, list]) -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),   # mediana calculada en train
        ("scaler", RobustScaler()),                      # robusto a los extremos conservados
    ])
    onehot_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # drop="first": evita la trampa de las dummies (las dummies suman 1 = intercepto)
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")),
        ("scaler", StandardScaler()),
    ])
    target_enc_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # cv interno: en fit_transform cada fila de train se codifica sin usar su propio target
        ("target_enc", TargetEncoder(cv=5, smooth="auto", random_state=RANDOM_STATE)),
        ("scaler", StandardScaler()),
    ])
    transformers = [
        ("numeric", numeric_pipe, features["numeric"]),
        ("spline", _spline_block_v3(features["spline"]), features["spline"]),
        ("onehot", onehot_pipe, features["onehot"]),
        ("target_enc", target_enc_pipe, features["target_enc"]),
    ]
    if features.get("time_spline"):
        # extrapolation="constant": fuera del rango de train se mantiene el último nivel observado
        time_pipe = Pipeline(steps=[
            ("spline", SplineTransformer(n_knots=TIME_SPLINE_KNOTS, degree=TIME_SPLINE_DEGREE,
                                         include_bias=False, extrapolation="constant")),
            ("orth", PCA(whiten=True, random_state=RANDOM_STATE)),
        ])
        transformers.append(("time", time_pipe, features["time_spline"]))
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)


def _build_preprocessor_legacy(features: dict[str, list]) -> ColumnTransformer:
    # Nota numérica (v2): las bases de splines y las dummies entran estandarizadas y el
    # one-hot usa drop="first". Sin esto la matriz de diseño queda casi singular
    # (cond ~ 1e17: las dummies suman 1 = intercepto y las bases B-spline suman ~1) y
    # HuberRegressor (L-BFGS) converge ~5x más lento o se detiene sin converger.
    # No cambia las predicciones de OLS; en Ridge/Lasso/Huber solo reescala la penalización.
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),   # mediana calculada en train
        ("scaler", RobustScaler()),                      # robusto a los extremos conservados
    ])
    spline_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("spline", SplineTransformer(n_knots=6, degree=3, include_bias=False)),
        ("scaler", StandardScaler()),
    ])
    onehot_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")),
        ("scaler", StandardScaler()),
    ])
    target_enc_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # cv interno: evita que cada fila codifique con su propio target
        ("target_enc", TargetEncoder(cv=5, smooth="auto", random_state=RANDOM_STATE)),
        ("scaler", StandardScaler()),
    ])

    transformers = [
        ("numeric", numeric_pipe, features["numeric"]),
        ("spline", spline_pipe, features["spline"]),
        ("onehot", onehot_pipe, features["onehot"]),
        ("target_enc", target_enc_pipe, features["target_enc"]),
    ]
    if features.get("time_spline"):
        # extrapolation="constant": fuera del rango de train se mantiene el último nivel
        # observado (la tendencia se aplana desde ~2024; una recta sobreestimaría).
        time_pipe = Pipeline(steps=[
            ("spline", SplineTransformer(n_knots=TIME_SPLINE_KNOTS, degree=TIME_SPLINE_DEGREE,
                                         include_bias=False, extrapolation="constant")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("time", time_pipe, features["time_spline"]))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",   # guard-rail: nada entra al modelo sin pasar por aquí
        verbose_feature_names_out=True,
    )


# --------------------------------------------------------------------------- #
# 3b. Objetivo ajustado para el TargetEncoder de district (v4)                #
# --------------------------------------------------------------------------- #
def district_adjusted_target(X_train: pd.DataFrame, y_train: pd.Series, features: dict[str, list],
                             alpha: float = 1.0, n_folds: int = 5) -> tuple[pd.Series, dict]:
    """Devuelve y_train - (parte explicada por las covariables NO-distrito), usando solo TRAIN.

    El efecto de distrito se estima junto con las covariables (Ridge sobre [covariables + dummies de
    distrito]); así los coeficientes de lat/lon/to_center_km no absorben el efecto del distrito y la
    media por distrito del residuo es el efecto *neto* del distrito. Se calcula fuera de muestra
    (KFold dentro de train) para que el TargetEncoder no vea residuos ya sobreajustados.
    Solo se usa como `y` al ajustar el preprocesador (el único paso que consume y es el
    TargetEncoder); el modelo final sigue entrenándose con log1p(renta) sin modificar."""
    cov_feats = {**features, "target_enc": []}
    oof_cov = np.full(len(X_train), np.nan)
    for tr_idx, te_idx in KFold(n_folds, shuffle=True, random_state=RANDOM_STATE).split(X_train):
        pre = build_preprocessor(cov_feats)
        pre.transformers = [t for t in pre.transformers if t[0] != "target_enc"]  # solo covariables
        oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        Xa, Xb = X_train.iloc[tr_idx], X_train.iloc[te_idx]
        Ca = pre.fit_transform(Xa)
        Da = oh.fit_transform(Xa[features["target_enc"]])
        model = Ridge(alpha=alpha).fit(np.c_[Ca, Da], y_train.iloc[tr_idx])
        # parte NO-distrito de la predicción fuera de muestra
        oof_cov[te_idx] = pre.transform(Xb) @ model.coef_[:Ca.shape[1]] + model.intercept_
    assert np.isfinite(oof_cov).all(), "NaN en el ajuste por covariables"
    y_adj = y_train - oof_cov
    info = {"metodo": f"Ridge(alpha={alpha}) conjunto covariables+distrito, KFold({n_folds}) dentro de train",
            "y_adj_std": float(y_adj.std()), "y_log_std": float(y_train.std())}
    return y_adj, info

def fit_trainvalid_preprocessor(X: dict, y_log: dict, features: dict[str, list], cfg: Config,
                                expected_names: list[str]) -> dict:
    """Preprocesador de despliegue: mismo diseño, ajustado con train+valid (nunca con test).
    Incluye el objetivo ajustado del TargetEncoder de district (v4) recalculado con train+valid."""
    X_tv = pd.concat([X["train"], X["valid"]])
    y_tv = pd.concat([y_log["train"], y_log["valid"]])
    y_fit = y_tv
    if cfg.district_adjust and features["target_enc"]:
        y_fit, _ = district_adjusted_target(X_tv, y_tv, features,
                                            cfg.district_adjust_alpha, cfg.district_adjust_folds)
    pre = build_preprocessor(features)
    A_tv = pre.fit_transform(X_tv, y_fit)
    names = list(pre.get_feature_names_out())
    assert names == expected_names, "El preprocesador train+valid cambió los nombres de columna"
    assert np.isfinite(A_tv).all(), "NaN/Inf en la matriz train+valid"
    assert np.isfinite(pre.transform(X["test"])).all(), "NaN/Inf en test con el preprocesador train+valid"
    path = cfg.artifact_dir / "preprocessor_03_trainvalid.joblib"
    joblib.dump(pre, path)
    return {"activo": True, "path": str(path), "n_filas_ajuste": int(len(X_tv)),
            "n_features": len(names)}


# --------------------------------------------------------------------------- #
# 4. Main                                                                     #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 03 — Preprocesamiento")
    parser.add_argument("--strict-hash", action="store_true",
                        help="Abortar si el SHA-256 de entrada no coincide")
    parser.add_argument("--exclude-metro-distance", action="store_true",
                        help="Excluir metro_distance_min (por defecto se incluye)")
    parser.add_argument("--baseline", action="store_true",
                        help="Reproduce el preprocesamiento v1 (sin tiempo, sin spline de área, sin metro)")
    parser.add_argument("--legacy-features", action="store_true",
                        help="Diseño anterior (v2, con dependencias exactas); solo para comparar antes/después")
    parser.add_argument("--no-district-adjust", action="store_true",
                        help="Desactiva el objetivo ajustado del TargetEncoder de district (reproduce v3)")
    parser.add_argument("--no-trainvalid-preprocessor", action="store_true",
                        help="No escribir artifacts/preprocessor_03_trainvalid.joblib (v5)")
    parser.add_argument("--train-frac", type=float, default=0.80,
                        help="Fracción de filas para train (split cronológico). Default 0.80")
    parser.add_argument("--valid-frac", type=float, default=0.10,
                        help="Fracción de filas para valid; test = resto. Default 0.10")
    parser.add_argument("--fixed-dates", action="store_true",
                        help="Usar los cortes por fecha de la v1 (2025-06-30 / 2025-12-31, ~85/9/6)")
    args = parser.parse_args()

    split_kw = dict(split_mode="fixed") if args.fixed_dates else dict(
        split_mode="proportional", train_frac=args.train_frac, valid_frac=args.valid_frac)
    if args.baseline:
        cfg = Config(include_metro_distance=False, time_mode=None, area_spline=False,
                     legacy_features=args.legacy_features,
                     district_adjust=not args.no_district_adjust, **split_kw)
    else:
        cfg = Config(include_metro_distance=not args.exclude_metro_distance,
                     legacy_features=args.legacy_features,
                     district_adjust=not args.no_district_adjust, **split_kw)
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    cfg.report_dir.mkdir(parents=True, exist_ok=True)

    # --- Carga -------------------------------------------------------------
    df = load_data(cfg, strict_hash=args.strict_hash)
    print(f"Entrada: {len(df)} filas × {df.shape[1]} columnas")

    # --- Features ---------------------------------------------------------
    df = add_engineered_features(df)

    # Guard-rail de fugas: ninguna columna prohibida puede entrar a X
    feature_lists = build_feature_lists(cfg.include_metro_distance, cfg.time_mode, cfg.area_spline,
                                        legacy=cfg.legacy_features)
    used = set(c for k in ("numeric", "spline", "onehot", "target_enc", "time_spline") for c in feature_lists[k])
    leaks = used & set(cfg.forbidden)
    assert not leaks, f"¡FUGA DETECTADA! Columnas prohibidas en features: {leaks}"
    assert cfg.target not in used, "El objetivo no puede ser feature"

    # --- Split temporal ---------------------------------------------------
    splits = temporal_split(df, cfg)
    for name, part in splits.items():
        print(f"  {name:5s}: n={len(part):5d}  "
              f"[{part['date_posted'].min().date()} → {part['date_posted'].max().date()}]")

    # --- Targets ----------------------------------------------------------
    y = {name: part[cfg.target].astype(float) for name, part in splits.items()}
    y_log = {name: np.log1p(v) for name, v in y.items()}

    X = {name: part.drop(columns=[cfg.target, "date_posted",
                                  "building_year"] + list(cfg.forbidden))
         for name, part in splits.items()}

    # --- Ajuste del preprocesador SOLO con train ---------------------------
    preprocessor = build_preprocessor(feature_lists)
    # El TargetEncoder se ajusta con el target en escala log (objetivo del modelo)
    # fit_transform en train aplica el cross-fitting interno del TargetEncoder; llamar a
    # .transform(train) después usaría codificaciones ajustadas con su propio target.
    # v4: el único paso de preprocessor que consume `y` es el TargetEncoder de district; se le pasa el
    # objetivo ajustado por covariables (calculado solo con train). El modelo sigue usando y_log.
    de_info = {"activo": False}
    y_fit = y_log["train"]
    if cfg.district_adjust and feature_lists["target_enc"]:
        y_fit, de_info = district_adjusted_target(X["train"], y_log["train"], feature_lists,
                                                  cfg.district_adjust_alpha, cfg.district_adjust_folds)
        de_info["activo"] = True
        print(f"  Objetivo ajustado para TE de district: std {de_info['y_log_std']:.3f} -> {de_info['y_adj_std']:.3f}")
    Xt = {"train": preprocessor.fit_transform(X["train"], y_fit)}
    Xt.update({name: preprocessor.transform(X[name]) for name in ("valid", "test")})
    for name, arr in Xt.items():
        assert np.isfinite(arr if isinstance(arr, np.ndarray) else arr.toarray()).all(), \
            f"NaN/Inf en la matriz transformada de {name}"
    feature_names = list(preprocessor.get_feature_names_out())
    print(f"Matriz de diseño: {Xt['train'].shape[0]} × {len(feature_names)} features "
          f"(train transformado)")

    # --- Persistencia atómica ---------------------------------------------
    out_ext = ".parquet" if _has_pyarrow() else ".csv"
    for name in splits:
        atomic_write_df(X[name], cfg.out_dir / f"X_{name}{out_ext}")
        atomic_write_df(pd.DataFrame({cfg.target: y[name], "log1p_target": y_log[name]}),
                        cfg.out_dir / f"y_{name}{out_ext}")
    joblib.dump(preprocessor, cfg.artifact_dir / "preprocessor_03.joblib")

    # v5: preprocesador de despliegue (train+valid) para 05_training_mlflow.py --fit-on train_valid
    tv_info = {"activo": False}
    if not args.no_trainvalid_preprocessor:
        tv_info = fit_trainvalid_preprocessor(X, y_log, feature_lists, cfg, feature_names)
        tv_info["fecha_max_ajuste"] = str(splits["valid"]["date_posted"].max().date())
        print(f"  Preprocesador train+valid: {tv_info['n_filas_ajuste']} filas -> {tv_info['path']}")

    report = {
        "etapa": "03_preprocessing",
        "input_sha256_verificado": sha256_file(cfg.input_csv) == INPUT_SHA256,
        "config": {k: str(v) for k, v in asdict(cfg).items()},
        "features": feature_lists,
        "excluidas": {
            "fuga_o_auditoria": list(cfg.forbidden),
            "semantica_inconsistente": ["metro_station", "metro_line"],
            "redundantes": ["okrug (V=0.997 con district)", "building_year (se deriva building_age)"]
                           + ([] if cfg.legacy_features else [
                               "floor (= floor_ratio*total_floors; VIF 8.7 -> 1.0, mismo RMSE valid)",
                               "area_per_room (= total_area/rooms; sin aporte sobre total_area en spline)",
                               "log_metro y log_metro_minus_log_center (transformaciones de metro_distance_min)",
                               "total_area y log_metro como términos lineales (la base spline ya los contiene)"]),
            "incluida_con_tratamiento_log_spline": ["metro_distance_min — include_metro_distance="
                                                    + str(cfg.include_metro_distance)],
        },
        "split_cutoffs": {
            "split_mode": cfg.split_mode,
            "split_train_end": str(resolve_cutoffs(df, cfg)[0].date()),
            "split_valid_start": str((resolve_cutoffs(df, cfg)[0] + pd.Timedelta(days=1)).date()),
            "split_valid_end": str(resolve_cutoffs(df, cfg)[1].date()),
            "split_test_start": str((resolve_cutoffs(df, cfg)[1] + pd.Timedelta(days=1)).date()),
        },
        "splits": {
            name: {
                "n": int(len(part)),
                "pct": round(100 * len(part) / len(df), 2),
                "date_min": str(part["date_posted"].min().date()),
                "date_max": str(part["date_posted"].max().date()),
                "y_median": float(part[cfg.target].median()),
                "y_mean": float(part[cfg.target].mean()),
            } for name, part in splits.items()
        },
        "multicolinealidad_train": design_diagnostics(Xt["train"], feature_names),
        "district_effect": de_info,
        "preprocesador_trainvalid": tv_info,
        "feature_names_out": feature_names,
        "n_features_out": len(feature_names),
        "building_age_missing_por_split": {
            name: int(part["building_age_missing"].sum()) for name, part in splits.items()
        },
    }
    atomic_write_json(report, cfg.report_dir / "preprocessing_report.json")
    print("Artefactos escritos:")
    print(f"  - {cfg.out_dir}/X_*.{{{out_ext[1:]}}} , y_*.{{{out_ext[1:]}}}")
    print(f"  - {cfg.artifact_dir / 'preprocessor_03.joblib'}")
    print(f"  - {cfg.report_dir / 'preprocessing_report.json'}")
    print("\nEtapa 03 completada. Listo para Etapa 04 (selección de modelos lineales).")


def design_diagnostics(A: np.ndarray, names: list[str]) -> dict:
    """Colinealidad de la matriz de diseño de TRAIN (con intercepto): VIF por columna
    (1/(1-R²) de regresar cada columna sobre las demás), rango, número de condición
    (con columnas de norma unitaria, medida estándar de Belsley) y columnas en dependencia exacta."""
    n, p = A.shape
    vif = np.empty(p)
    for j in range(p):
        o = np.c_[np.ones(n), np.delete(A, j, axis=1)]
        r = A[:, j] - o @ np.linalg.lstsq(o, A[:, j], rcond=None)[0]
        sst = np.sum((A[:, j] - A[:, j].mean()) ** 2)
        one_m = np.sum(r ** 2) / sst if sst > 0 else 0.0
        vif[j] = np.inf if one_m < 1e-10 else 1.0 / one_m
    Ac = np.c_[np.ones(n), A]
    sv = np.linalg.svd(Ac / np.linalg.norm(Ac, axis=0), compute_uv=False)
    fin = np.isfinite(vif)
    order = np.argsort(-np.where(fin, vif, 1e30))
    return {
        "n_columnas": int(p), "rango_con_intercepto": int(np.linalg.matrix_rank(Ac)),
        "columnas_con_intercepto": int(p + 1),
        "dependencias_exactas": int(p + 1 - np.linalg.matrix_rank(Ac)),
        "numero_condicion_norma_unitaria": float(sv[0] / sv[-1]),
        "vif_infinitos": int((~fin).sum()), "vif_mayor_5": int(((vif > 5) & fin).sum()),
        "vif_mayor_10": int(((vif > 10) & fin).sum()),
        "vif_max_finito": float(vif[fin].max()) if fin.any() else None,
        "top_vif": {names[i]: (None if not fin[i] else round(float(vif[i]), 2)) for i in order[:10]},
        "vif_por_columna": {names[i]: (None if not fin[i] else round(float(vif[i]), 3)) for i in range(p)},
    }


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    sys.exit(main())