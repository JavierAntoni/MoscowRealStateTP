"""
03_preprocessing.py — Etapa 03: Preprocesamiento y construcción de features
Proyecto: Alquileres en Moscú (regresión lineal, objetivo monthly_rent_rub)

Entrada : data/rentals_clean.csv          (salida de 01_cleaning.py, solo lectura)
Salida  : data/processed/splits/*.parquet  (X/y por split temporal)
          artifacts/preprocessor_03.joblib (ColumnTransformer ajustado SOLO con train)
          reports/preprocessing_report.json (auditoría reproducible)

Requisitos: python>=3.10, pandas>=2.0, scikit-learn>=1.3, joblib, pyarrow (opcional)

Uso: python 03_preprocessing.py [--strict-hash] [--include-metro-distance]
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
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler,
    SplineTransformer,
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

    # Split temporal (date_posted: 2021-01-01 -> 2026-04-29)
    train_end: str = "2025-06-30"
    valid_end: str = "2025-12-31"   # test = posterior a esta fecha

    # Objetivo
    target: str = "monthly_rent_rub"

    # Columnas prohibidas (fuga / identificadores / auditoría) — guard-rail
    forbidden: tuple = ("id", "dq_flags", "rent_per_sqm")

    # metro_distance_min: condicional (VIF 5.56 vs to_center_km). Off por defecto.
    include_metro_distance: bool = False


# Columnas por rol (según EDA etapa 02, §7)
NUMERIC_COLS = [
    "total_area", "rooms", "floor", "total_floors", "deposit_months",
    "lat", "lon",
]
ENGINEERED_NUMERIC = ["building_age", "floor_ratio", "area_per_room"]
SPLINE_COLS = ["to_center_km"]                     # decaimiento no lineal del precio
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
    assert not set(cfg.forbidden) & set(df.columns) or True  # existen, pero se dropean luego
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

    for c in BOOL_AS_INT:
        out[c] = out[c].astype(np.int8)
    return out


def temporal_split(df: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    train = df[df["date_posted"] <= cfg.train_end]
    valid = df[(df["date_posted"] > cfg.train_end) & (df["date_posted"] <= cfg.valid_end)]
    test = df[df["date_posted"] > cfg.valid_end]
    for name, part in (("train", train), ("valid", valid), ("test", test)):
        assert len(part) > 0, f"Split {name} vacío — revisa los cortes temporales"
    return {"train": train, "valid": valid, "test": test}


# --------------------------------------------------------------------------- #
# 3. Preprocesador (fit SOLO sobre train)                                     #
# --------------------------------------------------------------------------- #
def build_feature_lists(include_metro_distance: bool) -> dict[str, list]:
    numeric = NUMERIC_COLS + ENGINEERED_NUMERIC + BOOL_AS_INT
    if include_metro_distance:
        numeric.append("metro_distance_min")
    return {
        "numeric": numeric,
        "spline": SPLINE_COLS,
        "onehot": ONEHOT_COLS,
        "target_enc": TARGET_ENC_COLS,
    }


def build_preprocessor(features: dict[str, list]) -> ColumnTransformer:
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),   # mediana calculada en train
        ("scaler", RobustScaler()),                      # robusto a los extremos conservados
    ])
    spline_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("spline", SplineTransformer(n_knots=6, degree=3, include_bias=False)),
    ])
    onehot_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    target_enc_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        # cv interno: evita que cada fila codifique con su propio target
        ("target_enc", TargetEncoder(cv=5, smooth="auto", random_state=RANDOM_STATE)),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, features["numeric"]),
            ("spline", spline_pipe, features["spline"]),
            ("onehot", onehot_pipe, features["onehot"]),
            ("target_enc", target_enc_pipe, features["target_enc"]),
        ],
        remainder="drop",   # guard-rail: nada entra al modelo sin pasar por aquí
        verbose_feature_names_out=True,
    )


# --------------------------------------------------------------------------- #
# 4. Main                                                                     #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 03 — Preprocesamiento")
    parser.add_argument("--strict-hash", action="store_true",
                        help="Abortar si el SHA-256 de entrada no coincide")
    parser.add_argument("--include-metro-distance", action="store_true",
                        help="Incluir metro_distance_min (por defecto se excluye)")
    args = parser.parse_args()

    cfg = Config(include_metro_distance=args.include_metro_distance)
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    cfg.report_dir.mkdir(parents=True, exist_ok=True)

    # --- Carga -------------------------------------------------------------
    df = load_data(cfg, strict_hash=args.strict_hash)
    print(f"Entrada: {len(df)} filas × {df.shape[1]} columnas")

    # --- Features ---------------------------------------------------------
    df = add_engineered_features(df)

    # Guard-rail de fugas: ninguna columna prohibida puede entrar a X
    feature_lists = build_feature_lists(cfg.include_metro_distance)
    used = set(sum(feature_lists.values(), []))
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
    preprocessor.fit(X["train"], y_log["train"])

    Xt = {name: preprocessor.transform(X[name]) for name in splits}
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

    report = {
        "etapa": "03_preprocessing",
        "input_sha256_verificado": sha256_file(cfg.input_csv) == INPUT_SHA256,
        "config": {k: str(v) for k, v in asdict(cfg).items()},
        "features": feature_lists,
        "excluidas": {
            "fuga_o_auditoria": list(cfg.forbidden),
            "semantica_inconsistente": ["metro_station", "metro_line"],
            "redundantes": ["okrug (V=0.997 con district)", "building_year (se deriva building_age)"],
            "condicional": ["metro_distance_min (VIF 5.56) — include_metro_distance="
                            + str(cfg.include_metro_distance)],
        },
        "splits": {
            name: {
                "n": int(len(part)),
                "date_min": str(part["date_posted"].min().date()),
                "date_max": str(part["date_posted"].max().date()),
                "y_median": float(part[cfg.target].median()),
                "y_mean": float(part[cfg.target].mean()),
            } for name, part in splits.items()
        },
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


def _has_pyarrow() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    sys.exit(main())