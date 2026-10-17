#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
05_training_mlflow.py — Paso 5 · Entrenamiento final + tracking con MLflow
===============================================================================
Proyecto: predicción de monthly_rent_rub (alquileres mensuales, Moscú).

Objetivo de este script
-----------------------
1. Reentrenar el modelo ganador de la allowlist lineal con train
   (+ valid, opcional y documentado) sobre el target log1p(monthly_rent_rub).
2. Retransformar a RUB con Duan-smearing:
       y_rub = exp(y_log) * S - 1,   S = mean(exp(residuos_log_fit)).
3. Evaluar UNA única vez sobre test (intacto hasta esta etapa).
4. Registrar todo en MLflow bajo el experimento "primer parcial":
   params, metrics (valid y test), artifacts (joblibs, reportes, predicciones,
   gráficas de residuos) y tags.

Restricciones
-------------
* SOLO modelos lineales: LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor.
* Guard-rails anti-fuga: id, dq_flags, rent_per_sqm y el target jamás llegan a X.
* data/rentals_clean.csv NUNCA se modifica (lectura de bytes solo con --strict-hash).

Uso
---
    python 05_training_mlflow.py --strict-hash
    python 05_training_mlflow.py --fit-on train_valid --compare-all
    python 05_training_mlflow.py --tracking-uri file:./mlruns --metric rmse_log

Requisitos: python>=3.10, pandas>=2.0, numpy>=1.24, scikit-learn>=1.3,
            joblib>=1.3, pyarrow>=14, mlflow>=2.x, scipy, matplotlib.
===============================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from scipy import sparse as scipy_sparse
from scipy import stats as scipy_stats
from sklearn.base import clone
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------------------------------- #
# Constantes del proyecto (alineadas con etapas 01-04)
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TARGET = "monthly_rent_rub"
TARGET_LOG = "log1p(monthly_rent_rub)"

CLEAN_CSV = Path("data/rentals_clean.csv")
EXPECTED_CLEAN_SHA256 = (
    "e43378a0de55cdc2320b76f273ef47e1158a41e7fb0d06da0dc34666a5cc1a71"
)

SPLIT_CUTOFFS: Dict[str, str] = {
    "split_train_end": "2025-06-30",
    "split_valid_start": "2025-07-01",
    "split_valid_end": "2025-12-31",
    "split_test_start": "2026-01-01",
}

# Guard-rail anti-fuga: jamás deben aparecer como features.
FORBIDDEN_FEATURE_TOKENS = ("id", "dq_flags", "rent_per_sqm", "monthly_rent_rub")

# Allowlist lineal cerrada (idéntica en espíritu a la etapa 04).
ALLOWED_MODEL_CLASSES = {
    "LinearRegression": LinearRegression,
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "HuberRegressor": HuberRegressor,
}
FAMILY_ALIASES = {
    "linear": "LinearRegression",
    "linearregression": "LinearRegression",
    "ols": "LinearRegression",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elasticnet": "ElasticNet",
    "elastic_net": "ElasticNet",
    "huber": "HuberRegressor",
    "huberregressor": "HuberRegressor",
}

# Umbral de "cola alta" de negocio (etapa 04: extremos NO filtrados a propósito).
TAIL_RUB = 200_000.0

# Estética de figuras (sobria, sin depender de seaborn).
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#cfc9c0",
        "axes.labelcolor": "#2b2b2b",
        "text.color": "#2b2b2b",
        "xtick.color": "#6b6560",
        "ytick.color": "#6b6560",
        "axes.grid": True,
        "grid.color": "#e8e4de",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "figure.dpi": 120,
    }
)
INK = "#2b2b2b"
ACCENT = "#c1440e"

LOGGER = logging.getLogger("paso05")


# --------------------------------------------------------------------------- #
# Utilidades: hashing y escrituras atómicas
# --------------------------------------------------------------------------- #
def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 de un archivo leyendo en bloques (solo lectura)."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Escritura atómica: tmp en el mismo directorio + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path, json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    )


def atomic_joblib_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def save_figure_atomic(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.png")
    fig.savefig(tmp, bbox_inches="tight", facecolor="white")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# Guard-rails anti-fuga y allowlist
# --------------------------------------------------------------------------- #
def _tokenize(name: str) -> List[str]:
    # Separa por '__' (prefijos de sklearn) o símbolos, pero conserva '_' dentro del nombre
    return [t for t in re.split(r"__|[^0-9a-zA-Z_]+", str(name).lower()) if t]


def leakage_violations(names: Iterable[str]) -> List[str]:
    """Devuelve los nombres de features que contienen tokens prohibidos."""
    bad: List[str] = []
    for name in names:
        tokens = set(_tokenize(name))
        lowered = str(name).lower()
        for banned in FORBIDDEN_FEATURE_TOKENS:
            if lowered == banned or banned in tokens or (banned != "id" and banned in lowered):
                bad.append(str(name))
                break
    return sorted(set(bad))


def assert_no_leakage(names: Iterable[str], context: str) -> None:
    bad = leakage_violations(names)
    if bad:
        raise AssertionError(
            f"[anti-fuga] Features prohibidas detectadas en {context}: {bad}. "
            f"Prohibidas: {FORBIDDEN_FEATURE_TOKENS}"
        )


def assert_linear_allowlist(estimator: Any, context: str) -> str:
    """Valida que el estimador pertenezca a la allowlist lineal."""
    final = estimator
    if hasattr(estimator, "steps") and estimator.steps:
        final = estimator.steps[-1][1]
    clase = type(final).__name__
    if clase not in ALLOWED_MODEL_CLASSES:
        raise AssertionError(
            f"[allowlist] {context}: la clase '{clase}' NO está permitida. "
            f"Permitidas: {sorted(ALLOWED_MODEL_CLASSES)}"
        )
    return clase


# --------------------------------------------------------------------------- #
# Carga de datos (salidas de la etapa 03)
# --------------------------------------------------------------------------- #
def _read_table(base: Path) -> pd.DataFrame:
    """Intenta .parquet / .pq / .csv para un stem de nombre dado."""
    for ext in (".parquet", ".pq", ".csv"):
        candidate = base.with_suffix(ext)
        if not candidate.exists():
            continue
        if ext in (".parquet", ".pq"):
            return pd.read_parquet(candidate)
        df = pd.read_csv(candidate)
        if "Unnamed: 0" in df.columns:
            df = df.set_index("Unnamed: 0")
        return df
    raise FileNotFoundError(
        f"No se encontró {base} con extensiones .parquet/.pq/.csv "
        f"(¿ejecutaste antes 03_preprocessing.py?)"
    )


def load_splits(data_dir: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for split in ("train", "valid", "test"):
        stem = data_dir / f"X_{split}"
        try:
            out[f"X_{split}"] = _read_table(stem)
        except FileNotFoundError:
            fallback = data_dir.parent / f"X_{split}"
            LOGGER.warning("X_%s no está en %s; probando %s", split, data_dir, fallback)
            out[f"X_{split}"] = _read_table(fallback)
    return out


def load_targets(data_dir: Path) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for split in ("train", "valid", "test"):
        stem = data_dir / f"y_{split}"
        try:
            df = _read_table(stem)
        except FileNotFoundError:
            fallback = data_dir.parent / f"y_{split}"
            df = _read_table(fallback)
        if isinstance(df, pd.DataFrame):
            if TARGET in df.columns:
                series = df[TARGET]
            elif df.shape[1] == 1:
                series = df.iloc[:, 0]
            else:
                raise RuntimeError(
                    f"y_{split} tiene varias columnas y ninguna es '{TARGET}': "
                    f"{list(df.columns)}"
                )
        else:
            series = df
        out[f"y_{split}"] = pd.Series(np.asarray(series).ravel(), index=df.index,
                                      name=TARGET)
    return out


def resolve_y_scale(y: pd.Series, mode: str) -> str:
    """
    Determina si los y guardados están en RUB crudo o ya en log1p.
    Heurística: mediana > 1000 → RUB crudo; mediana < 100 → escala log.
    """
    if mode in ("raw", "log1p"):
        return mode
    med = float(np.median(np.asarray(y, dtype=float)))
    if med > 1000:
        return "raw"
    if med < 100:
        return "log1p"
    raise RuntimeError(
        f"No puedo inferir la escala de y (mediana={med:.3f}). Usa --y-scale raw|log1p."
    )


def to_log_and_rub(y: pd.Series, scale: str) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(y, dtype=float)
    if np.any(~np.isfinite(values)):
        raise RuntimeError("y contiene valores no finitos.")
    if scale == "raw":
        if np.any(values <= 0):
            raise RuntimeError("En escala raw, y debe ser estrictamente positivo.")
        y_log = np.log1p(values)
        y_rub = values
    else:
        y_log = values
        y_rub = np.expm1(values)
        if np.any(y_rub <= 0):
            raise RuntimeError("La retransformación expm1(y_log) produjo valores <= 0.")
    return y_log, y_rub


# --------------------------------------------------------------------------- #
# Alineación de features con el preprocesador de la etapa 03
# --------------------------------------------------------------------------- #
def transform_features(
    X: pd.DataFrame,
    preprocessor: Any,
    n_features_expected: Optional[int],
) -> Tuple[np.ndarray, List[str], str]:
    """
    Devuelve (matriz densa float64, nombres de entrada, modo).

    Modos posibles:
      - 'preprocessor_applied': X contiene las features crudas y se transforma aquí.
      - 'already_transformed' : X ya viene transformado de la etapa 03/04.
    """
    raw_names = list(getattr(preprocessor, "feature_names_in_", []))

    if raw_names and set(raw_names).issubset(set(map(str, X.columns))):
        assert_no_leakage(raw_names, "features crudas (entrada al preprocesador)")
        Xt = preprocessor.transform(X[raw_names])
        return _to_dense(Xt), [str(c) for c in raw_names], "preprocessor_applied"

    if n_features_expected is not None and X.shape[1] == n_features_expected:
        assert_no_leakage(X.columns, "features ya transformadas")
        return _to_dense(X), [str(c) for c in X.columns], "already_transformed"

    # Último intento: tratar X como crudo aunque no haya feature_names_in_.
    try:
        Xt = preprocessor.transform(X)
        expected = n_features_expected or getattr(Xt, "shape", (0, 0))[1]
        if Xt.shape[1] == expected:
            assert_no_leakage(X.columns, "features crudas (entrada al preprocesador)")
            return _to_dense(Xt), [str(c) for c in X.columns], "preprocessor_applied"
    except Exception as exc:  # noqa: BLE001 - informativo
        LOGGER.debug("transform() falló en el intento de fallback: %s", exc)

    raise RuntimeError(
        "No pude alinear X con el preprocesador de la etapa 03. "
        f"X.shape={X.shape}, n_features esperadas={n_features_expected}, "
        f"feature_names_in_={len(raw_names)} columnas."
    )


def _to_dense(Xt: Any) -> np.ndarray:
    if scipy_sparse.issparse(Xt):
        Xt = Xt.toarray()
    arr = np.asarray(Xt, dtype=np.float64)
    if arr.ndim != 2:
        raise RuntimeError(f"Matriz de features con dimensionalidad inesperada: {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(
            "La matriz de features contiene NaN/inf tras el preprocesador "
            "(la imputación por mediana de train debería haberlos resuelto)."
        )
    return arr


def transformed_feature_names(preprocessor: Any, fallback_width: int) -> List[str]:
    try:
        return [str(c) for c in preprocessor.get_feature_names_out()]
    except Exception:  # noqa: BLE001
        return [f"feat_{i}" for i in range(fallback_width)]


# --------------------------------------------------------------------------- #
# Bundle de selección (etapa 04)
# --------------------------------------------------------------------------- #
def load_selection_bundle(path: Path) -> Dict[str, Any]:
    """
    Carga artifacts/best_model_04.joblib.

    Se aceptan dos formatos (el paso 4 guarda 'modelo + smearing + params +
    n_features', pero el nombre exacto de claves puede variar):
      - dict con claves tipo model/estimator/best_model + smearing + n_features.
      - estimador sklearn suelto.
    """
    obj = joblib.load(path)
    bundle: Dict[str, Any] = {
        "estimator": None,
        "params": {},
        "smearing_04": None,
        "n_features": None,
        "raw": obj,
    }

    if isinstance(obj, dict):
        est = None
        for key in ("model", "estimator", "best_model", "best_estimator", "modelo", "reg"):
            if key in obj and hasattr(obj[key], "fit"):
                est = obj[key]
                break
        if est is None:
            for value in obj.values():
                if hasattr(value, "fit") and hasattr(value, "predict"):
                    est = value
                    break
        if est is None:
            raise RuntimeError(f"No encontré un estimador dentro del bundle {path}.")
        bundle["estimator"] = est

        for key in ("params", "best_params", "hyperparams", "parametros", "params_best"):
            if isinstance(obj.get(key), dict):
                bundle["params"] = obj[key]
                break
        for key in ("smearing_factor", "smearing", "S", "smearing_04", "duan_s"):
            if key in obj and obj[key] is not None:
                bundle["smearing_04"] = float(obj[key])
                break
        for key in ("n_features", "n_features_in", "num_features"):
            if key in obj and obj[key] is not None:
                bundle["n_features"] = int(obj[key])
                break
    else:
        if not (hasattr(obj, "fit") and hasattr(obj, "predict")):
            raise RuntimeError(f"{path} no contiene un estimador sklearn.")
        bundle["estimator"] = obj

    assert_linear_allowlist(bundle["estimator"], f"bundle {path.name}")
    return bundle


def final_estimator(estimator: Any) -> Any:
    if hasattr(estimator, "steps") and estimator.steps:
        return estimator.steps[-1][1]
    return estimator


def lookup_param(params: Dict[str, Any], name: str) -> Any:
    """Busca 'alpha' también como 'regressor__alpha', 'modelo__alpha', etc."""
    if name in params:
        return params[name]
    for key, value in params.items():
        if key.endswith("__" + name):
            return value
    return None


def scalar_params(estimator: Any) -> Dict[str, Any]:
    """get_params() filtrado a escalares, apto para mlflow.log_params."""
    out: Dict[str, Any] = {}
    for key, value in estimator.get_params().items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
    return out


# --------------------------------------------------------------------------- #
# Métricas, smearing y diagnóstico
# --------------------------------------------------------------------------- #
def compute_smearing_factor(y_log: np.ndarray, yhat_log: np.ndarray) -> float:
    """S = mean(exp(residuos_log)) sobre el conjunto de ajuste (Duan)."""
    resid = y_log - yhat_log
    S = float(np.mean(np.exp(resid)))
    if not np.isfinite(S) or S <= 0:
        raise RuntimeError(f"Factor de smearing inválido: {S}")
    return S


def predict_all(
    estimator: Any, X: np.ndarray, smearing: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    yhat_log = np.asarray(estimator.predict(X), dtype=float)
    yhat_rub_smeared = np.clip(np.exp(yhat_log) * smearing - 1.0, 0.0, None)
    yhat_rub_naive = np.clip(np.expm1(yhat_log), 0.0, None)
    return yhat_log, yhat_rub_smeared, yhat_rub_naive


def regression_metrics(
    y_log: np.ndarray,
    yhat_log: np.ndarray,
    y_rub: np.ndarray,
    yhat_rub_smeared: np.ndarray,
    yhat_rub_naive: np.ndarray,
) -> Dict[str, float]:
    out: Dict[str, float] = {
        "rmse_log": float(np.sqrt(mean_squared_error(y_log, yhat_log))),
        "mae_log": float(mean_absolute_error(y_log, yhat_log)),
        "r2_log": float(r2_score(y_log, yhat_log)),
    }
    for tag, yhat in (("smeared", yhat_rub_smeared), ("naive", yhat_rub_naive)):
        out[f"rmse_rub_{tag}"] = float(np.sqrt(mean_squared_error(y_rub, yhat)))
        out[f"mae_rub_{tag}"] = float(mean_absolute_error(y_rub, yhat))
        out[f"r2_rub_{tag}"] = float(r2_score(y_rub, yhat))
    out["mape_rub"] = float(np.mean(np.abs((y_rub - yhat_rub_smeared) / y_rub)) * 100.0)
    out["mape_rub_naive"] = float(np.mean(np.abs((y_rub - yhat_rub_naive) / y_rub)) * 100.0)
    return out


def residual_diagnostics(
    y_log: np.ndarray,
    yhat_log: np.ndarray,
    y_rub: np.ndarray,
    yhat_rub: np.ndarray,
    prefix: str,
) -> Dict[str, float]:
    """
    Diagnóstico cuantitativo de residuos en escala log (homocedasticidad) y
    RUB (impacto de negocio), replicando los hallazgos de la etapa 04:
    abanico/heterocedasticidad, colas pesadas en Q-Q y subestimación de la
    cola alta (>= 200k RUB).
    """
    resid_log = y_log - yhat_log
    resid_rub = y_rub - yhat_rub

    std_log = float(np.std(resid_log, ddof=1)) if resid_log.size > 1 else 0.0
    z = (resid_log - float(np.mean(resid_log))) / std_log if std_log > 0 else resid_log

    tail_mask = y_rub >= TAIL_RUB
    top_mask = y_rub >= float(np.quantile(y_rub, 0.95))

    def tail_stats(mask: np.ndarray) -> Dict[str, float]:
        n = int(mask.sum())
        if n == 0:
            return {
                f"{prefix}_tail_n": 0,
                f"{prefix}_tail_share": 0.0,
                f"{prefix}_tail_bias_rub": 0.0,
                f"{prefix}_tail_mape": 0.0,
            }
        return {
            f"{prefix}_tail_n": n,
            f"{prefix}_tail_share": float(n / mask.size),
            # y - yhat > 0 → el modelo SUBESTIMA la cola alta.
            f"{prefix}_tail_bias_rub": float(np.mean(y_rub[mask] - yhat_rub[mask])),
            f"{prefix}_tail_mape": float(
                np.mean(np.abs((y_rub[mask] - yhat_rub[mask]) / y_rub[mask])) * 100.0
            ),
        }

    out: Dict[str, float] = {
        f"{prefix}_resid_log_mean": float(np.mean(resid_log)),
        f"{prefix}_resid_log_std": std_log,
        f"{prefix}_resid_log_skew": float(pd.Series(resid_log).skew()),
        f"{prefix}_resid_log_kurtosis": float(pd.Series(resid_log).kurt()),
        f"{prefix}_resid_log_max_abs_z": float(np.max(np.abs(z))) if z.size else 0.0,
        # Spearman(|residuo|, predicho) como indicador de abanico/heterocedasticidad.
        f"{prefix}_hetero_log_spearman": float(
            pd.Series(np.abs(resid_log)).corr(pd.Series(yhat_log), method="spearman")
        ),
        f"{prefix}_hetero_rub_spearman": float(
            pd.Series(np.abs(resid_rub)).corr(pd.Series(yhat_rub), method="spearman")
        ),
        f"{prefix}_clip_share": float(np.mean(yhat_rub == 0.0)),
        f"{prefix}_mean_bias_rub": float(np.mean(y_rub - yhat_rub)),
    }
    out.update(tail_stats(tail_mask))
    out[f"{prefix}_top5pct_threshold"] = float(np.quantile(y_rub, 0.95))
    out[f"{prefix}_top5pct_bias_rub"] = (
        float(np.mean(y_rub[top_mask] - yhat_rub[top_mask])) if top_mask.any() else 0.0
    )
    return out


# --------------------------------------------------------------------------- #
# Figuras de residuos: Predicho vs Real | Residuos vs Predicho | Q-Q
# --------------------------------------------------------------------------- #
def _triplet(y_true: np.ndarray, yhat: np.ndarray, unit: str, suptitle: str) -> plt.Figure:
    resid = y_true - yhat
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.suptitle(suptitle, fontsize=11, color=INK)

    # 1) Predicho vs Real
    ax = axes[0]
    lo = float(min(y_true.min(), yhat.min()))
    hi = float(max(y_true.max(), yhat.max()))
    ax.scatter(yhat, y_true, s=9, alpha=0.28, color=ACCENT, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1.1, ls="--")
    ax.set_xlabel(f"Predicho ({unit})")
    ax.set_ylabel(f"Real ({unit})")
    ax.set_title("Predicho vs Real")

    # 2) Residuos vs Predicho (+ mediana por decil para ver el abanico)
    ax = axes[1]
    ax.scatter(yhat, resid, s=9, alpha=0.28, color=ACCENT, edgecolors="none")
    ax.axhline(0.0, color=INK, lw=1.0, ls="--")
    frame = pd.DataFrame({"f": yhat, "r": resid})
    try:
        frame["bin"] = pd.qcut(frame["f"], 10, duplicates="drop")
        med = frame.groupby("bin", observed=True).median(numeric_only=True)
        ax.plot(med["f"], med["r"], color=INK, lw=1.7, marker="o", ms=3.5,
                label="mediana por decil")
        ax.legend(frameon=False, loc="best", fontsize=8)
    except ValueError:
        pass
    ax.set_xlabel(f"Predicho ({unit})")
    ax.set_ylabel(f"Residuo ({unit})")
    ax.set_title("Residuos vs Predicho")

    # 3) Q-Q de residuos
    ax = axes[2]
    (osm, osr), (slope, intercept, _r) = scipy_stats.probplot(resid, dist="norm")
    ax.scatter(osm, osr, s=9, alpha=0.30, color=ACCENT, edgecolors="none")
    ax.plot(osm, slope * osm + intercept, color=INK, lw=1.2, ls="--")
    ax.set_xlabel("Cuantiles teóricos")
    ax.set_ylabel("Cuantiles de residuos")
    ax.set_title("Q-Q")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def make_figures(
    y_log: np.ndarray,
    yhat_log: np.ndarray,
    y_rub: np.ndarray,
    yhat_rub: np.ndarray,
    split_label: str,
) -> Dict[str, plt.Figure]:
    return {
        f"residuos_log_{split_label}": _triplet(
            y_log, yhat_log, "log1p(RUB)", f"Residuos escala log — {split_label}"
        ),
        f"residuos_rub_{split_label}": _triplet(
            y_rub, yhat_rub, "RUB", f"Residuos escala RUB — {split_label}"
        ),
    }


# --------------------------------------------------------------------------- #
# Narrativa del diagnóstico (comparación con hallazgos de la etapa 04)
# --------------------------------------------------------------------------- #
def _hetero_sentence(h_log: float, h_rub: float) -> str:
    if abs(h_rub) > abs(h_log) + 0.05:
        return (
            f"La retransformación a RUB reintroduce el abanico: Spearman(|residuo|, "
            f"predicho) pasa de {h_log:.3f} (log) a {h_rub:.3f} (RUB). "
            "Coherente con lo visto en la etapa 04: en log hay relativa "
            "homocedasticidad y en RUB la varianza crece con el nivel de precio."
        )
    return (
        f"Indicadores de abanico: log {h_log:.3f} vs RUB {h_rub:.3f}. "
        "La heterocedasticidad en RUB es comparable a la de log en este recorte."
    )


def build_diagnostic_md(
    diag: Dict[str, float],
    family: str,
    smearing: float,
    fit_set: str,
    prev: Optional[Dict[str, float]] = None,
) -> str:
    lines: List[str] = []
    a = "# Diagnóstico de residuos — Paso 5 (entrenamiento + MLflow)\n"
    lines.append(a)
    lines.append(f"- Modelo reentrenado: **{family}** (allowlist lineal).")
    lines.append(f"- Conjunto de ajuste: `{fit_set}`.")
    lines.append(f"- Smearing de Duan: `S = {smearing:.6f}`.")
    lines.append(f"- Fecha (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}.\n")

    lines.append("## Escala log (homocedasticidad)")
    lines.append(
        f"- Residuos log — media {diag['valid_resid_log_mean']:.4f}, "
        f"std {diag['valid_resid_log_std']:.4f}, "
        f"skew {diag['valid_resid_log_skew']:.3f}, "
        f"kurtosis {diag['valid_resid_log_kurtosis']:.3f} (valid)."
    )
    lines.append(
        f"- Q-Q: desviación máxima |z| = {diag['valid_resid_log_max_abs_z']:.2f} (valid), "
        f"{diag.get('test_resid_log_max_abs_z', float('nan')):.2f} (test). "
        "Valores altos confirman las colas pesadas observadas en la etapa 04."
    )
    lines.append(
        f"- Kurtosis > 0 implica colas más pesadas que una normal en escala log: "
        f"el Q-Q se separa en los extremos (patrón previo: colas pesadas).\n"
    )

    lines.append("## Escala RUB (impacto de negocio)")
    lines.append("- " + _hetero_sentence(
        diag["valid_hetero_log_spearman"], diag["valid_hetero_rub_spearman"]
    ))
    lines.append(
        f"- Cola alta (>= {TAIL_RUB:,.0f} RUB, valid): "
        f"n={int(diag['valid_tail_n'])} filas ({diag['valid_tail_share']*100:.2f}%), "
        f"sesgo medio (real - predicho) = {diag['valid_tail_bias_rub']:,.0f} RUB, "
        f"MAPE = {diag['valid_tail_mape']:.2f}%."
    )
    tail_bias = diag["valid_tail_bias_rub"]
    if tail_bias > 0:
        lines.append(
            f"- **Subestimación persistente de la cola alta**: con familia lineal "
            f"las predicciones quedan por debajo del valor real en promedio "
            f"{tail_bias:,.0f} RUB. Esperado: la restricción lineal acota el rango "
            f"predicho; el alivio viene de splines + smearing, no de más datos."
        )
    else:
        lines.append(
            "- Sin subestimación neta de la cola alta en valid (sesgo <= 0). "
            "Conviene verificarlo también en test."
        )
    lines.append(
        f"- Sesgo global (real - predicho) en valid: {diag['valid_mean_bias_rub']:,.0f} RUB; "
        f"MAPE valid = {diag.get('valid_mape_rub', float('nan')):.2f}%.\n"
    )

    lines.append("## Comparación con las gráficas previas (etapa 04)")
    if prev:
        lines.append(
            f"- Predicciones previas de valid (`predictions_valid.csv`): "
            f"abanico Spearman = {prev.get('hetero', float('nan')):.3f}, "
            f"sesgo cola alta = {prev.get('tail_bias', float('nan')):,.0f} RUB, "
            f"MAPE = {prev.get('mape', float('nan')):.2f}%."
        )
        delta_h = diag["valid_hetero_rub_spearman"] - prev.get("hetero", 0.0)
        delta_t = diag["valid_tail_bias_rub"] - prev.get("tail_bias", 0.0)
        lines.append(
            f"- Deltas (05 − 04): abanico {delta_h:+.3f}, sesgo cola {delta_t:+,.0f} RUB. "
            "Un sesgo de cola menos positivo indica mejoría en la subestimación."
        )
    else:
        lines.append(
            "- No fue posible leer `predictions_valid.csv` de la etapa 04 para una "
            "comparación numérica; la comparación cualitativa se sostiene en las "
            "gráficas guardadas en `reports/figures/`."
        )
    lines.append(
        "\n- Recuerde: el objetivo de modelado es `log1p(monthly_rent_rub)`; en RUB "
        "la heterocedasticidad reaparece por la retransformación. El diagnóstico "
        "decisivo de homocedasticidad es el de escala log; el de RUB mide impacto "
        "de negocio.\n"
    )
    return "\n".join(lines)


def load_prev_valid_diagnostics(path: Path) -> Optional[Dict[str, float]]:
    """Mejor esfuerzo: diagnosticar reports/predictions_valid.csv de la etapa 04."""
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None
    real_cols = [c for c in df.columns if any(t in c.lower()
                 for t in ("real", "true", "actual", "observ"))]
    pred_cols = [c for c in df.columns if any(t in c.lower()
                 for t in ("pred", "yhat", "y_hat", "estim"))]
    if not real_cols or not pred_cols:
        return None
    y = pd.to_numeric(df[real_cols[0]], errors="coerce").to_numpy()
    p = pd.to_numeric(df[pred_cols[0]], errors="coerce").to_numpy()
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if y.size < 20:
        return None
    scale = "log" if float(np.median(y)) < 100 else "rub"
    if scale == "log":
        y_rub, p_rub = np.expm1(y), np.clip(np.expm1(p), 0.0, None)
        resid_log = y - p
    else:
        y_rub, p_rub = y, p
        resid_log = np.log1p(y) - np.log1p(np.clip(p, 0.0, None))
    tail = y_rub >= TAIL_RUB
    return {
        "hetero": float(pd.Series(np.abs(y_rub - p_rub)).corr(pd.Series(p_rub),
                       method="spearman")),
        "tail_bias": float(np.mean(y_rub[tail] - p_rub[tail])) if tail.any() else 0.0,
        "mape": float(np.mean(np.abs((y_rub - p_rub) / y_rub)) * 100.0),
        "resid_log_kurt": float(pd.Series(resid_log).kurt()),
    }


# --------------------------------------------------------------------------- #
# Resolución del ganador y params por familia (etapa 04)
# --------------------------------------------------------------------------- #
def family_best_params(report: Any, family: str) -> Tuple[Dict[str, Any], str]:
    """
    Busca los hiperparámetros ganadores de una familia en el reporte de la
    etapa 04 (estructura tolerante). Devuelve (params, origen).
    """
    if not isinstance(report, dict):
        return {}, "default"
    aliases = {
        family.lower(),
        family.lower().replace("regressor", ""),
        family.lower().replace("linearregression", "linear"),
    }
    candidates: List[Dict[str, Any]] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            lower_keys = {str(k).lower() for k in node}
            if aliases & {p for p in path.split("|") if p} or any(
                alias in path.lower() for alias in aliases
            ):
                for key in ("best_params", "params", "best_params_", "parametros",
                            "best_hyperparams", "hyperparams"):
                    if isinstance(node.get(key), dict):
                        candidates.append(node[key])
                scalar_keys = lower_keys & {"alpha", "l1_ratio", "epsilon", "max_iter",
                                            "tol", "fit_intercept", "c"}
                if scalar_keys and not candidates:
                    candidates.append(
                        {k: v for k, v in node.items()
                         if isinstance(v, (int, float, str, bool)) or v is None}
                    )
            for key, value in node.items():
                walk(value, f"{path}|{str(key).lower()}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}|{i}")

    walk(report, family.lower())
    if not candidates:
        return {}, "default"
    # El candidato con más claves reconocibles por la clase es el más plausible.
    cls = ALLOWED_MODEL_CLASSES[family]
    valid_keys = set(cls().get_params().keys())
    best = max(candidates, key=lambda d: len(set(d) & valid_keys))
    filtered = {k: v for k, v in best.items() if k in valid_keys}
    return (filtered, "reporte_04") if filtered else ({}, "default")


def build_estimator(family: str, params: Dict[str, Any]) -> Any:
    cls = ALLOWED_MODEL_CLASSES[family]
    valid = set(cls().get_params().keys())
    clean = {k: v for k, v in params.items() if k in valid}
    est = cls(**clean)
    est.set_params(random_state=RANDOM_STATE) if "random_state" in valid else None
    return est


def resolve_family(estimator: Any) -> str:
    clase = assert_linear_allowlist(estimator, "ganador")
    return clase


# --------------------------------------------------------------------------- #
# Logging MLflow
# --------------------------------------------------------------------------- #
def flatten_params(estimator: Any, extra: Dict[str, Any]) -> Dict[str, Any]:
    flat = {f"modelo__{k}": v for k, v in scalar_params(estimator).items()}
    base = scalar_params(estimator)
    flat["alpha"] = lookup_param(base, "alpha")
    flat["l1_ratio"] = lookup_param(base, "l1_ratio")
    flat["epsilon"] = lookup_param(base, "epsilon")
    flat.update(extra)
    return {k: ("None" if v is None else v) for k, v in flat.items()}


def log_mlflow_run(
    *,
    run_name: str,
    tags: Dict[str, str],
    params: Dict[str, Any],
    metrics: Dict[str, float],
    figures: Dict[str, plt.Figure],
    artifacts: Sequence[Tuple[Path, str]],
    estimator: Optional[Any],
    log_sklearn_model: bool,
) -> str:
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags)
        mlflow.log_params({str(k): str(v) for k, v in params.items()})
        mlflow.log_metrics({str(k): float(v) for k, v in metrics.items()
                            if np.isfinite(float(v))})

        for fig_name, fig in figures.items():
            mlflow.log_figure(fig, f"figures/{fig_name}.png")

        for path, subdir in artifacts:
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path=subdir)
            else:
                LOGGER.warning("Artifact inexistente (se omite): %s", path)

        if estimator is not None and log_sklearn_model:
            try:
                mlflow.sklearn.log_model(sk_model=estimator, artifact_path="model")
            except Exception as exc:  # noqa: BLE001 - el joblib ya quedó registrado
                LOGGER.warning("mlflow.sklearn.log_model falló (%s); "
                               "el .joblib del modelo ya está como artifact.", exc)
        run_id = run.info.run_id
    return run_id


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paso 5: entrenamiento final + MLflow (experimento 'primer parcial').",
    )
    parser.add_argument("--tracking-uri", default="file:./mlruns",
                        help="URI de tracking MLflow (default: file:./mlruns).")
    parser.add_argument("--experiment", default="primer parcial",
                        help="Nombre del experimento MLflow (default: 'primer parcial').")
    parser.add_argument("--metric", default="rmse_log",
                        choices=["rmse_log", "mae_log", "rmse_rub", "mae_rub"],
                        help="Métrica de selección declarada (se registra como param).")
    parser.add_argument("--fit-on", default="train_valid", choices=["train", "train_valid"],
                        help="Conjunto de reentrenamiento (default: train_valid, documentado).")
    parser.add_argument("--compare-all", action="store_true",
                        help="Además del ganador, reentrena toda la allowlist para "
                             "comparación en valid (test SOLO se evalúa en el ganador).")
    parser.add_argument("--model-family", default=None,
                        choices=sorted(set(FAMILY_ALIASES.values())),
                        help="Forzar familia del 'ganador' (si difiere del bundle 04).")
    parser.add_argument("--data-dir", default="data/processed/splits",
                        help="Directorio con X_/y_ parquet|csv de la etapa 03.")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--y-scale", default="auto", choices=["auto", "raw", "log1p"],
                        help="Escala de los y guardados por la etapa 03.")
    parser.add_argument("--strict-hash", action="store_true",
                        help="Verifica SHA-256 de data/rentals_clean.csv (solo lectura).")
    parser.add_argument("--no-sklearn-model", action="store_true",
                        help="No registrar el modelo con mlflow.sklearn (solo joblib).")
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    if args.seed != RANDOM_STATE:
        LOGGER.warning("Se usa seed=%d (el proyecto declara RANDOM_STATE=%d).",
                       args.seed, RANDOM_STATE)
    seed = args.seed

    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir)
    reports_dir = Path(args.reports_dir)
    figures_dir = reports_dir / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # --- Guard-rail de integridad de la entrada de la etapa 01 (solo lectura) --
    sha_value = "no_verificado"
    if args.strict_hash:
        if not CLEAN_CSV.exists():
            raise FileNotFoundError(f"--strict-hash: no existe {CLEAN_CSV}")
        sha_value = sha256_file(CLEAN_CSV)
        if sha_value != EXPECTED_CLEAN_SHA256:
            raise RuntimeError(
                f"SHA-256 de {CLEAN_CSV} distinto al declarado en la etapa 01.\n"
                f"  esperado: {EXPECTED_CLEAN_SHA256}\n  obtenido: {sha_value}"
            )
        LOGGER.info("SHA-256 de %s verificado (solo lectura).", CLEAN_CSV)

    # --- Split temporal declarado (params) ---------------------------------- #
    split_params = dict(SPLIT_CUTOFFS)

    # --- Datos --------------------------------------------------------------- #
    LOGGER.info("Cargando splits desde %s ...", data_dir)
    X_splits = load_splits(data_dir)
    y_splits = load_targets(data_dir)
    y_scale = resolve_y_scale(y_splits["y_train"], args.y_scale)
    LOGGER.info("Escala de y detectada: %s", y_scale)

    y_log: Dict[str, np.ndarray] = {}
    y_rub: Dict[str, np.ndarray] = {}
    for split in ("train", "valid", "test"):
        log_arr, rub_arr = to_log_and_rub(y_splits[f"y_{split}"], y_scale)
        y_log[split], y_rub[split] = log_arr, rub_arr

    for split in ("train", "valid", "test"):
        if X_splits[f"X_{split}"].shape[0] != y_log[split].shape[0]:
            raise RuntimeError(
                f"Desalineación X/y en {split}: "
                f"{X_splits[f'X_{split}'].shape[0]} vs {y_log[split].shape[0]}"
            )

    # --- Preprocesador y bundle de selección -------------------------------- #
    preprocessor_path = artifacts_dir / "preprocessor_03.joblib"
    bundle_path = artifacts_dir / "best_model_04.joblib"
    for required in (preprocessor_path, bundle_path):
        if not required.exists():
            raise FileNotFoundError(
                f"Falta {required}. Ejecuta 03_preprocessing.py y 04_model_selection.py."
            )
    preprocessor = joblib.load(preprocessor_path)
    bundle = load_selection_bundle(bundle_path)
    winner_est_04 = bundle["estimator"]
    n_features = bundle["n_features"]

    # --- Transformación + guard-rail anti-fuga ------------------------------- #
    Xt: Dict[str, np.ndarray] = {}
    fit_mode = ""
    for split in ("train", "valid", "test"):
        Xt[split], input_names, fit_mode = transform_features(
            X_splits[f"X_{split}"], preprocessor, n_features
        )
    LOGGER.info("Alineación de features: %s | X_train=%s", fit_mode, Xt["train"].shape)

    out_names = transformed_feature_names(preprocessor, Xt["train"].shape[1])
    assert_no_leakage(out_names, "features transformadas (salida del preprocesador)")
    n_features_transformed = Xt["train"].shape[1]
    if n_features is None:
        n_features = n_features_transformed

    # --- Conjunto de ajuste -------------------------------------------------- #
    if args.fit_on == "train_valid":
        X_fit = np.vstack([Xt["train"], Xt["valid"]])
        y_log_fit = np.concatenate([y_log["train"], y_log["valid"]])
        y_rub_fit = np.concatenate([y_rub["train"], y_rub["valid"]])
    else:
        X_fit, y_log_fit, y_rub_fit = Xt["train"], y_log["train"], y_rub["train"]
    Xt["fit"] = X_fit
    y_log["fit"] = y_log_fit
    y_rub["fit"] = y_rub_fit
    LOGGER.info("Conjunto de ajuste: %s (%d filas).", args.fit_on, X_fit.shape[0])

    # --- Familia ganadora ---------------------------------------------------- #
    if args.model_family:
        family = args.model_family
        params_winner, params_origin = family_best_params(
            _safe_load_json(reports_dir / "model_selection_report.json"), family
        )
        if params_origin == "default" and family == resolve_family(winner_est_04):
            params_winner = scalar_params(winner_est_04)
            params_origin = "bundle_04"
    else:
        family = resolve_family(winner_est_04)
        params_winner = scalar_params(winner_est_04) or bundle["params"]
        params_origin = "bundle_04"
    LOGGER.info("Ganador: %s (params desde %s).", family, params_origin)

    # --- Reentrenamiento del ganador ----------------------------------------- #
    winner = clone(winner_est_04) if params_origin == "bundle_04" \
        else build_estimator(family, params_winner)
    if args.seed == RANDOM_STATE and hasattr(winner, "get_params"):
        if "random_state" in winner.get_params():
            winner.set_params(random_state=seed)
    winner.fit(X_fit, y_log_fit)

    smearing_fit = compute_smearing_factor(y_log_fit, winner.predict(X_fit))
    LOGGER.info("Smearing de Duan (fit): S = %.6f", smearing_fit)
    if bundle["smearing_04"] is not None:
        LOGGER.info("Smearing etapa 04: S = %.6f (se usa el recalculado en fit).",
                    bundle["smearing_04"])

    # --- Métricas: fit, valid y TEST (una única evaluación de test) ---------- #
    metrics: Dict[str, float] = {}
    preds_store: Dict[str, Dict[str, np.ndarray]] = {}

    def evaluate(split: str) -> None:
        yhat_log, yhat_rub_s, yhat_rub_n = predict_all(winner, Xt[split], smearing_fit)
        preds_store[split] = {
            "yhat_log": yhat_log, "yhat_rub_smeared": yhat_rub_s,
            "yhat_rub_naive": yhat_rub_n,
        }
        block = regression_metrics(y_log[split], yhat_log, y_rub[split], yhat_rub_s, yhat_rub_n)
        metrics.update({f"{split}_{k}": v for k, v in block.items()})
        metrics.update(residual_diagnostics(
            y_log[split], yhat_log, y_rub[split], yhat_rub_s, split
        ))

    evaluate("fit")
    evaluate("valid")
    evaluate("test")  # <-- única evaluación sobre test en todo el proceso
    LOGGER.info(
        "test | RMSE_log=%.4f  RMSE_RUB_smeared=%.0f  MAPE=%.2f%%  R2_log=%.4f",
        metrics["test_rmse_log"], metrics["test_rmse_rub_smeared"],
        metrics["test_mape_rub"], metrics["test_r2_log"],
    )

    # --- Figuras y diagnóstico ------------------------------------------------ #
    figures: Dict[str, plt.Figure] = {}
    for split in ("valid", "test"):
        figures.update(make_figures(
            y_log[split], preds_store[split]["yhat_log"],
            y_rub[split], preds_store[split]["yhat_rub_smeared"], split,
        ))
    for fig_name, fig in figures.items():
        save_figure_atomic(fig, figures_dir / f"{fig_name}.png")

    prev_diag = load_prev_valid_diagnostics(reports_dir / "predictions_valid.csv")
    diag_md = build_diagnostic_md(metrics, family, smearing_fit, args.fit_on, prev_diag)
    atomic_write_text(reports_dir / "diagnostico_residuos.md", diag_md)
    atomic_write_json(reports_dir / "residual_diagnostics.json", metrics)

    # --- Predicciones y bundle reentrenado ----------------------------------- #
    for split in ("valid", "test"):
        frame = pd.DataFrame({
            "y_true_log": y_log[split],
            "yhat_log": preds_store[split]["yhat_log"],
            "y_true_rub": y_rub[split],
            "yhat_rub_smeared": preds_store[split]["yhat_rub_smeared"],
            "yhat_rub_naive": preds_store[split]["yhat_rub_naive"],
            "resid_log": y_log[split] - preds_store[split]["yhat_log"],
            "resid_rub_smeared": y_rub[split] - preds_store[split]["yhat_rub_smeared"],
        }, index=y_splits[f"y_{split}"].index)
        frame.to_csv(reports_dir / f"predictions_{split}_05.csv", index_label="row_index")

    refit_path = artifacts_dir / "model_05_refit.joblib"
    atomic_joblib_dump(
        {
            "modelo": winner,
            "clase": family,
            "params": scalar_params(winner),
            "smearing_factor": smearing_fit,
            "smearing_metodo": "duan_mean_exp_resid_log_fit",
            "n_features": int(n_features),
            "n_features_transformed": int(n_features_transformed),
            "target": TARGET_LOG,
            "fit_set": args.fit_on,
            "random_state": seed,
            "creado_por": "05_training_mlflow.py",
            "fecha_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        refit_path,
    )

# --- MLflow: tracking explícito ------------------------------------------ #
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    if args.tracking_uri.startswith("file:"):
        store_path = Path(args.tracking_uri.replace("file:", "")).resolve()
        store_path.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(store_path.as_uri())
        LOGGER.info("MLflow file-store en: %s", store_path)
    else:
        mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    LOGGER.info("Experimento MLflow: %r", args.experiment)

    common_tags = {
        "etapa": "05",
        "familia": "lineal",
        "target": TARGET_LOG,
        "proyecto": "alquileres-moscu",
        "fit_set": args.fit_on,
        "metrica_seleccion": args.metric,
        "seed": str(seed),
    }

    winner_params = flatten_params(winner, {
        "familia": "lineal",
        "modelo": family,
        "clase": type(final_estimator(winner)).__name__,
        "modelo_origen_params": params_origin,
        "target": TARGET_LOG,
        "retransformacion": "duan_smearing",
        "smearing_factor_S": smearing_fit,
        "smearing_factor_S_etapa04": bundle["smearing_04"],
        "n_features": int(n_features),
        "n_features_transformed": int(n_features_transformed),
        "random_state": seed,
        "fit_set": args.fit_on,
        "metrica_seleccion": args.metric,
        "y_scale_detectada": y_scale,
        "feature_mode": fit_mode,
        "n_train": int(Xt["train"].shape[0]),
        "n_valid": int(Xt["valid"].shape[0]),
        "n_test": int(Xt["test"].shape[0]),
        "sha256_rentals_clean": sha_value,
        "tracking_uri": args.tracking_uri,
        **split_params,
    })

    winner_artifacts: List[Tuple[Path, str]] = [
        (preprocessor_path, "model_artifacts"),
        (bundle_path, "model_artifacts"),
        (refit_path, "model_artifacts"),
        (reports_dir / "preprocessing_report.json", "reports"),
        (reports_dir / "model_selection_report.json", "reports"),
        (reports_dir / "model_selection_ranking.csv", "reports"),
        (reports_dir / "predictions_valid.csv", "reports"),
        (reports_dir / "predictions_valid_05.csv", "reports"),
        (reports_dir / "predictions_test_05.csv", "reports"),
        (reports_dir / "residual_diagnostics.json", "reports"),
        (reports_dir / "diagnostico_residuos.md", "reports"),
    ]

    run_id = log_mlflow_run(
        run_name=f"paso05-{family.lower()}-{args.fit_on}",
        tags={**common_tags, "test_evaluado": "true", "modelo": family},
        params=winner_params,
        metrics=metrics,
        figures=figures,
        artifacts=winner_artifacts,
        estimator=winner,
        log_sklearn_model=not args.no_sklearn_model,
    )
    for fig in figures.values():
        plt.close(fig)
    LOGGER.info("Run ganador registrado: %s", run_id)

    # --- Comparación opcional (valid; test NO se toca) ----------------------- #
    if args.compare_all:
        report = _safe_load_json(reports_dir / "model_selection_report.json")
        for other in sorted(ALLOWED_MODEL_CLASSES):
            if other == family:
                continue
            params_other, origin_other = family_best_params(report, other)
            est_other = build_estimator(other, params_other)
            est_other.fit(X_fit, y_log_fit)
            S_other = compute_smearing_factor(y_log_fit, est_other.predict(X_fit))
            yhat_log_v, yhat_rub_vs, yhat_rub_vn = predict_all(
                est_other, Xt["valid"], S_other
            )
            metrics_other = {
                f"valid_{k}": v for k, v in regression_metrics(
                    y_log["valid"], yhat_log_v, y_rub["valid"], yhat_rub_vs, yhat_rub_vn
                ).items()
            }
            metrics_other.update(residual_diagnostics(
                y_log["valid"], yhat_log_v, y_rub["valid"], yhat_rub_vs, "valid"
            ))
            figs_other = make_figures(
                y_log["valid"], yhat_log_v, y_rub["valid"], yhat_rub_vs, "valid"
            )
            other_path = artifacts_dir / f"model_05_{other.lower()}_comparison.joblib"
            atomic_joblib_dump(
                {"modelo": est_other, "clase": other, "smearing_factor": S_other,
                 "fit_set": args.fit_on, "uso": "comparacion_valid"},
                other_path,
            )
            other_run = log_mlflow_run(
                run_name=f"paso05-comparacion-{other.lower()}",
                tags={**common_tags, "test_evaluado": "false", "modelo": other,
                      "rol": "comparacion_valid"},
                params=flatten_params(est_other, {
                    "familia": "lineal",
                    "modelo": other,
                    "modelo_origen_params": origin_other,
                    "target": TARGET_LOG,
                    "smearing_factor_S": S_other,
                    "n_features": int(n_features),
                    "random_state": seed,
                    "fit_set": args.fit_on,
                    **split_params,
                }),
                metrics=metrics_other,
                figures=figs_other,
                artifacts=[(other_path, "model_artifacts")],
                estimator=None,
                log_sklearn_model=False,
            )
            for fig in figs_other.values():
                plt.close(fig)
            LOGGER.info("Run comparación %-15s → %s (valid only).", other, other_run)

    # --- Resumen local -------------------------------------------------------- #
    summary = {
        "experimento": args.experiment,
        "tracking_uri": args.tracking_uri,
        "run_ganador": run_id,
        "modelo": family,
        "fit_set": args.fit_on,
        "smearing_factor_S": smearing_fit,
        "valid_rmse_log": metrics["valid_rmse_log"],
        "valid_rmse_rub_smeared": metrics["valid_rmse_rub_smeared"],
        "valid_mape_rub": metrics["valid_mape_rub"],
        "test_rmse_log": metrics["test_rmse_log"],
        "test_rmse_rub_smeared": metrics["test_rmse_rub_smeared"],
        "test_mape_rub": metrics["test_mape_rub"],
        "test_evaluaciones": 1,
    }
    atomic_write_json(reports_dir / "training_mlflow_report.json", summary)

    print("\n=== Paso 5 completado ===")
    print(f"Experimento : {args.experiment}")
    print(f"Run ganador : {run_id}")
    print(f"Modelo      : {family}  (fit={args.fit_on}, S={smearing_fit:.6f})")
    print(f"valid       : RMSE_log={metrics['valid_rmse_log']:.4f}  "
          f"RMSE_RUB={metrics['valid_rmse_rub_smeared']:.0f}  "
          f"MAPE={metrics['valid_mape_rub']:.2f}%")
    print(f"test        : RMSE_log={metrics['test_rmse_log']:.4f}  "
          f"RMSE_RUB={metrics['test_rmse_rub_smeared']:.0f}  "
          f"MAPE={metrics['test_mape_rub']:.2f}%")
    print(f"Figuras     : {figures_dir}/")
    print(f"Diagnóstico : {reports_dir / 'diagnostico_residuos.md'}")
    return 0


def _safe_load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        LOGGER.warning("No se pudo leer %s; se continúa sin él.", path)
        return None


if __name__ == "__main__":
    sys.exit(main())