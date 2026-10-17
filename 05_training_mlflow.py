"""
===============================================================================
05_training_mlflow.py — Paso 5 + 6 Unificado · Entrenamiento, Diagnóstico,
                        Fórmula Ganadora, VIF, Ablación y Tracking con MLflow
===============================================================================
Proyecto: predicción de monthly_rent_rub (alquileres mensuales, Moscú).

Objetivo de este script
-----------------------
1. Reentrenar el modelo ganador de la allowlist lineal con train
   (+ valid, opcional y documentado) sobre el target log1p(monthly_rent_rub).
2. Retransformar a RUB con Duan-smearing:
       y_rub = exp(y_log) * S - 1,   S = mean(exp(residuos_log_fit)).
3. Evaluar UNA única vez sobre test (intacto hasta esta etapa).
4. Ejecutar el diagnóstico econométrico completo (Parte B del script 06/07):
   - Métricas avanzadas (MSPE, sMAPE, R² ajustado, Breusch-Pagan, Jarque-Bera,
     Durbin-Watson, aciertos ±10%/±20%) y semáforo de calidad.
   - Ecuación algebraica completa + modelo Ridge compañero interpretable + VIF
     en reports/diagnostics/formula_ganadora.txt y metricas_comparativas.{md,csv}.
   - 10 gráficas de diagnóstico en reports/diagnostics/ (01 a 09).
5. Ejecutar el estudio de ablación (Parte A del script 06/07):
   - reports/ablation_report.{json,md} y reports/figures/07_before_after.png.
6. Registrar absolutamente todo en MLflow bajo el experimento "primer parcial":
   params, metrics (básicas + econométricas), artifacts (joblibs, reportes .md,
   .txt, .csv, .json) y todas las gráficas (.png).

Uso
---
    python 05_training_mlflow.py --strict-hash
    python 05_training_mlflow.py --fit-on train_valid --compare-all
    python 05_training_mlflow.py --skip-ablation
===============================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
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
from sklearn.model_selection import KFold, learning_curve

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Constantes del proyecto (alineadas con etapas 01-06)
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

FORBIDDEN_FEATURE_TOKENS = ("id", "dq_flags", "rent_per_sqm", "monthly_rent_rub")

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

TAIL_RUB = 200_000.0
SPLIT_NAMES = ("train", "valid", "test")
COLORS = {"train": "#6b6560", "valid": "#c2410c", "test": "#1f2937"}
ALPHAS_ABLATION = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

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
    return [t for t in re.split(r"__|[^0-9a-zA-Z_]+", str(name).lower()) if t]


def leakage_violations(names: Iterable[str]) -> List[str]:
    bad: List[str] = []
    for name in names:
        tokens = set(_tokenize(name))
        lowered = str(name).lower()
        for banned in FORBIDDEN_FEATURE_TOKENS:
            if (
                lowered == banned
                or banned in tokens
                or (banned != "id" and banned in lowered)
            ):
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
        out[f"y_{split}"] = pd.Series(
            np.asarray(series).ravel(), index=df.index, name=TARGET
        )
    return out


def resolve_y_scale(y: pd.Series, mode: str) -> str:
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
    raw_names = list(getattr(preprocessor, "feature_names_in_", []))

    if raw_names and set(raw_names).issubset(set(map(str, X.columns))):
        assert_no_leakage(raw_names, "features crudas (entrada al preprocesador)")
        Xt = preprocessor.transform(X[raw_names])
        return _to_dense(Xt), [str(c) for c in raw_names], "preprocessor_applied"

    if n_features_expected is not None and X.shape[1] == n_features_expected:
        assert_no_leakage(X.columns, "features ya transformadas")
        return _to_dense(X), [str(c) for c in X.columns], "already_transformed"

    try:
        Xt = preprocessor.transform(X)
        expected = n_features_expected or getattr(Xt, "shape", (0, 0))[1]
        if Xt.shape[1] == expected:
            assert_no_leakage(X.columns, "features crudas (entrada al preprocesador)")
            return _to_dense(Xt), [str(c) for c in X.columns], "preprocessor_applied"
    except Exception as exc:  # noqa: BLE001
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
    if name in params:
        return params[name]
    for key, value in params.items():
        if key.endswith("__" + name):
            return value
    return None


def scalar_params(estimator: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in estimator.get_params().items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            out[key] = value
    return out


# --------------------------------------------------------------------------- #
# Métricas, smearing y diagnóstico (Paso 5)
# --------------------------------------------------------------------------- #
def compute_smearing_factor(y_log: np.ndarray, yhat_log: np.ndarray) -> float:
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
    out["mape_rub_naive"] = float(
        np.mean(np.abs((y_rub - yhat_rub_naive) / y_rub)) * 100.0
    )
    return out


def residual_diagnostics(
    y_log: np.ndarray,
    yhat_log: np.ndarray,
    y_rub: np.ndarray,
    yhat_rub: np.ndarray,
    prefix: str,
) -> Dict[str, float]:
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
# Figuras de residuos Paso 5: Predicho vs Real | Residuos vs Predicho | Q-Q
# --------------------------------------------------------------------------- #
def _triplet(
    y_true: np.ndarray, yhat: np.ndarray, unit: str, suptitle: str
) -> plt.Figure:
    resid = y_true - yhat
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    fig.suptitle(suptitle, fontsize=11, color=INK)

    ax = axes[0]
    lo = float(min(y_true.min(), yhat.min()))
    hi = float(max(y_true.max(), yhat.max()))
    ax.scatter(yhat, y_true, s=9, alpha=0.28, color=ACCENT, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1.1, ls="--")
    ax.set_xlabel(f"Predicho ({unit})")
    ax.set_ylabel(f"Real ({unit})")
    ax.set_title("Predicho vs Real")

    ax = axes[1]
    ax.scatter(yhat, resid, s=9, alpha=0.28, color=ACCENT, edgecolors="none")
    ax.axhline(0.0, color=INK, lw=1.0, ls="--")
    frame = pd.DataFrame({"f": yhat, "r": resid})
    try:
        frame["bin"] = pd.qcut(frame["f"], 10, duplicates="drop")
        med = frame.groupby("bin", observed=True).median(numeric_only=True)
        ax.plot(
            med["f"],
            med["r"],
            color=INK,
            lw=1.7,
            marker="o",
            ms=3.5,
            label="mediana por decil",
        )
        ax.legend(frameon=False, loc="best", fontsize=8)
    except ValueError:
        pass
    ax.set_xlabel(f"Predicho ({unit})")
    ax.set_ylabel(f"Residuo ({unit})")
    ax.set_title("Residuos vs Predicho")

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
# Narrativa del diagnóstico (Paso 5)
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
    lines.append("# Diagnóstico de residuos — Paso 5 (entrenamiento + MLflow)\n")
    lines.append(f"- Modelo reentrenado: **{family}** (allowlist lineal).")
    lines.append(f"- Conjunto de ajuste: `{fit_set}`.")
    lines.append(f"- Smearing de Duan: `S = {smearing:.6f}`.")
    lines.append(
        f"- Fecha (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}.\n"
    )

    lines.append("## Escala log (homocedasticidad)")
    lines.append(
        f"- Residuos log — media {diag['valid_resid_log_mean']:.4f}, "
        f"std {diag['valid_resid_log_std']:.4f}, "
        f"skew {diag['valid_resid_log_skew']:.3f}, "
        f"kurtosis {diag['valid_resid_log_kurtosis']:.3f} (valid)."
    )
    kurt_v = diag["valid_resid_log_kurtosis"]
    kurt_t = diag.get("test_resid_log_kurtosis", float("nan"))
    zmax_v = diag["valid_resid_log_max_abs_z"]
    zmax_t = diag.get("test_resid_log_max_abs_z", float("nan"))
    zmax = max(zmax_v, zmax_t if np.isfinite(zmax_t) else 0.0)
    qq_note = (
        "Valores >= 4 indican outliers o colas pesadas en el Q-Q."
        if zmax >= 4.0
        else "Ningún residuo supera |z| = 4: sin outliers extremos en el Q-Q."
    )
    lines.append(
        f"- Q-Q: desviación máxima |z| = {zmax_v:.2f} (valid), {zmax_t:.2f} (test). {qq_note}"
    )

    def _kurt_word(k: float) -> str:
        if not np.isfinite(k):
            return "n/d"
        if k > 0.5:
            return "colas más pesadas que la normal"
        if k < -0.5:
            return "colas más ligeras que la normal"
        return "colas cercanas a la normal"

    lines.append(
        f"- Exceso de curtosis: valid {kurt_v:.3f} ({_kurt_word(kurt_v)}), "
        f"test {kurt_t:.3f} ({_kurt_word(kurt_t)})."
    )
    bias_v = diag["valid_resid_log_mean"]
    bias_t = diag.get("test_resid_log_mean", float("nan"))
    drift_note = (
        " Sesgo positivo creciente tras el corte de train: deriva de mercado (el spline "
        "temporal mantiene constante el último nivel); `--fit-on train_valid` con el "
        "preprocesador train+valid la reduce."
        if fit_set == "train" and np.isfinite(bias_t) and bias_t > 0.01
        else ""
    )
    lines.append(
        f"- Sesgo medio en log (real − predicho): valid {bias_v:+.4f}, test {bias_t:+.4f}."
        f"{drift_note}\n"
    )

    lines.append("## Escala RUB (impacto de negocio)")
    lines.append(
        "- "
        + _hetero_sentence(
            diag["valid_hetero_log_spearman"], diag["valid_hetero_rub_spearman"]
        )
    )
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


def load_prev_valid_diagnostics(
    path: Path, family: Optional[str] = None
) -> Optional[Dict[str, float]]:
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return None
    real_col = pred_col = None
    if (
        family
        and "y_true_rub" in df.columns
        and f"{family}__pred_rub_smeared" in df.columns
    ):
        real_col, pred_col = "y_true_rub", f"{family}__pred_rub_smeared"
    else:
        real_cols = [
            c
            for c in df.columns
            if any(t in c.lower() for t in ("real", "true", "actual", "observ"))
        ]
        pred_cols = [
            c
            for c in df.columns
            if any(t in c.lower() for t in ("pred", "yhat", "y_hat", "estim"))
        ]
        if not real_cols or not pred_cols:
            return None
        real_col, pred_col = real_cols[0], pred_cols[0]
    y = pd.to_numeric(df[real_col], errors="coerce").to_numpy()
    p = pd.to_numeric(df[pred_col], errors="coerce").to_numpy()
    _my, _mp = float(np.nanmedian(np.abs(y))), float(np.nanmedian(np.abs(p)))
    if _my <= 0 or _mp <= 0 or abs(np.log10(_my / _mp)) > 1.0:
        return None
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
        "hetero": float(
            pd.Series(np.abs(y_rub - p_rub)).corr(pd.Series(p_rub), method="spearman")
        ),
        "tail_bias": float(np.mean(y_rub[tail] - p_rub[tail])) if tail.any() else 0.0,
        "mape": float(np.mean(np.abs((y_rub - p_rub) / y_rub)) * 100.0),
        "resid_log_kurt": float(pd.Series(resid_log).kurt()),
    }


# --------------------------------------------------------------------------- #
# Resolución del ganador y params por familia (etapa 04)
# --------------------------------------------------------------------------- #
def family_best_params(report: Any, family: str) -> Tuple[Dict[str, Any], str]:
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
                for key in (
                    "best_params",
                    "params",
                    "best_params_",
                    "parametros",
                    "best_hyperparams",
                    "hyperparams",
                ):
                    if isinstance(node.get(key), dict):
                        candidates.append(node[key])
                scalar_keys = lower_keys & {
                    "alpha",
                    "l1_ratio",
                    "epsilon",
                    "max_iter",
                    "tol",
                    "fit_intercept",
                    "c",
                }
                if scalar_keys and not candidates:
                    candidates.append(
                        {
                            k: v
                            for k, v in node.items()
                            if isinstance(v, (int, float, str, bool)) or v is None
                        }
                    )
            for key, value in node.items():
                walk(value, f"{path}|{str(key).lower()}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}|{i}")

    walk(report, family.lower())
    if not candidates:
        return {}, "default"
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
    if "random_state" in valid:
        est.set_params(random_state=RANDOM_STATE)
    return est


def resolve_family(estimator: Any) -> str:
    return assert_linear_allowlist(estimator, "ganador")


# =========================================================================== #
# BLOQUE PASO 6 (PARTE B): Diagnóstico completo, Fórmula Ganadora y VIF       #
# =========================================================================== #
def full_metrics(
    y_log: Any,
    p_log: Any,
    smear: float,
    n_feat: int,
    Xt: Optional[np.ndarray] = None,
    dates: Optional[pd.Series] = None,
) -> Dict[str, float]:
    y_log, p_log = np.asarray(y_log, float), np.asarray(p_log, float)
    e = y_log - p_log
    n = len(e)
    y, yh = np.expm1(y_log), np.exp(p_log) * smear - 1.0
    er = y - yh
    pe = er / y
    sst_l, sst_r = np.sum((y_log - y_log.mean()) ** 2), np.sum((y - y.mean()) ** 2)
    r2_l, r2_r = 1 - np.sum(e**2) / sst_l, 1 - np.sum(er**2) / sst_r
    z = (e - e.mean()) / e.std(ddof=1)
    m: Dict[str, float] = {
        "MSE_log": np.mean(e**2),
        "RMSE_log": np.sqrt(np.mean(e**2)),
        "MAE_log": np.mean(np.abs(e)),
        "MedAE_log": np.median(np.abs(e)),
        "R2_log": r2_l,
        "R2_ajustado_log": 1 - (1 - r2_l) * (n - 1) / max(n - n_feat - 1, 1),
        "Sesgo_log (media real-pred)": e.mean(),
        "Desv_residuos_log": e.std(ddof=1),
        "Asimetria_residuos": scipy_stats.skew(e),
        "Exceso_curtosis_residuos": scipy_stats.kurtosis(e),
        "Max_|z|_residuo": np.max(np.abs(z)),
        "MSE_RUB": np.mean(er**2),
        "RMSE_RUB": np.sqrt(np.mean(er**2)),
        "MAE_RUB": np.mean(np.abs(er)),
        "R2_RUB": r2_r,
        "Sesgo_RUB (media real-pred)": er.mean(),
        "MAPE_%": np.mean(np.abs(pe)) * 100,
        "MSPE (≈MAPE², error % al cuadrado) [%²]": np.mean((pe * 100) ** 2),
        "RMSPE_% (raíz del MSPE)": np.sqrt(np.mean((pe * 100) ** 2)),
        "sMAPE_%": np.mean(2 * np.abs(er) / (np.abs(y) + np.abs(yh))) * 100,
        "Aciertos_±10%": np.mean(np.abs(pe) <= 0.10) * 100,
        "Aciertos_±20%": np.mean(np.abs(pe) <= 0.20) * 100,
        "Spearman(|resid|, ŷ) [heteroced.]": scipy_stats.spearmanr(p_log, np.abs(e))[0],
    }
    jb = scipy_stats.jarque_bera(e)
    m["Jarque-Bera p-valor"] = jb.pvalue
    if Xt is not None:
        A = np.c_[np.ones(n), Xt]
        g = e**2
        coef = np.linalg.lstsq(A, g, rcond=None)[0]
        r2g = 1 - np.sum((g - A @ coef) ** 2) / np.sum((g - g.mean()) ** 2)
        df_bp = max(np.linalg.matrix_rank(A) - 1, 1)
        m["Breusch-Pagan p-valor"] = scipy_stats.chi2.sf(n * r2g, df_bp)
    if dates is not None:
        ee = e[np.argsort(np.asarray(dates), kind="stable")]
        m["Durbin-Watson (≈2 = sin autocorr.)"] = np.sum(np.diff(ee) ** 2) / np.sum(
            ee**2
        )
    m["n"] = n
    return {k: float(v) for k, v in m.items()}


def traffic_light(M: Dict[str, Dict[str, float]]) -> List[Tuple[str, float, float, str, str]]:
    def st(v: float, ok: float, warn: float, absval: bool = True, lower_better: bool = True) -> str:
        x = abs(v) if absval else v
        if lower_better:
            return "OK" if x <= ok else ("ATENCIÓN" if x <= warn else "PROBLEMA")
        return "OK" if x >= ok else ("ATENCIÓN" if x >= warn else "PROBLEMA")

    rows: List[Tuple[str, float, float, str, str]] = []

    def add(label: str, key_fn: Any, rule: str, ok: float, warn: float, **kw: Any) -> None:
        v, t = key_fn("valid"), key_fn("test")
        worst = max(
            [st(v, ok, warn, **kw), st(t, ok, warn, **kw)],
            key=["OK", "ATENCIÓN", "PROBLEMA"].index,
        )
        rows.append((label, v, t, rule, worst))

    add(
        "Sobreajuste: RMSE_log (valid | test) / train",
        lambda s: M[s]["RMSE_log"] / M["train"]["RMSE_log"],
        "≤1.10 OK, ≤1.25 atención",
        1.10,
        1.25,
    )
    add(
        "Sesgo medio (log)",
        lambda s: M[s]["Sesgo_log (media real-pred)"],
        "|sesgo| ≤0.02 OK, ≤0.05 atención",
        0.02,
        0.05,
    )
    add(
        "R² (log)",
        lambda s: M[s]["R2_log"],
        "≥0.90 OK, ≥0.80 atención",
        0.90,
        0.80,
        absval=False,
        lower_better=False,
    )
    add("MAPE %", lambda s: M[s]["MAPE_%"], "≤10 OK, ≤15 atención", 10, 15)
    add(
        "Asimetría de residuos",
        lambda s: M[s]["Asimetria_residuos"],
        "|skew| ≤0.5 OK, ≤1 atención",
        0.5,
        1.0,
    )
    add(
        "Exceso de curtosis",
        lambda s: M[s]["Exceso_curtosis_residuos"],
        "|kurt| ≤1 OK, ≤3 atención",
        1.0,
        3.0,
    )
    add(
        "Heterocedasticidad Spearman(|r|,ŷ)",
        lambda s: M[s]["Spearman(|resid|, ŷ) [heteroced.]"],
        "|ρ| ≤0.10 OK, ≤0.25 atención",
        0.10,
        0.25,
    )
    if "Durbin-Watson (≈2 = sin autocorr.)" in M["valid"]:
        add(
            "Autocorrelación (Durbin-Watson)",
            lambda s: M[s]["Durbin-Watson (≈2 = sin autocorr.)"] - 2,
            "|DW−2| ≤0.5 OK, ≤1 atención",
            0.5,
            1.0,
        )
    return rows


def compute_vif_table(P: Dict[str, Any]) -> pd.DataFrame:
    X = np.asarray(P["Xt"]["train"], dtype=float)
    names = P["names"]

    std = X.std(axis=0, ddof=1)
    valid_cols = std > 1e-12
    X_std = np.zeros_like(X)
    X_std[:, valid_cols] = (
        X[:, valid_cols] - X[:, valid_cols].mean(axis=0)
    ) / std[valid_cols]

    n = X.shape[0]
    R = (X_std.T @ X_std) / max(n - 1, 1)

    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R)

    vif_vals = np.diag(R_inv).copy()
    vif_vals[~valid_cols] = np.nan
    vif_vals = np.where(vif_vals < 1.0, np.inf, vif_vals)
    r2_vals = np.clip(1.0 - (1.0 / vif_vals), 0.0, 1.0)

    bloque_map: Dict[int, str] = {}
    if hasattr(P["pre"], "output_indices_"):
        for b_name, sl in P["pre"].output_indices_.items():
            if b_name != "remainder":
                for idx in range(sl.start, sl.stop):
                    bloque_map[idx] = b_name

    rows = []
    for i, term in enumerate(names):
        v = float(vif_vals[i]) if np.isfinite(vif_vals[i]) else np.inf
        est = "OK (<5)" if v < 5 else ("MODERADO (5-10)" if v < 10 else "ALTO (≥10)")
        rows.append(
            {
                "bloque": bloque_map.get(i, "otro"),
                "termino": term,
                "VIF": v,
                "R2_auxiliar": float(r2_vals[i]),
                "tolerancia (1/VIF)": float(1.0 / v)
                if np.isfinite(v) and v > 0
                else 0.0,
                "estado_vif": est,
            }
        )
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def linear_equation_string(tbl: pd.DataFrame, target_name: str = "ŷ_log") -> str:
    b0 = tbl.attrs["intercepto"]
    terms = [f"{b0:+.6f}"]
    for _, r in tbl.iterrows():
        sign = "+" if r.coeficiente >= 0 else "-"
        terms.append(f"{sign} {abs(r.coeficiente):.6f}·[{r.termino}]")

    one_line = f"{target_name} = " + " ".join(terms)
    wrapped = f"{target_name} =\n    " + "\n  ".join(terms)
    return (
        "ECUACIÓN LINEAL COMPLETA (una línea):\n"
        f"{one_line}\n\n"
        "ECUACIÓN LINEAL DESGLOSADA:\n"
        f"{wrapped}"
    )


def coefficient_table(P: Dict[str, Any], model: Any = None) -> pd.DataFrame:
    model = final_estimator(model or P["model"])
    pre, names = P["pre"], P["names"]
    coef = np.asarray(model.coef_, float).ravel()
    sd = P["Xt"]["train"].std(axis=0)
    rows = []
    if hasattr(pre, "output_indices_") and hasattr(pre, "named_transformers_"):
        for name, sl in pre.output_indices_.items():
            if name == "remainder":
                continue
            tr = pre.named_transformers_[name]
            scaler = (
                tr.named_steps.get("scaler") if hasattr(tr, "named_steps") else None
            )
            scale = getattr(scaler, "scale_", None)
            for j, idx in enumerate(range(sl.start, sl.stop)):
                per_unit = pct = np.nan
                if scale is not None and name in ("numeric", "onehot"):
                    per_unit = coef[idx] / scale[j]
                    pct = (np.exp(per_unit) - 1) * 100
                rows.append(
                    {
                        "bloque": name,
                        "termino": names[idx],
                        "coeficiente": coef[idx],
                        "desv_est_train": sd[idx],
                        "efecto_estandarizado": coef[idx] * sd[idx],
                        "coef_por_unidad_original": per_unit,
                        "efecto_pct_por_unidad": pct,
                    }
                )
    else:
        for idx, term in enumerate(names):
            rows.append(
                {
                    "bloque": "features",
                    "termino": term,
                    "coeficiente": coef[idx],
                    "desv_est_train": sd[idx],
                    "efecto_estandarizado": coef[idx] * sd[idx],
                    "coef_por_unidad_original": np.nan,
                    "efecto_pct_por_unidad": np.nan,
                }
            )
    df = pd.DataFrame(rows)
    df.attrs["intercepto"] = float(np.ravel(model.intercept_)[0])
    return df


def formula_text(
    tbl: pd.DataFrame,
    title: str,
    smear: float,
    vif_df: Optional[pd.DataFrame] = None,
) -> str:
    b0 = tbl.attrs["intercepto"]
    L = [
        f"{'=' * 78}",
        title,
        f"{'=' * 78}",
        "(One-hot con drop='first': la categoría base —la omitida de cada variable— queda en el intercepto.)",
        "ŷ_log = log1p(renta_estimada) = b0 + Σ ( coeficiente_i × x_i )",
        "donde x_i son las columnas YA TRANSFORMADAS por el preprocesador de la etapa 03",
        "(imputación + escalado + splines + one-hot + target encoding).",
        "",
        linear_equation_string(tbl, target_name="log1p(monthly_rent_rub)"),
        "",
        f"  b0 (intercepto) = {b0:+.6f}",
    ]
    for bloque, g in tbl.groupby("bloque", sort=False):
        L.append(f"\n  --- bloque «{bloque}» ({len(g)} términos) ---")
        for _, r in g.iterrows():
            L.append(f"  {r.coeficiente:+12.6f} × {r.termino}")
    L += [
        "",
        "Retransformación a RUB (corrección de Duan):",
        f"  renta_RUB = exp(ŷ_log) × S − 1,   con S = {smear:.6f}",
        "",
        "Lectura de efectos (solo bloques numeric/onehot; ver coeficientes.csv):",
        "  efecto_pct_por_unidad = (exp(coef/escala) − 1) × 100  → % de cambio de la renta por unidad",
        "  original, manteniendo lo demás fijo. CUIDADO: total_area, log_metro y to_center_km entran a la vez",
        "  como término lineal Y como spline, así que su efecto total está repartido entre ambos y el %",
        "  del término lineal es solo una parte. Para splines y tiempo NO hay efecto por unidad: la",
        "  contribución es una curva (ver gráficas) y los coeficientes sueltos no se interpretan.",
    ]
    mx = tbl.coeficiente.abs().max()
    if mx > 5:
        L += [
            "",
            f"AVISO DE COLINEALIDAD: |coef| máximo = {mx:.1f}. Hay coeficientes grandes que se cancelan",
            "entre sí (p. ej. log_metro aparece en el bloque numérico y en splines). La fórmula es exacta",
            "para predecir, pero los coeficientes individuales NO son interpretables; usa la fórmula",
            "del Ridge compañero (más abajo) para explicar efectos.",
        ]
    if vif_df is not None:
        L += [
            "",
            f"{'-' * 78}",
            "FACTOR DE INFLACIÓN DE LA VARIANZA (VIF) — TODAS LAS VARIABLES (Train)",
            f"{'-' * 78}",
            "Regla práctica: VIF < 5 (baja colinealidad) | 5 ≤ VIF < 10 (moderada) | VIF ≥ 10 (alta).",
            f"{'Término':<42s} | {'Bloque':<10s} | {'VIF':>12s} | {'R² aux':>8s} | {'Estado'}",
            "-" * 92,
        ]
        for _, r in vif_df.iterrows():
            v_str = (
                "INF (>1e6)"
                if (np.isinf(r.VIF) or r.VIF > 1e6)
                else f"{r.VIF:12.2f}"
            )
            L.append(
                f"{str(r.termino)[:42]:<42s} | {str(r.bloque)[:10]:<10s} | {v_str:>12s} | {r.R2_auxiliar:8.4f} | {r.estado_vif}"
            )
    return "\n".join(L)


def ridge_companion(P: Dict[str, Any]) -> Tuple[Ridge, float]:
    best = None
    for a in (0.01, 0.1, 1.0, 10.0, 100.0):
        m = Ridge(alpha=a).fit(P["Xt"]["train"], P["ylog"]["train"].to_numpy())
        rm = float(
            np.sqrt(
                np.mean(
                    (P["ylog"]["valid"].to_numpy() - m.predict(P["Xt"]["valid"])) ** 2
                )
            )
        )
        if best is None or rm < best[0]:
            best = (rm, a, m)
    return best[2], best[1]


def _roll(x: np.ndarray, y: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tendencia media suavizada sin 'dientes de sierra'.
    Usa ventana adaptativa (~25% de la muestra) con doble pase (triangular/gaussiano)
    para reflejar la verdadera tendencia de homocedasticidad sin sobreajustar ruido local.
    """
    o = np.argsort(x)
    xs, ys = x[o], y[o]
    n = len(xs)
    if k is None:
        k = max(int(n * 0.25), 150)
    min_p = max(k // 5, 30)
    # Doble media móvil centrada = kernel triangular suave (elimina oscilaciones bruscas)
    s1 = pd.Series(ys).rolling(k, center=True, min_periods=min_p).mean()
    s2 = s1.rolling(max(k // 2, 50), center=True, min_periods=min_p // 2).mean()
    return xs, s2.to_numpy()


def make_diagnostic_figures(
    P: Dict[str, Any],
    M: Dict[str, Dict[str, float]],
    tbl: pd.DataFrame,
    vif_df: pd.DataFrame,
    out: Path,
    lc_reps: int = 3,
) -> List[Path]:
    out.mkdir(parents=True, exist_ok=True)
    files: List[Path] = []
    R = {s: np.asarray(P["ylog"][s]) - P["pred"][s] for s in SPLIT_NAMES}
    idx = {s: np.asarray(P["ylog"][s].index) for s in SPLIT_NAMES}

    def pair(
        fname: str,
        title: str,
        draw: Any,
        size: Tuple[float, float] = (13.5, 5.0),
        sharex: bool = True,
        sharey: bool = True,
    ) -> None:
        fig, axs = plt.subplots(1, 2, figsize=size, sharex=sharex, sharey=sharey)
        for ax, s in zip(axs, ("valid", "test")):
            draw(ax, s)
            ax.tick_params(labelbottom=True, labelleft=True)
        fig.suptitle(title, fontsize=13, y=1.02)
        fig.tight_layout()
        p = out / fname
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        files.append(p)

    # --- Límites globales para compartir escalas exactas entre valid y test ---
    all_r = np.concatenate([R["valid"], R["test"]])
    r_lim = float(np.max(np.abs(all_r)) * 1.06)
    # Bins óptimos por Freedman-Diaconis (acotado entre 22 y 28 para evitar picos artificiales)
    n_bins = int(np.clip( len(np.histogram_bin_edges(all_r, bins="fd")) - 1, 22, 28 ))
    shared_bin_edges = np.linspace(-r_lim, r_lim, n_bins + 1)

    all_f = np.concatenate([P["pred"]["valid"], P["pred"]["test"]])
    f_min, f_max = float(all_f.min() - 0.15), float(all_f.max() + 0.15)

    def hist(ax: plt.Axes, s: str) -> None:
        r = R[s]
        mu, sd = r.mean(), r.std(ddof=1)
        ax.hist(
            r,
            bins=shared_bin_edges,
            density=True,
            color="#e0a48a",
             alpha=0.80,
            edgecolor="white",
            label=f"Histograma ({n_bins} bins FD)",
        )
        x = np.linspace(-r_lim, r_lim, 350)
        # Curva KDE real de los datos para ver la forma suave sin el ruido del binning
        kde = scipy_stats.gaussian_kde(r, bw_method="scott")
        ax.plot(
            x,
            kde(x),
            color="#9a3412",
            lw=2.0,
            ls="-",
            label="Densidad empírica (KDE)",
        )
        ax.plot(
            x,
            scipy_stats.norm.pdf(x, mu, sd),
            "k--",
            lw=1.8,
            label=f"Normal teórica (μ={mu:.3f}, σ={sd:.3f})",
        )
        ax.axvline(0, color="#2563eb", ls=":", lw=2, label="Cero (sin sesgo)")
        ax.axvline(mu, color="#c2410c", lw=2, label=f"Media real ({mu:+.3f})")
        ax.set_xlim(-r_lim, r_lim)
        ax.set_ylim(0, 5.6)
        ax.set_title(
            f"{s} — skew={scipy_stats.skew(r):.3f} | exceso curtosis={scipy_stats.kurtosis(r):.3f}"
        )
        ax.set_xlabel("Residuo = real − predicho (log1p RUB)")
        ax.set_ylabel("Densidad")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.92)

    pair(
        "01_histograma_residuos_valid_vs_test.png",
        "Distribución de residuos (escala log)",
        hist,
    )

    def qq(ax: plt.Axes, s: str) -> None:
        z = (R[s] - R[s].mean()) / R[s].std(ddof=1)
        (osm, osr), (_sl, _ic, rr) = scipy_stats.probplot(z, dist="norm")
        ax.scatter(
            osm, osr, s=8, alpha=0.5, color=COLORS["valid" if s == "valid" else "test"]
        )
        lim = [-3.6, 3.6]
        ax.plot(lim, lim, color="#d62728", lw=2, label="Normal perfecta (y = x)")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(f"{s} — QQ-plot (R² del ajuste = {rr ** 2:.4f})")
        ax.set_xlabel("Cuantiles teóricos (Normal)")
        ax.set_ylabel("Cuantiles de residuos estandarizados")
        ax.legend(fontsize=8, loc="upper left")

    pair(
        "02_qqplot_valid_vs_test.png",
        "QQ-plot de residuos: colas pesadas se ven como curvas lejos de la diagonal",
        qq,
    )

    # Techo común para Scale-Location (deja margen superior limpio para la leyenda y etiquetas)
    z_max_global = max(
        float(np.max(np.sqrt(np.abs((R[s] - R[s].mean()) / R[s].std(ddof=1)))))
        for s in ("valid", "test")
    )

    def scale_loc(ax: plt.Axes, s: str) -> None:
        f, r = P["pred"][s], R[s]
        z = np.sqrt(np.abs((r - r.mean()) / r.std(ddof=1)))
        ax.scatter(f, z, s=9, alpha=0.25, color="#555555", edgecolors="none")
        xs, ys = _roll(f, z)
        ax.axhline(
            0.822,
            color="#1f2937",
            ls=":",
            lw=1.2,
            alpha=0.7,
            label="Esperado Normal (≈0.82)",
        )
        ax.plot(
            xs,
            ys,
            color="#d62728",
            lw=2.5,
            label="Tendencia media (horizontal = homocedástico)",
        )
        # Posicionar etiquetas de los 3 mayores residuos evitando solapamiento entre sí y con la leyenda
        top3 = np.argsort(-z)[:3]
        offsets = [(-6, 5), (6, 4), (-6, -10)]
        for rank, i in enumerate(top3):
            dx, dy = offsets[rank % len(offsets)]
            ha = "right" if dx < 0 else "left"
            ax.annotate(
                str(idx[s][i]),
                (f[i], z[i]),
                fontsize=7.5,
                color="#2b2b2b",
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
            )
        bp_p = M[s].get("Breusch-Pagan p-valor", float("nan"))
        bp_txt = f" | BP p={bp_p:.2f}" if np.isfinite(bp_p) else ""
        ax.set_xlim(f_min, f_max)
        ax.set_ylim(-0.04, z_max_global + 0.28)
        ax.set_title(
            f"{s} — Scale-Location (ρ={M[s]['Spearman(|resid|, ŷ) [heteroced.]']:+.2f}{bp_txt})"
        )
        ax.set_xlabel("Valores ajustados ŷ (log1p RUB)")
        ax.set_ylabel("√|residuo estandarizado|")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.92)

    pair(
        "03_scale_location_valid_vs_test.png",
        "Scale-Location (homocedasticidad)",
        scale_loc,
    )

    def rvf(ax: plt.Axes, s: str) -> None:
        f, r = P["pred"][s], R[s]
        ax.scatter(
            f, r, s=8, alpha=0.28, color=COLORS["valid" if s == "valid" else "test"]
        )
        xs, ys = _roll(f, r)
        ax.plot(xs, ys, color="#d62728", lw=2.5, label="Tendencia media")
        ax.axhline(0, color="k", ls="--", lw=1.2)
        ax.set_xlim(f_min, f_max)
        ax.set_ylim(-r_lim, r_lim)
        ax.set_title(f"{s} — Residuo vs ajustado")
        ax.set_xlabel("ŷ (log1p RUB)")
        ax.set_ylabel("Residuo (real − predicho)")
        ax.legend(fontsize=8, loc="upper right")

    pair(
        "04_residuo_vs_ajustado_valid_vs_test.png",
        "Residuos vs valores ajustados (linealidad / patrones)",
        rvf,
    )

    def rvp(ax: plt.Axes, s: str) -> None:
        y, f = np.asarray(P["ylog"][s]), P["pred"][s]
        ax.scatter(
            f, y, s=8, alpha=0.28, color=COLORS["valid" if s == "valid" else "test"]
        )
        lim = [f_min, f_max]
        ax.plot(lim, lim, color="#d62728", lw=2, label="Predicción perfecta")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(
            f"{s} — Real vs predicho (R²={M[s]['R2_log']:.3f}, RMSE={M[s]['RMSE_log']:.4f})"
        )
        ax.set_xlabel("Predicho (log1p RUB)")
        ax.set_ylabel("Real (log1p RUB)")
        ax.legend(fontsize=8, loc="upper left")

    pair("05_real_vs_predicho_valid_vs_test.png", "Real vs predicho", rvp)

    if all(P["dates"][s] is not None for s in ("valid", "test")):

        def tdrift(ax: plt.Axes, s: str) -> None:
            d = pd.DataFrame(
                {
                    "m": pd.to_datetime(P["dates"][s]).dt.to_period("M").astype(str),
                    "r": R[s],
                }
            )
            g = d.groupby("m").r.agg(["mean", "std"])
            ax.errorbar(
                range(len(g)),
                g["mean"],
                yerr=g["std"],
                fmt="o-",
                color="#c2410c",
                capsize=4,
                label="media ± 1σ del residuo",
            )
            ax.axhline(0, color="k", ls="--")
            ax.set_xticks(range(len(g)))
            ax.set_xticklabels(g.index, rotation=45)
            ax.set_title(f"{s} — sesgo por mes (deriva temporal)")
            ax.set_ylabel("Residuo medio (log)")
            ax.legend(fontsize=8)

        pair(
            "06_residuos_en_el_tiempo_valid_vs_test.png",
            "Deriva temporal: ¿el sesgo crece con el tiempo?",
            tdrift,
        )

    # --- 07. Curva de aprendizaje estilo CV ---
    LC_SEEDS = 5
    tr_all, cv_all = [], []
    for seed_lc in range(LC_SEEDS):
        n_lc, tr_lc, cv_lc = learning_curve(
            clone(P["model"]),
            P["Xt"]["train"],
            np.asarray(P["ylog"]["train"]),
            cv=KFold(5, shuffle=True, random_state=seed_lc),
            train_sizes=np.linspace(0.15, 1.0, 8),
            shuffle=True,
            random_state=seed_lc,
            scoring="neg_root_mean_squared_error",
            n_jobs=1,
        )
        tr_all.append(-tr_lc.mean(axis=1))
        cv_all.append(-cv_lc.mean(axis=1))
    tr_m, cv_m = np.mean(tr_all, axis=0), np.mean(cv_all, axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    ax[0].plot(
        n_lc, tr_m, "o-", color="#1f77b4", lw=2.5, label="RMSE Entrenamiento (Train)"
    )
    ax[0].plot(
        n_lc, cv_m, "s-", color="#c2410c", lw=2.5, label="RMSE Validación Cruzada (CV)"
    )
    ax[0].fill_between(n_lc, tr_m, cv_m, color="#c2410c", alpha=0.10)
    ax[0].set_title("Curva de Aprendizaje (Diagnóstico de Overfitting)")
    ax[0].set_xlabel("Cantidad de filas de entrenamiento")
    ax[0].set_ylabel("Error RMSE (escala log)")
    ax[0].legend()
    ax[0].text(
        0.30,
        0.04,
        "CV = KFold(5) mezclado, 5 semillas promediadas: aísla el efecto del tamaño de muestra.\n"
        "La brecha fuera de tiempo (train→valid/test) se ve en la figura 07b.",
        transform=ax[0].transAxes,
        fontsize=8,
        color="#555",
    )
    vals_lc = [M[s]["RMSE_log"] for s in SPLIT_NAMES]
    ax[1].bar(
        ["Fit (Entrenamiento)", "Valid", "Test"],
        vals_lc,
        color=["#6b6560", "#c2410c", "#2a2a2a"],
        width=0.5,
    )
    for i, v in enumerate(vals_lc):
        ax[1].text(
            i, v + max(vals_lc) * 0.01, f"{v:.4f}", ha="center", fontweight="bold"
        )
    ax[1].set_ylim(0, max(vals_lc) * 1.25)
    ax[1].set_title("Comparación de Error RMSE_log entre Particiones")
    ax[1].set_ylabel("RMSE (log1p(RUB))")
    fig.tight_layout()
    p = out / "07_curva_aprendizaje_estilo_original.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p)

    # --- 07b. Curva de aprendizaje fuera de tiempo ---
    Xtr, ytr = P["Xt"]["train"], np.asarray(P["ylog"]["train"])
    n = len(ytr)
    sizes = np.unique((np.linspace(0.10, 1.0, 8) * n).astype(int))
    rng = np.random.default_rng(42)
    curves: Dict[str, List[Tuple[float, float]]] = {s: [] for s in SPLIT_NAMES}
    for sz in sizes:
        acc: Dict[str, List[float]] = {s: [] for s in SPLIT_NAMES}
        for _ in range(1 if sz == n else lc_reps):
            sub = rng.choice(n, sz, replace=False)
            m = clone(P["model"]).fit(Xtr[sub], ytr[sub])
            acc["train"].append(
                float(np.sqrt(np.mean((ytr[sub] - m.predict(Xtr[sub])) ** 2)))
            )
            for s in ("valid", "test"):
                acc[s].append(
                    float(
                        np.sqrt(
                            np.mean(
                                (np.asarray(P["ylog"][s]) - m.predict(P["Xt"][s])) ** 2
                            )
                        )
                    )
                )
        for s in SPLIT_NAMES:
            curves[s].append((float(np.mean(acc[s])), float(np.std(acc[s]))))
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    for s, mk in (("train", "o"), ("valid", "s"), ("test", "^")):
        mu = np.array([c[0] for c in curves[s]])
        sd = np.array([c[1] for c in curves[s]])
        ax[0].plot(sizes, mu, marker=mk, color=COLORS[s], lw=2.2, label=f"RMSE {s}")
        ax[0].fill_between(sizes, mu - sd, mu + sd, color=COLORS[s], alpha=0.12)
    ax[0].set_title(
        "Curva fuera de tiempo: entrena con submuestras de train, evalúa en valid y test (posteriores)"
    )
    ax[0].set_xlabel("Filas de entrenamiento")
    ax[0].set_ylabel("RMSE (log1p RUB)")
    ax[0].legend()
    vals = [M[s]["RMSE_log"] for s in SPLIT_NAMES]
    ax[1].bar(
        ["Fit (train)", "Valid", "Test"],
        vals,
        color=[COLORS[s] for s in SPLIT_NAMES],
        width=0.5,
    )
    for i, v in enumerate(vals):
        ax[1].text(i, v + max(vals) * 0.01, f"{v:.4f}", ha="center", fontweight="bold")
    ax[1].set_ylim(0, max(vals) * 1.25)
    ax[1].set_title("RMSE_log por partición")
    ax[1].set_ylabel("RMSE (log1p RUB)")
    fig.tight_layout()
    p = out / "07b_curva_aprendizaje_fuera_de_tiempo.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p)

    # --- 08. Coeficientes más influyentes ---
    top = (
        tbl.reindex(tbl.efecto_estandarizado.abs().sort_values(ascending=False).index)
        .head(20)
        .iloc[::-1]
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(
        top.termino,
        top.efecto_estandarizado,
        color=np.where(top.efecto_estandarizado >= 0, "#c2410c", "#2563eb"),
    )
    ax.axvline(0, color="k")
    ax.set_xlabel(
        "coeficiente × desv. estándar (efecto en log-renta por 1σ del término)"
    )
    ax.set_title("Top 20 términos por efecto estandarizado (modelo ganador)")
    fig.tight_layout()
    p = out / "08_coeficientes_top20.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    files.append(p)

    # --- 09. Gráfico de VIF de TODAS las variables ---
    vplot = vif_df.copy().iloc[::-1]
    CAP_VIF = 100.0
    vals_raw = vplot["VIF"].to_numpy()
    vals_clipped = np.clip(
        np.where(np.isinf(vals_raw), CAP_VIF, vals_raw), 1.0, CAP_VIF
    )

    bar_colors = np.where(
        vals_raw >= 10,
        "#d62728",
        np.where(vals_raw >= 5, "#d97706", "#2563eb"),
    )

    fig_h = max(6.5, 0.30 * len(vplot) + 1.8)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    bars = ax.barh(
        vplot["termino"],
        vals_clipped,
        color=bar_colors,
        alpha=0.88,
        edgecolor="white",
    )

    ax.set_xscale("log")
    ax.axvline(5, color="#d97706", ls="--", lw=1.8, label="Umbral moderado (VIF = 5)")
    ax.axvline(10, color="#d62728", ls="--", lw=2.0, label="Umbral severo (VIF = 10)")

    for bar, raw_v, clip_v in zip(bars, vals_raw, vals_clipped):
        lbl = (
            f" >{CAP_VIF:.0f} (colin. estructural)"
            if (np.isinf(raw_v) or raw_v > CAP_VIF)
            else f" {raw_v:.2f}"
        )
        ax.text(
            clip_v * 1.04,
            bar.get_y() + bar.get_height() / 2,
            lbl,
            va="center",
            ha="left",
            fontsize=8,
            color="#1f2937",
            fontweight="bold" if raw_v >= 10 else "normal",
        )

    ax.set_xlim(0.9, CAP_VIF * 2.8)
    ax.set_xlabel("VIF (escala logarítmica, truncada en 100 para visualización)")
    ax.set_title(
        f"Factor de Inflación de la Varianza (VIF) — Todas las variables (n = {len(vplot)})"
    )
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    p = out / "09_vif_todas_las_variables.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    files.append(p)

    return files


def run_diagnostics_06(
    *,
    winner: Any,
    family: str,
    preprocessor: Any,
    smearing_fit: float,
    X_splits: Dict[str, pd.DataFrame],
    Xt: Dict[str, np.ndarray],
    y_splits: Dict[str, pd.Series],
    y_log: Dict[str, np.ndarray],
    out_names: List[str],
    fit_on: str,
    diag_dir: Path,
    lc_reps: int = 3,
) -> Tuple[List[Path], Dict[str, float]]:
    """
    Ejecuta la Parte B del script 06/07 y guarda todos los archivos en diag_dir
    (reports/diagnostics/). Devuelve la lista de archivos generados y métricas
    normalizadas para MLflow.
    """
    dates: Dict[str, Optional[pd.Series]] = {s: None for s in SPLIT_NAMES}
    if CLEAN_CSV.exists():
        try:
            d = pd.read_csv(
                CLEAN_CSV, usecols=["date_posted"], parse_dates=["date_posted"]
            )["date_posted"]
            for s in SPLIT_NAMES:
                if set(X_splits[f"X_{s}"].index).issubset(set(d.index)):
                    dates[s] = d.loc[X_splits[f"X_{s}"].index].reset_index(drop=True)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("No se pudieron cargar fechas para Durbin-Watson/deriva: %s", exc)

    P = {
        "pre": preprocessor,
        "model": final_estimator(winner),
        "smear": float(smearing_fit),
        "X": {s: X_splits[f"X_{s}"] for s in SPLIT_NAMES},
        "Xt": {s: Xt[s] for s in SPLIT_NAMES},
        "ylog": {
            s: pd.Series(y_log[s], index=y_splits[f"y_{s}"].index) for s in SPLIT_NAMES
        },
        "pred": {
            s: np.asarray(winner.predict(Xt[s]), dtype=float) for s in SPLIT_NAMES
        },
        "dates": dates,
        "names": out_names,
    }

    nf = P["Xt"]["train"].shape[1]
    tot = sum(len(P["ylog"][s]) for s in SPLIT_NAMES)
    M = {
        s: full_metrics(
            P["ylog"][s], P["pred"][s], P["smear"], nf, P["Xt"][s], P["dates"][s]
        )
        for s in SPLIT_NAMES
    }
    diag_dir.mkdir(parents=True, exist_ok=True)
    generated_files: List[Path] = []

    # 1) Tabla de métricas comparativas (.csv y .md)
    df = pd.DataFrame(M)[list(SPLIT_NAMES)]
    csv_metrics_path = diag_dir / "metricas_comparativas.csv"
    df.to_csv(csv_metrics_path, encoding="utf-8-sig")
    generated_files.append(csv_metrics_path)

    tl = traffic_light(M)
    L = [
        f"# Diagnóstico completo — modelo ganador: **{family}** ({nf} features)\n",
        "Split: "
        + " / ".join(
            f"{s} n={len(P['ylog'][s])} ({100 * len(P['ylog'][s]) / tot:.1f} %)"
            for s in SPLIT_NAMES
        )
        + "\n",
        f"Conjunto de ajuste en este entrenamiento: `{fit_on}`.\n",
        "## 1. Métricas por partición\n",
        "| Métrica | train | valid | test |",
        "|---|---:|---:|---:|",
    ]
    for k, r in df.iterrows():
        fmt = (
            (lambda v: f"{v:,.0f}")
            if k
            in (
                "n",
                "MSE_RUB",
                "RMSE_RUB",
                "MAE_RUB",
                "Sesgo_RUB (media real-pred)",
            )
            else (lambda v: f"{v:.4g}" if abs(v) < 1e-3 else f"{v:.4f}")
        )
        L.append(f"| {k} | {fmt(r['train'])} | {fmt(r['valid'])} | {fmt(r['test'])} |")
    L += [
        "\n> **MAPE²:** el cuadrado literal del MAPE no tiene interpretación estándar; se reporta el **MSPE** "
        "(media del error porcentual al cuadrado, en %²) y su raíz RMSPE, que son su equivalente correcto.\n",
        "## 2. Semáforo de calidad (reglas prácticas, NO estándares universales)\n",
        "| Criterio | valid | test | Regla | Estado |",
        "|---|---:|---:|---|---|",
    ]
    for lab, v, t, rule, est in tl:
        L.append(f"| {lab} | {v:.4f} | {t:.4f} | {rule} | **{est}** |")
    L += [
        "\n**Cómo leerlo:** con n ≈ 2 000 los tests de Jarque-Bera y Breusch-Pagan rechazan casi siempre "
        "(son muy sensibles); guíate por magnitud (skew, curtosis, ρ) y por las gráficas, no solo por el p-valor.\n",
        "Valid y test deberían dar métricas parecidas entre sí y cercanas a train; una brecha grande entre "
        "train y valid/test indica sobreajuste, y un sesgo creciente por mes indica deriva temporal.\n",
    ]
    md_metrics_path = diag_dir / "metricas_comparativas.md"
    atomic_write_text(md_metrics_path, "\n".join(L))
    generated_files.append(md_metrics_path)

    # 2) Fórmula ganadora + VIF + Ridge compañero
    vif_df = compute_vif_table(P)
    vif_csv_path = diag_dir / "vif_variables.csv"
    vif_df.to_csv(vif_csv_path, index=False, encoding="utf-8-sig")
    generated_files.append(vif_csv_path)

    tbl = coefficient_table(P)
    b0_ganador = tbl.attrs["intercepto"]
    tbl = tbl.merge(
        vif_df[["termino", "VIF", "R2_auxiliar", "estado_vif"]],
        on="termino",
        how="left",
    )
    tbl.attrs["intercepto"] = b0_ganador
    coef_ganador_path = diag_dir / "coeficientes_ganador.csv"
    tbl.to_csv(coef_ganador_path, index=False, encoding="utf-8-sig")
    generated_files.append(coef_ganador_path)

    txt = formula_text(
        tbl, f"FÓRMULA DEL MODELO GANADOR: {family}", P["smear"], vif_df=vif_df
    )

    rid, a = ridge_companion(P)
    pr_v, pr_t = (
        float(np.sqrt(np.mean((np.asarray(P["ylog"][s]) - rid.predict(P["Xt"][s])) ** 2)))
        for s in ("valid", "test")
    )
    tbl_r = coefficient_table(P, rid)
    b0_ridge = tbl_r.attrs["intercepto"]
    tbl_r = tbl_r.merge(
        vif_df[["termino", "VIF", "R2_auxiliar", "estado_vif"]],
        on="termino",
        how="left",
    )
    tbl_r.attrs["intercepto"] = b0_ridge
    coef_ridge_path = diag_dir / "coeficientes_ridge_companero.csv"
    tbl_r.to_csv(coef_ridge_path, index=False, encoding="utf-8-sig")
    generated_files.append(coef_ridge_path)

    txt_r = formula_text(
        tbl_r,
        f"FÓRMULA COMPAÑERA INTERPRETABLE: Ridge(alpha={a})  —  RMSE_log valid {pr_v:.4f} | test {pr_t:.4f} "
        f"(ganador: valid {M['valid']['RMSE_log']:.4f} | test {M['test']['RMSE_log']:.4f})",
        P["smear"],
    )
    formula_path = diag_dir / "formula_ganadora.txt"
    atomic_write_text(formula_path, txt + "\n\n\n" + txt_r)
    generated_files.append(formula_path)

    # 3) Gráficas 01 a 09 en reports/diagnostics/
    fig_files = make_diagnostic_figures(P, M, tbl, vif_df, diag_dir, lc_reps=lc_reps)
    generated_files.extend(fig_files)

    # 4) Extraer métricas econométricas limpias para MLflow
    mlflow_extra_metrics: Dict[str, float] = {}
    key_map = {
        "R2_ajustado_log": "r2_adj_log",
        "MedAE_log": "medae_log",
        "MSPE (≈MAPE², error % al cuadrado) [%²]": "mspe_pct2",
        "RMSPE_% (raíz del MSPE)": "rmspe_pct",
        "sMAPE_%": "smape_pct",
        "Aciertos_±10%": "aciertos_10pct",
        "Aciertos_±20%": "aciertos_20pct",
        "Jarque-Bera p-valor": "jarque_bera_pvalue",
        "Breusch-Pagan p-valor": "breusch_pagan_pvalue",
        "Durbin-Watson (≈2 = sin autocorr.)": "durbin_watson",
    }
    for s in SPLIT_NAMES:
        for orig_k, clean_k in key_map.items():
            if orig_k in M[s] and np.isfinite(M[s][orig_k]):
                mlflow_extra_metrics[f"{s}_{clean_k}"] = float(M[s][orig_k])

    LOGGER.info(
        "Diagnóstico (Paso 6 Parte B) guardado en %s (%d archivos).",
        diag_dir,
        len(generated_files),
    )
    return generated_files, mlflow_extra_metrics


# =========================================================================== #
# BLOQUE PASO 6 (PARTE A): Estudio de Ablación                                #
# =========================================================================== #
def _load_p3_module() -> Optional[Any]:
    p3_path = Path(__file__).with_name("03_preprocessing.py")
    if not p3_path.exists():
        p3_path = Path("03_preprocessing.py")
    if not p3_path.exists():
        LOGGER.warning(
            "No se encontró 03_preprocessing.py; se omite el estudio de ablación (Parte A)."
        )
        return None
    spec = importlib.util.spec_from_file_location("p3", p3_path)
    if spec is None or spec.loader is None:
        return None
    p3 = importlib.util.module_from_spec(spec)
    sys.modules["p3"] = p3
    spec.loader.exec_module(p3)
    return p3


def _ablation_metrics(
    y_log: pd.Series, pred_log: np.ndarray, smear: float = 1.0
) -> Dict[str, float]:
    r = y_log.values - pred_log
    y_rub = np.expm1(y_log.values)
    yhat_rub = np.expm1(pred_log) * smear
    return {
        "n": int(len(r)),
        "rmse_log": float(np.sqrt(np.mean(r**2))),
        "bias_log": float(np.mean(r)),
        "mape_pct": float(np.mean(np.abs(y_rub - yhat_rub) / y_rub) * 100),
        "mae_rub": float(np.mean(np.abs(y_rub - yhat_rub))),
        "rho_abs_resid_vs_fit": float(scipy_stats.spearmanr(pred_log, np.abs(r))[0])
        if len(r) > 2
        else float("nan"),
    }


def _segment_table(
    df_part: pd.DataFrame, y: pd.Series, pred: np.ndarray, edges: np.ndarray, col: str
) -> List[Dict[str, Any]]:
    seg = pd.cut(df_part[col], bins=edges, include_lowest=True)
    r = y.values - pred
    out = []
    for lvl, idx in pd.Series(np.arange(len(r))).groupby(seg.values, observed=True):
        rr = r[idx.values]
        out.append(
            {
                "segmento": str(lvl),
                "n": int(len(rr)),
                "rmse_log": float(np.sqrt(np.mean(rr**2))),
                "std_resid": float(np.std(rr)),
                "bias_log": float(np.mean(rr)),
            }
        )
    return out


def _luxury_sensitivity(
    y: pd.Series, pred: np.ndarray, q: float = 0.95
) -> Dict[str, Any]:
    cut = np.quantile(pred, q)
    keep = pred <= cut
    m_all, m_keep = _ablation_metrics(y, pred), _ablation_metrics(y[keep], pred[keep])
    return {
        "umbral_yhat_log": float(cut),
        "n_excluidas": int((~keep).sum()),
        "todas": m_all,
        "sin_top": m_keep,
    }


def run_ablation_06(reports_dir: Path, figures_dir: Path) -> List[Path]:
    """
    Ejecuta la Parte A del script 06/07 (estudio de ablación) y guarda:
      - reports/ablation_report.json
      - reports/ablation_report.md
      - reports/figures/07_before_after.png
    """
    p3 = _load_p3_module()
    if p3 is None:
        return []

    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rep = reports_dir / "preprocessing_report.json"
    base_cfg = p3.Config()
    if rep.exists():
        sc = json.loads(rep.read_text(encoding="utf-8")).get("split_cutoffs")
        if sc:
            base_cfg = p3.Config(
                split_mode="fixed",
                train_end=sc["split_train_end"],
                valid_end=sc["split_valid_end"],
            )

    df = pd.read_csv(base_cfg.input_csv, parse_dates=["date_posted"])
    df = p3.add_engineered_features(df)
    splits = p3.temporal_split(df, base_cfg)
    drop = [base_cfg.target, "date_posted", "building_year"] + list(base_cfg.forbidden)
    ys = {k: np.log1p(v[base_cfg.target].astype(float)) for k, v in splits.items()}

    def fit_variant(cfg: Any, lam: float = 0.0) -> Dict[str, Any]:
        feats = p3.build_feature_lists(
            cfg.include_metro_distance, cfg.time_mode, cfg.area_spline
        )
        pre = p3.build_preprocessor(feats)
        Xs = {k: v.drop(columns=drop) for k, v in splits.items()}
        Xtr = pre.fit_transform(Xs["train"], ys["train"])
        Xva, Xte = pre.transform(Xs["valid"]), pre.transform(Xs["test"])

        yrub = {k: np.expm1(v.values) for k, v in ys.items()}
        if lam == 0.0:
            f, finv = (lambda y: np.log(y)), (lambda z: np.exp(z))
        else:
            f = lambda y: (y**lam - 1) / lam
            finv = lambda z: np.maximum(lam * z + 1, 1e-9) ** (1 / lam)
        ztr = f(yrub["train"])

        best = None
        for a in ALPHAS_ABLATION:
            m = Ridge(alpha=a).fit(Xtr, ztr)
            pv = np.log1p(finv(m.predict(Xva)))
            rm = float(np.sqrt(np.mean((ys["valid"].values - pv) ** 2)))
            if best is None or rm < best[0]:
                best = (rm, a, m)
        _, alpha, model = best

        pred = {
            "train": np.log1p(finv(model.predict(Xtr))),
            "valid": np.log1p(finv(model.predict(Xva))),
            "test": np.log1p(finv(model.predict(Xte))),
        }
        smear = float(np.mean(np.exp(ys["train"].values - pred["train"])))
        res = {k: _ablation_metrics(ys[k], pred[k], smear) for k in splits}
        return {
            "alpha": alpha,
            "metrics": res,
            "pred": pred,
            "smear": smear,
            "model": model,
            "pre": pre,
            "n_features": int(Xtr.shape[1]),
        }

    variants = [
        (
            "0. Baseline (v1)",
            dict(include_metro_distance=False, time_mode=None, area_spline=False),
            0.0,
        ),
        (
            "c. Control: tendencia lineal",
            dict(include_metro_distance=False, time_mode="linear", area_spline=False),
            0.0,
        ),
        (
            "1. + tiempo (spline, const.)",
            dict(include_metro_distance=False, time_mode="spline", area_spline=False),
            0.0,
        ),
        (
            "2. + spline total_area",
            dict(include_metro_distance=False, time_mode="spline", area_spline=True),
            0.0,
        ),
        (
            "3. + metro log/spline (FINAL)",
            dict(include_metro_distance=True, time_mode="spline", area_spline=True),
            0.0,
        ),
        (
            "c. Control: FINAL + Box-Cox λ=-0.2",
            dict(include_metro_distance=True, time_mode="spline", area_spline=True),
            -0.2,
        ),
    ]
    results, fitted = {}, {}
    for name, kw, lam in variants:
        fv = fit_variant(p3.Config(**kw), lam)
        fitted[name] = fv
        results[name] = {
            "alpha": fv["alpha"],
            "n_features": fv["n_features"],
            "metrics": fv["metrics"],
        }

    lam_mle = float(scipy_stats.boxcox(np.expm1(ys["train"].values))[1])
    final_name = "3. + metro log/spline (FINAL)"
    fin, base = fitted[final_name], fitted["0. Baseline (v1)"]

    tr = splits["train"]
    edges_c = np.unique(np.quantile(tr["to_center_km"], np.linspace(0, 1, 6)))
    edges_c[0], edges_c[-1] = -np.inf, np.inf
    yhat_tr = fin["pred"]["train"]
    edges_f = np.unique(np.quantile(yhat_tr, np.linspace(0, 1, 6)))
    edges_f[0], edges_f[-1] = -np.inf, np.inf
    seg = {}
    for part in ("valid", "test"):
        d = splits[part].copy()
        d["yhat"] = fin["pred"][part]
        d["yhat_base"] = base["pred"][part]
        seg[part] = {
            "por_distancia_centro": {
                "final": _segment_table(
                    d, ys[part], fin["pred"][part], edges_c, "to_center_km"
                ),
                "baseline": _segment_table(
                    d, ys[part], base["pred"][part], edges_c, "to_center_km"
                ),
            },
            "por_cuantil_yhat": {
                "final": _segment_table(
                    d, ys[part], fin["pred"][part], edges_f, "yhat"
                ),
                "baseline": _segment_table(
                    d.assign(yhat=d["yhat_base"]),
                    ys[part],
                    base["pred"][part],
                    edges_f,
                    "yhat",
                ),
            },
        }
    lux = {
        part: {
            "final": _luxury_sensitivity(ys[part], fin["pred"][part]),
            "baseline": _luxury_sensitivity(ys[part], base["pred"][part]),
        }
        for part in ("valid", "test")
    }

    allp = pd.concat([splits[k].assign(split=k) for k in splits])
    yall = pd.concat([ys[k] for k in splits])
    q = allp["date_posted"].dt.to_period("Q").astype(str).values
    bias_q = {}
    for nm, fv in (("baseline", base), ("final", fin)):
        r = yall.values - np.concatenate([fv["pred"][k] for k in splits])
        bias_q[nm] = pd.Series(r).groupby(q).mean().round(4).to_dict()

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    qs = sorted(bias_q["final"])
    ax[0].plot(
        qs, [bias_q["baseline"][k] for k in qs], "o-", color="#888", label="Baseline v1"
    )
    ax[0].plot(
        qs, [bias_q["final"][k] for k in qs], "o-", color="#c2410c", label="Final"
    )
    ax[0].axhline(0, color="k", lw=0.8)
    if "2025Q2" in qs:
        ax[0].axvline(qs.index("2025Q2") + 0.5, color="#2563eb", ls=":", label="fin train")
    ax[0].set_title("Sesgo medio del residuo por trimestre (log1p)")
    ax[0].set_xticks(range(0, len(qs), 3))
    ax[0].set_xticklabels(qs[::3], rotation=45)
    ax[0].legend()
    ax[0].set_ylabel("Real − predicho")

    for nm, fv, c in (("Baseline v1", base, "#888"), ("Final", fin, "#c2410c")):
        p_arr, y_arr = fv["pred"]["test"], ys["test"].values
        z = np.sqrt(np.abs((y_arr - p_arr) / np.std(y_arr - p_arr)))
        ax[1].scatter(p_arr, z, s=6, alpha=0.25, color=c)
        order = np.argsort(p_arr)
        sm = pd.Series(z[order]).rolling(150, center=True, min_periods=30).mean()
        ax[1].plot(
            p_arr[order],
            sm,
            color=c,
            lw=2.5,
            label=f"{nm}  (ρ={fv['metrics']['test']['rho_abs_resid_vs_fit']:+.2f})",
        )
    ax[1].set_title("Scale-Location — test")
    ax[1].set_xlabel("ŷ (log1p RUB)")
    ax[1].set_ylabel("√|residuo estandarizado|")
    ax[1].legend()

    names = [n for n, *_ in variants[:5]]
    ax[2].bar(
        range(len(names)),
        [results[n]["metrics"]["test"]["rmse_log"] for n in names],
        color="#c2410c",
    )
    for i, n in enumerate(names):
        v = results[n]["metrics"]["test"]["rmse_log"]
        ax[2].text(i, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
    ax[2].set_xticks(range(len(names)))
    ax[2].set_xticklabels(
        ["Baseline", "Control:\nlineal", "+ tiempo", "+ spline\nárea", "+ metro\n(FINAL)"],
        fontsize=9,
    )
    ax[2].set_title("RMSE_log en test (ablación acumulativa)")
    ax[2].set_ylim(0, 0.18)
    plt.tight_layout()
    fig_ablation_path = figures_dir / "07_before_after.png"
    plt.savefig(fig_ablation_path, dpi=130)
    plt.close(fig)

    payload = {
        "variantes": results,
        "lambda_boxcox_mle_train": lam_mle,
        "segmentos": seg,
        "sensibilidad_sin_top5pct_yhat": lux,
        "sesgo_por_trimestre": bias_q,
        "alpha_final": fin["alpha"],
        "smearing_duan": fin["smear"],
    }
    ablation_json_path = reports_dir / "ablation_report.json"
    atomic_write_json(ablation_json_path, payload)

    L = [
        "# Reporte de ablación y diagnóstico (Etapa 03/04)\n",
        "Ridge sobre `log1p(monthly_rent_rub)`; alpha elegido solo con *valid*. "
        "Test no interviene en ninguna decisión.\n",
        "## 1. Ablación acumulativa\n",
        "| Variante | α | RMSE valid | RMSE test | Sesgo valid | Sesgo test | MAPE test % | ρ(\\|r\\|,ŷ) test |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n, r in results.items():
        v, t = r["metrics"]["valid"], r["metrics"]["test"]
        L.append(
            f"| {n} | {r['alpha']} | {v['rmse_log']:.4f} | {t['rmse_log']:.4f} | {v['bias_log']:+.3f} | "
            f"{t['bias_log']:+.3f} | {t['mape_pct']:.2f} | {t['rho_abs_resid_vs_fit']:+.2f} |"
        )
    L += [
        f"\nλ de Box-Cox por máxima verosimilitud (solo train): **{lam_mle:.3f}** (≈ log). "
        "Un λ más agresivo (−0.2) reduce ρ (ver fila de control) a cambio de algo más de RMSE: "
        "es un intercambio de precisión por homocedasticidad, no una mejora neta.\n",
        "## 2. Sesgo por segmento — test (modelo final vs baseline)\n",
    ]
    for key, title in (
        ("por_distancia_centro", "Distancia al centro (quintiles de train)"),
        ("por_cuantil_yhat", "Cuantil de ŷ (quintiles de train)"),
    ):
        L += [
            f"**{title}**\n",
            "| Segmento | n | RMSE baseline | RMSE final | Desv. resid. baseline | Desv. resid. final |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for b, f_seg in zip(seg["test"][key]["baseline"], seg["test"][key]["final"]):
            L.append(
                f"| {f_seg['segmento']} | {f_seg['n']} | {b['rmse_log']:.4f} | {f_seg['rmse_log']:.4f} | "
                f"{b['std_resid']:.4f} | {f_seg['std_resid']:.4f} |"
            )
        L.append("")
    L += [
        "## 3. Sensibilidad «sin propiedades de lujo» (solo diagnóstico)\n",
        "Se excluye el 5 % con mayor ŷ. **No** se propone como cambio del modelo: filtrar por una "
        "función del objetivo introduce sesgo de selección y no resuelve la causa.\n",
        "| Partición | Modelo | RMSE (todas) | RMSE (sin top 5 %) | ρ (todas) | ρ (sin top 5 %) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for part in ("valid", "test"):
        for nm in ("baseline", "final"):
            a_m, b_m = lux[part][nm]["todas"], lux[part][nm]["sin_top"]
            L.append(
                f"| {part} | {nm} | {a_m['rmse_log']:.4f} | {b_m['rmse_log']:.4f} | "
                f"{a_m['rho_abs_resid_vs_fit']:+.2f} | {b_m['rho_abs_resid_vs_fit']:+.2f} |"
            )
    L += ["", "![antes/después](figures/07_before_after.png)"]
    ablation_md_path = reports_dir / "ablation_report.md"
    atomic_write_text(ablation_md_path, "\n".join(L))
    LOGGER.info(
        "Ablación (Paso 6 Parte A) guardada en %s y %s.",
        ablation_md_path,
        fig_ablation_path,
    )
    return [ablation_json_path, ablation_md_path, fig_ablation_path]


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
        mlflow.log_metrics(
            {
                str(k): float(v)
                for k, v in metrics.items()
                if np.isfinite(float(v))
            }
        )

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
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "mlflow.sklearn.log_model falló (%s); "
                    "el .joblib del modelo ya está como artifact.",
                    exc,
                )
        run_id = run.info.run_id
    return run_id


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paso 5+6 unificado: entrenamiento final, diagnóstico, fórmula ganadora, VIF, ablación y MLflow.",
    )
    parser.add_argument(
        "--tracking-uri",
        default="file:./mlruns",
        help="URI de tracking MLflow (default: file:./mlruns).",
    )
    parser.add_argument(
        "--experiment",
        default="primer parcial",
        help="Nombre del experimento MLflow (default: 'primer parcial').",
    )
    parser.add_argument(
        "--metric",
        default="rmse_log",
        choices=["rmse_log", "mae_log", "rmse_rub", "mae_rub"],
        help="Métrica de selección declarada (se registra como param).",
    )
    parser.add_argument(
        "--fit-on",
        default="train",
        choices=["train", "train_valid"],
        help="Conjunto de ajuste. Default: train.",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Además del ganador, reentrena toda la allowlist para comparación en valid.",
    )
    parser.add_argument(
        "--model-family",
        default=None,
        choices=sorted(set(FAMILY_ALIASES.values())),
        help="Forzar familia del 'ganador' (si difiere del bundle 04).",
    )
    parser.add_argument(
        "--data-dir",
        default="data/processed/splits",
        help="Directorio con X_/y_ parquet|csv de la etapa 03.",
    )
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument(
        "--diagnostics-dir",
        default="reports/diagnostics",
        help="Directorio donde se guardan las gráficas, el .txt y el .md del paso 6 (default: reports/diagnostics).",
    )
    parser.add_argument(
        "--lc-reps",
        type=int,
        default=3,
        help="Repeticiones por tamaño en la curva de aprendizaje fuera de tiempo.",
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Omitir la ablación (Parte A del paso 6).",
    )
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Omitir el diagnóstico extra del paso 6 (Parte B).",
    )
    parser.add_argument(
        "--y-scale",
        default="auto",
        choices=["auto", "raw", "log1p"],
        help="Escala de los y guardados por la etapa 03.",
    )
    parser.add_argument(
        "--strict-hash",
        action="store_true",
        help="Verifica SHA-256 de data/rentals_clean.csv (solo lectura).",
    )
    parser.add_argument(
        "--no-sklearn-model",
        action="store_true",
        help="No registrar el modelo con mlflow.sklearn (solo joblib).",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    if args.seed != RANDOM_STATE:
        LOGGER.warning(
            "Se usa seed=%d (el proyecto declara RANDOM_STATE=%d).",
            args.seed,
            RANDOM_STATE,
        )
    seed = args.seed

    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir)
    reports_dir = Path(args.reports_dir)
    figures_dir = reports_dir / "figures"
    diagnostics_dir = Path(args.diagnostics_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

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
    try:
        _rep = json.loads(
            (reports_dir / "preprocessing_report.json").read_text(encoding="utf-8")
        )
        _sc = _rep.get("split_cutoffs", {})
        split_params.update({k: v for k, v in _sc.items() if k in SPLIT_CUTOFFS})
    except Exception as exc:  # noqa: BLE001
        print(
            f"[AVISO] No se pudieron leer los cortes de preprocessing_report.json: {exc}"
        )

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

    train_only_preprocessor_path = preprocessor_path
    preprocessor_fit_set = "train"
    if args.fit_on == "train_valid":
        tv_path = artifacts_dir / "preprocessor_03_trainvalid.joblib"
        if tv_path.exists():
            preprocessor_path = tv_path
            preprocessor_fit_set = "train_valid"
            LOGGER.info("Preprocesador ajustado con train+valid: %s", tv_path)
        else:
            LOGGER.warning(
                "--fit-on train_valid pero falta %s: se usa el preprocesador ajustado solo con "
                "train. Re-ejecuta 03_preprocessing.py (v5).",
                tv_path,
            )
    preprocessor = joblib.load(preprocessor_path)
    bundle = load_selection_bundle(bundle_path)
    winner_est_04 = bundle["estimator"]
    n_features = bundle["n_features"]

    # --- Transformación + guard-rail anti-fuga ------------------------------- #
    Xt: Dict[str, np.ndarray] = {}
    fit_mode = ""
    for split in ("train", "valid", "test"):
        Xt[split], _input_names, fit_mode = transform_features(
            X_splits[f"X_{split}"], preprocessor, n_features
        )
    LOGGER.info(
        "Alineación de features: %s | X_train=%s", fit_mode, Xt["train"].shape
    )

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
    winner = (
        clone(winner_est_04)
        if params_origin == "bundle_04"
        else build_estimator(family, params_winner)
    )
    if args.seed == RANDOM_STATE and hasattr(winner, "get_params"):
        if "random_state" in winner.get_params():
            winner.set_params(random_state=seed)
    winner.fit(X_fit, y_log_fit)

    smearing_fit = compute_smearing_factor(y_log_fit, winner.predict(X_fit))
    LOGGER.info("Smearing de Duan (fit): S = %.6f", smearing_fit)
    if bundle["smearing_04"] is not None:
        LOGGER.info(
            "Smearing etapa 04: S = %.6f (se usa el recalculado en fit).",
            bundle["smearing_04"],
        )

    # --- Métricas: fit, valid y TEST (una única evaluación de test) ---------- #
    metrics: Dict[str, float] = {}
    preds_store: Dict[str, Dict[str, np.ndarray]] = {}

    def evaluate(split: str) -> None:
        yhat_log, yhat_rub_s, yhat_rub_n = predict_all(winner, Xt[split], smearing_fit)
        preds_store[split] = {
            "yhat_log": yhat_log,
            "yhat_rub_smeared": yhat_rub_s,
            "yhat_rub_naive": yhat_rub_n,
        }
        block = regression_metrics(
            y_log[split], yhat_log, y_rub[split], yhat_rub_s, yhat_rub_n
        )
        metrics.update({f"{split}_{k}": v for k, v in block.items()})
        metrics.update(
            residual_diagnostics(
                y_log[split], yhat_log, y_rub[split], yhat_rub_s, split
            )
        )

    evaluate("fit")
    evaluate("valid")
    evaluate("test")
    LOGGER.info(
        "test | RMSE_log=%.4f  RMSE_RUB_smeared=%.0f  MAPE=%.2f%%  R2_log=%.4f",
        metrics["test_rmse_log"],
        metrics["test_rmse_rub_smeared"],
        metrics["test_mape_rub"],
        metrics["test_r2_log"],
    )

    # --- Figuras y diagnóstico Paso 5 ---------------------------------------- #
    figures: Dict[str, plt.Figure] = {}
    for split in ("valid", "test"):
        figures.update(
            make_figures(
                y_log[split],
                preds_store[split]["yhat_log"],
                y_rub[split],
                preds_store[split]["yhat_rub_smeared"],
                split,
            )
        )
    for fig_name, fig in figures.items():
        save_figure_atomic(fig, figures_dir / f"{fig_name}.png")

    prev_diag = load_prev_valid_diagnostics(
        reports_dir / "predictions_valid.csv", family
    )
    diag_md = build_diagnostic_md(
        metrics, family, smearing_fit, args.fit_on, prev_diag
    )
    atomic_write_text(reports_dir / "diagnostico_residuos.md", diag_md)
    atomic_write_json(reports_dir / "residual_diagnostics.json", metrics)

    # --- Predicciones y bundle reentrenado ----------------------------------- #
    for split in ("valid", "test"):
        frame = pd.DataFrame(
            {
                "y_true_log": y_log[split],
                "yhat_log": preds_store[split]["yhat_log"],
                "y_true_rub": y_rub[split],
                "yhat_rub_smeared": preds_store[split]["yhat_rub_smeared"],
                "yhat_rub_naive": preds_store[split]["yhat_rub_naive"],
                "resid_log": y_log[split] - preds_store[split]["yhat_log"],
                "resid_rub_smeared": y_rub[split]
                - preds_store[split]["yhat_rub_smeared"],
            },
            index=y_splits[f"y_{split}"].index,
        )
        frame.to_csv(
            reports_dir / f"predictions_{split}_05.csv", index_label="row_index"
        )

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

    # --- Ejecución integrada del Paso 6 (Parte B: Diagnóstico, TXT, MD y PNGs) #
    diag_06_files: List[Path] = []
    if not args.skip_diagnostics:
        try:
            diag_06_files, extra_metrics_06 = run_diagnostics_06(
                winner=winner,
                family=family,
                preprocessor=preprocessor,
                smearing_fit=smearing_fit,
                X_splits=X_splits,
                Xt=Xt,
                y_splits=y_splits,
                y_log=y_log,
                out_names=out_names,
                fit_on=args.fit_on,
                diag_dir=diagnostics_dir,
                lc_reps=args.lc_reps,
            )
            metrics.update(extra_metrics_06)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Falló el bloque de diagnóstico del paso 6: %s", exc)

    # --- Ejecución integrada del Paso 6 (Parte A: Ablación) ------------------ #
    ablation_06_files: List[Path] = []
    if not args.skip_ablation:
        try:
            ablation_06_files = run_ablation_06(reports_dir, figures_dir)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Falló el bloque de ablación del paso 6: %s", exc)

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
        "etapa": "05_06_unificado",
        "familia": "lineal",
        "target": TARGET_LOG,
        "proyecto": "alquileres-moscu",
        "fit_set": args.fit_on,
        "preprocessor_fit_set": preprocessor_fit_set,
        "metrica_seleccion": args.metric,
        "seed": str(seed),
    }

    winner_params = flatten_params(
        winner,
        {
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
            "preprocessor_fit_set": preprocessor_fit_set,
            "metrica_seleccion": args.metric,
            "y_scale_detectada": y_scale,
            "feature_mode": fit_mode,
            "n_train": int(Xt["train"].shape[0]),
            "n_valid": int(Xt["valid"].shape[0]),
            "n_test": int(Xt["test"].shape[0]),
            "sha256_rentals_clean": sha_value,
            "tracking_uri": args.tracking_uri,
            **split_params,
        },
    )

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

    if train_only_preprocessor_path != preprocessor_path:
        winner_artifacts.append((train_only_preprocessor_path, "model_artifacts"))

    # Registrar todos los archivos del Paso 6 (gráficas, .txt, .md, .csv) en MLflow:
    # 1. En la carpeta "diagnostics" (espejo exacto de reports/diagnostics/)
    for fpath in diag_06_files:
        winner_artifacts.append((fpath, "diagnostics"))
        # También exponer las gráficas en "figures" y los reportes .txt/.md/.csv en "reports"
        if fpath.suffix.lower() == ".png":
            winner_artifacts.append((fpath, "figures"))
        elif fpath.suffix.lower() in (".txt", ".md", ".csv"):
            winner_artifacts.append((fpath, "reports"))

    # 2. Archivos de ablación (ablation_report.md, ablation_report.json, 07_before_after.png)
    for fpath in ablation_06_files:
        if fpath.suffix.lower() == ".png":
            winner_artifacts.append((fpath, "figures"))
        else:
            winner_artifacts.append((fpath, "reports"))

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
    LOGGER.info("Run ganador registrado en MLflow: %s", run_id)

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
                f"valid_{k}": v
                for k, v in regression_metrics(
                    y_log["valid"],
                    yhat_log_v,
                    y_rub["valid"],
                    yhat_rub_vs,
                    yhat_rub_vn,
                ).items()
            }
            metrics_other.update(
                residual_diagnostics(
                    y_log["valid"], yhat_log_v, y_rub["valid"], yhat_rub_vs, "valid"
                )
            )
            figs_other = make_figures(
                y_log["valid"], yhat_log_v, y_rub["valid"], yhat_rub_vs, "valid"
            )
            other_path = artifacts_dir / f"model_05_{other.lower()}_comparison.joblib"
            atomic_joblib_dump(
                {
                    "modelo": est_other,
                    "clase": other,
                    "smearing_factor": S_other,
                    "fit_set": args.fit_on,
                    "uso": "comparacion_valid",
                },
                other_path,
            )
            other_run = log_mlflow_run(
                run_name=f"paso05-comparacion-{other.lower()}",
                tags={
                    **common_tags,
                    "test_evaluado": "false",
                    "modelo": other,
                    "rol": "comparacion_valid",
                },
                params=flatten_params(
                    est_other,
                    {
                        "familia": "lineal",
                        "modelo": other,
                        "modelo_origen_params": origin_other,
                        "target": TARGET_LOG,
                        "smearing_factor_S": S_other,
                        "n_features": int(n_features),
                        "random_state": seed,
                        "fit_set": args.fit_on,
                        **split_params,
                    },
                ),
                metrics=metrics_other,
                figures=figs_other,
                artifacts=[(other_path, "model_artifacts")],
                estimator=None,
                log_sklearn_model=False,
            )
            for fig in figs_other.values():
                plt.close(fig)
            LOGGER.info(
                "Run comparación %-15s → %s (valid only).", other, other_run
            )

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
        "diagnostics_dir": str(diagnostics_dir),
        "archivos_diagnostics_06": [str(p) for p in diag_06_files],
        "archivos_ablation_06": [str(p) for p in ablation_06_files],
    }
    atomic_write_json(reports_dir / "training_mlflow_report.json", summary)

    print("\n=== Paso 5 + 6 (Unificado) completado ===")
    print(f"Experimento     : {args.experiment}")
    print(f"Run ganador     : {run_id}")
    print(f"Modelo          : {family}  (fit={args.fit_on}, S={smearing_fit:.6f})")
    print(
        f"valid           : RMSE_log={metrics['valid_rmse_log']:.4f}  "
        f"RMSE_RUB={metrics['valid_rmse_rub_smeared']:.0f}  "
        f"MAPE={metrics['valid_mape_rub']:.2f}%"
    )
    print(
        f"test            : RMSE_log={metrics['test_rmse_log']:.4f}  "
        f"RMSE_RUB={metrics['test_rmse_rub_smeared']:.0f}  "
        f"MAPE={metrics['test_mape_rub']:.2f}%"
    )
    print(f"Figuras Paso 5  : {figures_dir}/")
    print(f"Diagnóstico 06  : {diagnostics_dir}/ (fórmula, .md, .csv, VIF y 10 gráficas)")
    print(f"Reporte Residuos: {reports_dir / 'diagnostico_residuos.md'}")
    if ablation_06_files:
        print(f"Reporte Ablación: {reports_dir / 'ablation_report.md'}")
    return 0


def _safe_load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        LOGGER.warning("No se pudo leer %s; se continúa sin él.", path)
        return None


if __name__ == "__main__":
    sys.exit(main())