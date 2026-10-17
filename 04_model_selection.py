"""
04_model_selection.py — Etapa 04: Selección de modelos (SOLO regresión lineal)

Entrada : data/processed/splits/X_*.parquet|csv, y_*.parquet|csv   (etapa 03)
          artifacts/preprocessor_03.joblib                        (fit solo en train)
          reports/preprocessing_report.json                       (auditoría etapa 03)
Salida  : reports/model_selection_report.json
          reports/model_selection_ranking.csv
          reports/predictions_valid.csv
          artifacts/best_model_04.joblib
El split TEST no se toca en esta etapa (reservado para etapa 05 con MLflow).

Requisitos: python>=3.10, pandas>=2.0, numpy>=1.24, scikit-learn>=1.3, joblib, pyarrow (opcional)
Uso: python 04_model_selection.py [--metric rmse_log|mae_log|rmse_rub|mae_rub]
"""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    Ridge,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

RANDOM_STATE = 42
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("data/processed/splits")
    source_csv: Path = Path("data/rentals_clean.csv")   # para restaurar orden temporal
    preprocessor: Path = Path("artifacts/preprocessor_03.joblib")
    reports_dir: Path = Path("reports")
    artifacts_dir: Path = Path("artifacts")
    target: str = "monthly_rent_rub"
    cv_splits: int = 5
    metric: str = "rmse_log"     # criterio de selección final sobre VALID
    n_jobs: int = -1


# --------------------------------------------------------------------------- #
# Allowlist: SOLO familia de regresión lineal                                  #
# --------------------------------------------------------------------------- #
def build_model_space() -> dict:
    return {
        "LinearRegression": (
            LinearRegression(),
            {},
        ),
        "Ridge": (
            Ridge(random_state=RANDOM_STATE),
            {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]},
        ),
        "Lasso": (
            Lasso(max_iter=50_000, tol=1e-4, random_state=RANDOM_STATE),
            {"alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1.0]},
        ),
        "ElasticNet": (
            ElasticNet(max_iter=50_000, tol=1e-4, random_state=RANDOM_STATE),
            {
                "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
                "l1_ratio": [0.15, 0.5, 0.8, 0.95],
            },
        ),
        "HuberRegressor": (
            HuberRegressor(max_iter=50_000),
            {"alpha": [1e-4, 1e-3, 1e-2, 1e-1], "epsilon": [1.1, 1.35, 1.5]},
        ),
    }
    # NOTA: se modela log1p(y) directamente con estimadores sklearn.
    # Es matemáticamente equivalente a TransformedTargetRegressor(log1p, expm1)
    # y permite validar en la escala de modelado (log) sin retransformar en cada fold.


LINEAR_ALLOWLIST = {"LinearRegression", "Ridge", "Lasso", "ElasticNet", "HuberRegressor"}


# --------------------------------------------------------------------------- #
# Utilidades                                                                   #
# --------------------------------------------------------------------------- #
def _read_table(path_no_ext: Path) -> pd.DataFrame:
    pq, csv = path_no_ext.with_suffix(".parquet"), path_no_ext.with_suffix(".csv")
    path = pq if pq.exists() else csv
    if not path.exists():
        raise FileNotFoundError(f"No existe {pq} ni {csv}")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, index_col=0)


def atomic_write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, suffix=".tmp",
                                     delete=False, encoding="utf-8") as tmp:
        json.dump(obj, tmp, indent=2, ensure_ascii=False, default=str)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    resid = y_true - y_pred
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "mape_pct": float(np.mean(np.abs(resid / np.clip(np.abs(y_true), 1.0, None))) * 100),
        "r2": float(1.0 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
    }


def duan_smearing_factor(y_log: np.ndarray, y_log_pred: np.ndarray) -> float:
    """S = mean(exp(eps)); y_hat_rub = exp(y_hat_log) * S - 1  (objetivo log1p)."""
    eps = np.asarray(y_log, float) - np.asarray(y_log_pred, float)
    return float(np.mean(np.exp(eps)))


def chronological_sort(X: pd.DataFrame, y: pd.Series, cfg: Config, name: str):
    if not cfg.source_csv.exists():
        print(f"[AVISO] {cfg.source_csv} no existe; split '{name}' en orden de archivo "
              "(TimeSeriesSplit podría no ser estrictamente temporal).")
        return X, y
    dates = pd.read_csv(cfg.source_csv, usecols=["date_posted"],
                        parse_dates=["date_posted"])["date_posted"]
    if not set(X.index).issubset(set(dates.index)):
        print(f"[AVISO] Índices de '{name}' no alinean con el CSV fuente; sin ordenar.")
        return X, y
    order = dates.loc[X.index].sort_values().index
    return X.loc[order], y.loc[order]


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 04 — Selección de modelos lineales")
    parser.add_argument("--metric", default="rmse_log",
                        choices=["rmse_log", "mae_log", "rmse_rub", "mae_rub"])
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    cfg = Config(metric=args.metric, cv_splits=args.cv_splits, n_jobs=args.n_jobs)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # --- Artefactos de etapa 03 -------------------------------------------
    preproc = joblib.load(cfg.preprocessor)
    eda_path = cfg.reports_dir / "preprocessing_report.json"
    etapa03 = json.loads(eda_path.read_text(encoding="utf-8")) if eda_path.exists() else {}
    if not etapa03:
        print("[AVISO] Falta preprocessing_report.json; se continúa sin validación cruzada de etapas.")

    # --- Splits ------------------------------------------------------------
    X_raw = {s: _read_table(cfg.data_dir / f"X_{s}") for s in SPLITS}
    y_df = {s: _read_table(cfg.data_dir / f"y_{s}") for s in SPLITS}
    y = {s: y_df[s][cfg.target].astype(float) for s in SPLITS}
    y_log = {s: np.log1p(y[s].to_numpy()) for s in SPLITS}

    X_raw["train"], y["train"] = chronological_sort(X_raw["train"], y["train"], cfg, "train")
    y_log["train"] = np.log1p(y["train"].to_numpy())

    # Transformar con el preprocesador YA ajustado en train (etapa 03)
    Xt = {s: np.asarray(preproc.transform(X_raw[s])) for s in SPLITS}
    for s in SPLITS:
        assert np.isfinite(Xt[s]).all(), f"NaN/Inf en Xt[{s}]"
    n_features = Xt["train"].shape[1]
    print(f"Matrices: train={Xt['train'].shape} valid={Xt['valid'].shape} "
          f"test={Xt['test'].shape} (test NO se evalúa)")

    # --- Búsqueda de hiperparámetros (TimeSeriesSplit sobre train) ----------
    model_space = build_model_space()
    assert set(model_space) <= LINEAR_ALLOWLIST, "Familia no lineal detectada — abortado"

    cv_results, fitted, valid_preds = {}, {}, {}
    tscv = TimeSeriesSplit(n_splits=cfg.cv_splits)

    for name, (estimator, grid) in model_space.items():
        print(f"\n>> {name}: grid={grid or 'sin hiperparámetros'}")
        gs = GridSearchCV(
            estimator, grid or {"fit_intercept": [True]},
            scoring="neg_root_mean_squared_error",
            cv=tscv, n_jobs=cfg.n_jobs, refit=True, return_train_score=False,
        )
        gs.fit(Xt["train"], y_log["train"])
        best_cv_rmse = float(-gs.best_score_)
        print(f"   best_params={gs.best_params_}  CV RMSE(log)={best_cv_rmse:.4f}")

        model = gs.best_estimator_
        S = duan_smearing_factor(y_log["train"], model.predict(Xt["train"]))

        yv_log = model.predict(Xt["valid"])
        yv_naive = np.expm1(yv_log)
        yv_smeared = np.exp(yv_log) * S - 1.0
        yv_true = y["valid"].to_numpy()

        m = {
            "cv_rmse_log": best_cv_rmse,
            "best_params": gs.best_params_,
            "smearing_factor": S,
            "valid_log": metrics(y_log["valid"], yv_log),
            "valid_rub_naive_expm1": metrics(yv_true, yv_naive),
            "valid_rub_smeared": metrics(yv_true, yv_smeared),
        }
        cv_results[name] = m
        fitted[name] = {"model": model, "smearing": S, "params": gs.best_params_}
        valid_preds[name] = {"pred_log": yv_log, "pred_rub_naive": yv_naive,
                             "pred_rub_smeared": yv_smeared}

    # --- Selección final sobre VALID ---------------------------------------
    key_map = {
        "rmse_log": ("valid_log", "rmse"),
        "mae_log": ("valid_log", "mae"),
        "rmse_rub": ("valid_rub_smeared", "rmse"),
        "mae_rub": ("valid_rub_smeared", "mae"),
    }
    block, sub = key_map[cfg.metric]
    ranking = sorted(cv_results, key=lambda n: cv_results[n][block][sub])
    best_name = ranking[0]
    print(f"\n>> Ganador ({cfg.metric}): {best_name}")

    # --- Predicciones de VALID para diagnóstico de residuos -----------------
    pred_df = pd.DataFrame({
        "y_true_rub": y["valid"].to_numpy(),
        "y_true_log": y_log["valid"],
    })
    for name in cv_results:
        pred_df[f"{name}__pred_log"] = valid_preds[name]["pred_log"]
        pred_df[f"{name}__pred_rub_smeared"] = valid_preds[name]["pred_rub_smeared"]
    pred_df.to_csv(cfg.reports_dir / "predictions_valid.csv", index=True)

    ranking_df = pd.DataFrame([
        {
            "model": n,
            "cv_rmse_log": cv_results[n]["cv_rmse_log"],
            "valid_rmse_log": cv_results[n]["valid_log"]["rmse"],
            "valid_mae_log": cv_results[n]["valid_log"]["mae"],
            "valid_rmse_rub_smeared": cv_results[n]["valid_rub_smeared"]["rmse"],
            "valid_mae_rub_smeared": cv_results[n]["valid_rub_smeared"]["mae"],
            "valid_mape_rub_smeared": cv_results[n]["valid_rub_smeared"]["mape_pct"],
            "valid_r2_rub_smeared": cv_results[n]["valid_rub_smeared"]["r2"],
            "smearing_factor": cv_results[n]["smearing_factor"],
            "best_params": json.dumps(cv_results[n]["best_params"]),
        }
        for n in ranking
    ])
    ranking_df.to_csv(cfg.reports_dir / "model_selection_ranking.csv", index=False)

    joblib.dump(
        {
            "model_name": best_name,
            "model": fitted[best_name]["model"],
            "smearing_factor": fitted[best_name]["smearing"],
            "params": fitted[best_name]["params"],
            "preprocessor": str(cfg.preprocessor),
            "n_features": int(n_features),
            "selected_by": cfg.metric,
        },
        cfg.artifacts_dir / "best_model_04.joblib",
    )

    report = {
        "etapa": "04_model_selection",
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "familia_permitida": sorted(LINEAR_ALLOWLIST),
        "target_modelado": "log1p(monthly_rent_rub)",
        "retransformacion": "Duan-smearing: y_hat = exp(y_hat_log)*S - 1",
        "cv": {"tipo": "TimeSeriesSplit", "n_splits": cfg.cv_splits,
               "scoring": "neg_root_mean_squared_error (escala log)"},
        "criterio_seleccion": f"{cfg.metric} sobre split VALID",
        "n_features": int(n_features),
        "resultados": cv_results,
        "ranking": ranking,
        "ganador": best_name,
        "test_intacto": True,
        "etapa03_ok": bool(etapa03.get("input_sha256_verificado", "desconocido")),
    }
    atomic_write_json(report, cfg.reports_dir / "model_selection_report.json")

    print("\nArtefactos escritos:")
    print("  - reports/model_selection_report.json")
    print("  - reports/model_selection_ranking.csv")
    print("  - reports/predictions_valid.csv   (para gráficas de residuos)")
    print("  - artifacts/best_model_04.joblib")
    print("\nEtapa 04 completada. Listo para Etapa 05 (entrenamiento final + MLflow).")


if __name__ == "__main__":
    main()