#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_graficos_informe.py
Extrae la fórmula ganadora (coeficientes + intercepto) y genera:
1. Histograma de residuos + curva Normal teórica (normalidad de errores)
2. Scale-Location Plot (homocedasticidad, igual al de la diapositiva de clase)
3. Learning Curve + Comparativa Train/Valid/Test (diagnóstico de Overfitting/Underfitting)
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import learning_curve
from sklearn.base import clone
from importlib import import_module

# Importamos funciones de carga del script 05
paso05 = import_module("05_training_mlflow")

REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"
ARTIFACTS_DIR = Path("artifacts")
DATA_DIR = Path("data/processed/splits")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def extraer_formula_y_coeficientes():
    refit_bundle = joblib.load(ARTIFACTS_DIR / "model_05_refit.joblib")
    preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor_03.joblib")
    X_splits = paso05.load_splits(DATA_DIR)

    modelo = paso05.final_estimator(refit_bundle["modelo"])
    clase = refit_bundle["clase"]
    smearing = refit_bundle["smearing_factor"]

    coefs = np.asarray(modelo.coef_, dtype=float).ravel()
    intercept = float(np.asarray(modelo.intercept_, dtype=float).ravel()[0])

    # Obtener nombres de las variables
    feat_names = paso05.transformed_feature_names(preprocessor, len(coefs))
    if feat_names and feat_names[0] == "feat_0" and X_splits["X_train"].shape[1] == len(coefs):
        feat_names = [str(c) for c in X_splits["X_train"].columns]

    df_coef = pd.DataFrame({
        "variable": feat_names,
        "coeficiente": coefs,
        "abs_coef": np.abs(coefs),
        "efecto_pct_aprox_en_rub": (np.exp(coefs) - 1.0) * 100.0,
    }).sort_values("abs_coef", ascending=False)

    df_coef.to_csv(REPORTS_DIR / "coeficientes_modelo.csv", index=False)

    terminos = [f"{intercept:.6f}"]
    for _, row in df_coef.iterrows():
        if abs(row["coeficiente"]) > 1e-8:
            signo = "+" if row["coeficiente"] >= 0 else "-"
            terminos.append(f"{signo} {abs(row['coeficiente']):.6f} * ({row['variable']})")
    ecuacion_str = " ".join(terminos)

    md_lines = [
        f"# Fórmula Ganadora y Coeficientes — Modelo `{clase}`\n",
        f"- **Familia del modelo:** `{clase}`",
        f"- **Intercepto (b0):** `{intercept:.6f}`",
        f"- **Factor de Duan Smearing (S):** `{smearing:.6f}`",
        f"- **Variables totales tras preprocesamiento:** `{len(coefs)}` "
        f"(activas != 0: `{int(np.sum(np.abs(coefs) > 1e-8))}`)\n",
        "## 1. Ecuación en escala logarítmica",
        "```text",
        f"y_log_hat = {ecuacion_str}",
        "```\n",
        "## 2. Ecuación retransformada a Rublos (con Duan Smearing)",
        "```text",
        f"y_rub_hat = exp(y_log_hat) * {smearing:.6f} - 1",
        "```\n",
        "## 3. Top 25 Variables con mayor peso (Variables estandarizadas)",
        "| Ranking | Variable | Coeficiente | Impacto aprox. en RUB por +1 unidad (%) |",
        "|---:|---|---:|---:|",
    ]
    for idx, (_, row) in enumerate(df_coef.head(25).iterrows(), start=1):
        md_lines.append(
            f"| {idx} | `{row['variable']}` | {row['coeficiente']:+.6f} | {row['efecto_pct_aprox_en_rub']:+.2f}% |"
        )

    out_md = REPORTS_DIR / "formula_y_coeficientes.md"
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[OK] Fórmula y coeficientes guardados en: {out_md} y reports/coeficientes_modelo.csv")


def graficar_histograma_y_scale_location(split: str):
    csv_path = REPORTS_DIR / f"predictions_{split}_05.csv"
    df = pd.read_csv(csv_path)
    resid = df["resid_log"].to_numpy()
    yhat = df["yhat_log"].to_numpy()

    # --- 1. HISTOGRAMA DE ERRORES + CURVA NORMAL ---
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    mu, sigma = float(np.mean(resid)), float(np.std(resid, ddof=1))
    skew, kurt = float(stats.skew(resid)), float(stats.kurtosis(resid))

    ax.hist(resid, bins=40, density=True, color="#c1440e", alpha=0.45,
            edgecolor="white", label="Histograma de residuos")

    x_grid = np.linspace(resid.min(), resid.max(), 300)
    ax.plot(x_grid, stats.norm.pdf(x_grid, mu, sigma), color="#2b2b2b", lw=2.0,
            ls="--", label=f"Normal teórica (μ={mu:.3f}, σ={sigma:.3f})")
    ax.axvline(0.0, color="#1f77b4", lw=1.2, ls=":", label="Cero (sin sesgo)")
    ax.axvline(mu, color="#c1440e", lw=1.5, ls="-", label=f"Media real ({mu:+.3f})")

    ax.set_title(f"Distribución de Residuos (escala log) — {split}\n"
                 f"Skewness = {skew:.3f} | Exceso de Kurtosis = {kurt:.3f}")
    ax.set_xlabel("Residuo = Real - Predicho (log1p(RUB))")
    ax.set_ylabel("Densidad")
    ax.legend(frameon=False)
    fig.tight_layout()
    hist_path = FIGURES_DIR / f"histograma_errores_{split}.png"
    fig.savefig(hist_path, dpi=140)
    plt.close(fig)

    # --- 2. SCALE-LOCATION PLOT (Idéntico a la Foto 4 de clase) ---
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    std_resid = (resid - mu) / sigma
    sqrt_abs_std_resid = np.sqrt(np.abs(std_resid))

    ax.scatter(yhat, sqrt_abs_std_resid, s=12, alpha=0.30, color="#2b2b2b", edgecolors="none")

    # Línea roja de tendencia suavizada por cuantiles
    tmp = pd.DataFrame({"yhat": yhat, "y_sl": sqrt_abs_std_resid})
    tmp["bin"] = pd.qcut(tmp["yhat"], q=12, duplicates="drop")
    trend = tmp.groupby("bin", observed=True).mean(numeric_only=True)
    ax.plot(trend["yhat"], trend["y_sl"], color="#d62728", lw=2.2,
            label="Tendencia media (homocedasticidad si es horizontal)")

    # Etiquetar los 3 outliers más altos
    top3_idx = np.argsort(sqrt_abs_std_resid)[-3:]
    col_id = "row_index" if "row_index" in df.columns else df.columns[0]
    for i in top3_idx:
        ax.annotate(str(df[col_id].iloc[i]),
                    (yhat[i], sqrt_abs_std_resid[i]),
                    fontsize=8, xytext=(4, 2), textcoords="offset points")

    ax.set_title(f"Scale-Location Plot — {split}")
    ax.set_xlabel("Valores Ajustados / Predichos (log1p(RUB))")
    ax.set_ylabel("√|Residuos Estandarizados|")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    sl_path = FIGURES_DIR / f"scale_location_{split}.png"
    fig.savefig(sl_path, dpi=140)
    plt.close(fig)
    print(f"[OK] Gráficos de {split} guardados: {hist_path.name} y {sl_path.name}")


def graficar_diagnostico_overfitting():
    refit_bundle = joblib.load(ARTIFACTS_DIR / "model_05_refit.joblib")
    preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor_03.joblib")
    diag_metrics = json.loads((REPORTS_DIR / "residual_diagnostics.json").read_text(encoding="utf-8"))

    X_splits = paso05.load_splits(DATA_DIR)
    y_splits = paso05.load_targets(DATA_DIR)
    y_scale = paso05.resolve_y_scale(y_splits["y_train"], "auto")

    Xt_train, _, _ = paso05.transform_features(
        X_splits["X_train"], preprocessor, refit_bundle["n_features"]
    )
    y_log_train, _ = paso05.to_log_and_rub(y_splits["y_train"], y_scale)

    estimator = clone(paso05.final_estimator(refit_bundle["modelo"]))
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, Xt_train, y_log_train,
        train_sizes=np.linspace(0.15, 1.0, 8),
        cv=5, scoring="neg_root_mean_squared_error", n_jobs=1
    )
    train_rmse = -np.mean(train_scores, axis=1)
    val_rmse = -np.mean(val_scores, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # Panel 1: Curva de Aprendizaje (Overfitting vs Underfitting)
    ax = axes[0]
    ax.plot(train_sizes, train_rmse, "o-", color="#1f77b4", lw=2, label="RMSE Entrenamiento (Train)")
    ax.plot(train_sizes, val_rmse, "s-", color="#c1440e", lw=2, label="RMSE Validación Cruzada (CV)")
    ax.fill_between(train_sizes, train_rmse, val_rmse, color="#c1440e", alpha=0.10)
    ax.set_title("Curva de Aprendizaje (Diagnóstico de Overfitting)")
    ax.set_xlabel("Cantidad de filas de entrenamiento")
    ax.set_ylabel("Error RMSE (escala log)")
    ax.legend(frameon=False)

    # Panel 2: Comparación directa Fit vs Valid vs Test
    ax2 = axes[1]
    splits_names = ["Fit (Entrenamiento)", "Valid", "Test"]
    rmses = [
        diag_metrics.get("fit_rmse_log", 0.0),
        diag_metrics.get("valid_rmse_log", 0.0),
        diag_metrics.get("test_rmse_log", 0.0),
    ]
    bars = ax2.bar(splits_names, rmses, color=["#6b6560", "#c1440e", "#2b2b2b"], width=0.5)
    for bar, val in zip(bars, rmses):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.002, f"{val:.4f}",
                 ha="center", va="bottom", fontweight="bold")
    ax2.set_ylim(0, max(rmses) * 1.18 if max(rmses) > 0 else 1.0)
    ax2.set_title("Comparación de Error RMSE_log entre Particiones")
    ax2.set_ylabel("RMSE (log1p(RUB))")

    fig.tight_layout()
    ov_path = FIGURES_DIR / "overfitting_learning_curve.png"
    fig.savefig(ov_path, dpi=140)
    plt.close(fig)
    print(f"[OK] Gráfico de Overfitting/Underfitting guardado en: {ov_path}")


if __name__ == "__main__":
    extraer_formula_y_coeficientes()
    for s in ("valid", "test"):
        graficar_histograma_y_scale_location(s)
    graficar_diagnostico_overfitting()