"""
07_ablation_diagnostics.py — Diagnóstico completo del modelo ganador + ablación
Proyecto: Alquileres en Moscú (regresión lineal, objetivo monthly_rent_rub, escala log1p)

PARTE B (diagnóstico del modelo REAL; usa artifacts/best_model_04.joblib, ajustado solo con train):
  * Métricas por partición (train/valid/test): RMSE, MSE, MAE, MAPE, MSPE (≈MAPE²), sMAPE, R², R² ajustado,
    sesgo, asimetría, curtosis, Spearman |resid|-ŷ, Breusch-Pagan, Jarque-Bera, Durbin-Watson,
    aciertos ±10 %/±20 %, en escala log y en RUB (con smearing de Duan).
  * Semáforo de calidad con reglas prácticas (OK / ATENCIÓN / PROBLEMA).
  * Gráficas valid vs test: histograma de residuos, QQ-plot, Scale-Location, residuo vs ajustado,
    real vs predicho, sesgo por mes, curva de aprendizaje estilo CV (07) y fuera de tiempo (07b) + barras de RMSE, top-20 coeficientes.
  * Fórmula del modelo ganador (intercepto + Σ coef × término) y la de un Ridge compañero interpretable.
PARTE A (ablación, referencia con Ridge): baseline -> +tiempo -> +spline área -> +metro, controles
  negativos, métricas por segmento y sensibilidad "sin lujo" (solo diagnóstico).

Requiere haber corrido 03 y 04 (y opcionalmente 05). Salidas:
  reports/diagnostics/*.png, metricas_comparativas.{csv,md}, formula_ganadora.txt, coeficientes_*.csv
  reports/ablation_report.{json,md}, reports/figures/07_before_after.png

Uso: python 07_ablation_diagnostics.py [--bundle artifacts/best_model_04.joblib] [--skip-ablation] [--lc-reps 3]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import Ridge

warnings.filterwarnings("ignore")

# --- Reutiliza 03_preprocessing.py ------------------------------------------------
_spec = importlib.util.spec_from_file_location("p3", Path(__file__).with_name("03_preprocessing.py"))
p3 = importlib.util.module_from_spec(_spec)
sys.modules["p3"] = p3
_spec.loader.exec_module(p3)

ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
OUT_DIR = Path("reports")
FIG_DIR = OUT_DIR / "figures"


# --------------------------------------------------------------------------- #
# Datos y métricas                                                            #
# --------------------------------------------------------------------------- #
def load_splits(cfg):
    df = pd.read_csv(cfg.input_csv, parse_dates=["date_posted"])
    df = p3.add_engineered_features(df)
    X = p3.temporal_split(df, cfg)
    drop = [cfg.target, "date_posted", "building_year"] + list(cfg.forbidden)
    return X, {k: np.log1p(v[cfg.target].astype(float)) for k, v in X.items()}, drop


def metrics(y_log: pd.Series, pred_log: np.ndarray, smear: float = 1.0) -> dict:
    r = y_log.values - pred_log
    y_rub = np.expm1(y_log.values)
    yhat_rub = np.expm1(pred_log) * smear
    return {
        "n": int(len(r)),
        "rmse_log": float(np.sqrt(np.mean(r ** 2))),
        "bias_log": float(np.mean(r)),
        "mape_pct": float(np.mean(np.abs(y_rub - yhat_rub) / y_rub) * 100),
        "mae_rub": float(np.mean(np.abs(y_rub - yhat_rub))),
        # heterocedasticidad: correlación de Spearman entre |residuo| y ŷ (0 = homocedástico)
        "rho_abs_resid_vs_fit": float(stats.spearmanr(pred_log, np.abs(r))[0]) if len(r) > 2 else float("nan"),
    }


def fit_variant(cfg, splits, ys, drop, lam: float = 0.0):
    """Ajusta preprocesador (solo train) + Ridge (alpha elegido en valid). lam=0 -> log."""
    feats = p3.build_feature_lists(cfg.include_metro_distance, cfg.time_mode, cfg.area_spline)
    pre = p3.build_preprocessor(feats)
    Xs = {k: v.drop(columns=drop) for k, v in splits.items()}
    # fit_transform aplica el cross-fitting del TargetEncoder en train
    Xtr = pre.fit_transform(Xs["train"], ys["train"])
    Xva, Xte = pre.transform(Xs["valid"]), pre.transform(Xs["test"])

    yrub = {k: np.expm1(v.values) for k, v in ys.items()}
    if lam == 0.0:
        f, finv = (lambda y: np.log(y)), (lambda z: np.exp(z))
    else:
        f = lambda y: (y ** lam - 1) / lam
        finv = lambda z: np.maximum(lam * z + 1, 1e-9) ** (1 / lam)
    ztr = f(yrub["train"])

    best = None
    for a in ALPHAS:
        m = Ridge(alpha=a).fit(Xtr, ztr)
        pv = np.log1p(finv(m.predict(Xva)))
        rm = float(np.sqrt(np.mean((ys["valid"].values - pv) ** 2)))
        if best is None or rm < best[0]:
            best = (rm, a, m)
    _, alpha, model = best

    pred = {  # siempre en escala log1p para comparabilidad entre variantes
        "train": np.log1p(finv(model.predict(Xtr))),
        "valid": np.log1p(finv(model.predict(Xva))),
        "test": np.log1p(finv(model.predict(Xte))),
    }
    # Duan smearing con residuos de train (escala log1p) para retransformar a RUB (media)
    smear = float(np.mean(np.exp(ys["train"].values - pred["train"])))
    res = {k: metrics(ys[k], pred[k], smear) for k in splits}
    return {"alpha": alpha, "metrics": res, "pred": pred, "smear": smear, "model": model, "pre": pre,
            "n_features": int(Xtr.shape[1])}


# --------------------------------------------------------------------------- #
# Segmentos y sensibilidad                                                    #
# --------------------------------------------------------------------------- #
def segment_table(df_part, y, pred, edges, col):
    seg = pd.cut(df_part[col], bins=edges, include_lowest=True)
    r = y.values - pred
    out = []
    for lvl, idx in pd.Series(np.arange(len(r))).groupby(seg.values, observed=True):
        rr = r[idx.values]
        out.append({"segmento": str(lvl), "n": int(len(rr)), "rmse_log": float(np.sqrt(np.mean(rr ** 2))),
                    "std_resid": float(np.std(rr)), "bias_log": float(np.mean(rr))})
    return out


def luxury_sensitivity(y, pred, q=0.95):
    """Diagnóstico: métricas excluyendo el top (1-q) por ŷ. NO modifica el modelo."""
    cut = np.quantile(pred, q)
    keep = pred <= cut
    m_all, m_keep = metrics(y, pred), metrics(y[keep], pred[keep])
    return {"umbral_yhat_log": float(cut), "n_excluidas": int((~keep).sum()),
            "todas": m_all, "sin_top": m_keep}


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def run_ablation() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    base_cfg = config_from_report()
    splits, ys, drop = load_splits(base_cfg)

    variants = [
        ("0. Baseline (v1)",                 dict(include_metro_distance=False, time_mode=None, area_spline=False), 0.0),
        ("c. Control: tendencia lineal",     dict(include_metro_distance=False, time_mode="linear", area_spline=False), 0.0),
        ("1. + tiempo (spline, const.)",     dict(include_metro_distance=False, time_mode="spline", area_spline=False), 0.0),
        ("2. + spline total_area",           dict(include_metro_distance=False, time_mode="spline", area_spline=True), 0.0),
        ("3. + metro log/spline (FINAL)",    dict(include_metro_distance=True, time_mode="spline", area_spline=True), 0.0),
        ("c. Control: FINAL + Box-Cox λ=-0.2", dict(include_metro_distance=True, time_mode="spline", area_spline=True), -0.2),
    ]
    results, fitted = {}, {}
    for name, kw, lam in variants:
        fv = fit_variant(p3.Config(**kw), splits, ys, drop, lam)
        fitted[name] = fv
        results[name] = {"alpha": fv["alpha"], "n_features": fv["n_features"], "metrics": fv["metrics"]}
        m = fv["metrics"]
        print(f"{name:38s} valid {m['valid']['rmse_log']:.4f} | test {m['test']['rmse_log']:.4f} "
              f"| bias test {m['test']['bias_log']:+.3f} | rho {m['test']['rho_abs_resid_vs_fit']:+.2f}")

    # Λ óptimo por máxima verosimilitud (solo train) — evidencia sobre Box-Cox
    lam_mle = float(stats.boxcox(np.expm1(ys["train"].values))[1])

    final_name = "3. + metro log/spline (FINAL)"
    fin, base = fitted[final_name], fitted["0. Baseline (v1)"]

    # --- Segmentos (bordes definidos con train para no filtrar información) ----
    tr = splits["train"]
    edges_c = np.unique(np.quantile(tr["to_center_km"], np.linspace(0, 1, 6))); edges_c[-1] = np.inf
    edges_c[0] = -np.inf
    yhat_tr = fin["pred"]["train"]
    edges_f = np.unique(np.quantile(yhat_tr, np.linspace(0, 1, 6))); edges_f[0], edges_f[-1] = -np.inf, np.inf
    seg = {}
    for part in ("valid", "test"):
        d = splits[part].copy()
        d["yhat"] = fin["pred"][part]
        d["yhat_base"] = base["pred"][part]
        seg[part] = {
            "por_distancia_centro": {
                "final": segment_table(d, ys[part], fin["pred"][part], edges_c, "to_center_km"),
                "baseline": segment_table(d, ys[part], base["pred"][part], edges_c, "to_center_km")},
            "por_cuantil_yhat": {
                "final": segment_table(d, ys[part], fin["pred"][part], edges_f, "yhat"),
                "baseline": segment_table(d.assign(yhat=d["yhat_base"]), ys[part], base["pred"][part], edges_f, "yhat")},
        }
    lux = {part: {"final": luxury_sensitivity(ys[part], fin["pred"][part]),
                  "baseline": luxury_sensitivity(ys[part], base["pred"][part])} for part in ("valid", "test")}

    # --- Sesgo por trimestre (todas las particiones) ---------------------------
    allp = pd.concat([splits[k].assign(split=k) for k in splits])
    yall = pd.concat([ys[k] for k in splits])
    q = allp["date_posted"].dt.to_period("Q").astype(str).values
    bias_q = {}
    for nm, fv in (("baseline", base), ("final", fin)):
        r = yall.values - np.concatenate([fv["pred"][k] for k in splits])
        bias_q[nm] = pd.Series(r).groupby(q).mean().round(4).to_dict()

    # --- Figura ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    qs = sorted(bias_q["final"])
    ax[0].plot(qs, [bias_q["baseline"][k] for k in qs], "o-", color="#888", label="Baseline v1")
    ax[0].plot(qs, [bias_q["final"][k] for k in qs], "o-", color="#c2410c", label="Final")
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].axvline(qs.index("2025Q2") + .5, color="#2563eb", ls=":", label="fin train")
    ax[0].set_title("Sesgo medio del residuo por trimestre (log1p)")
    ax[0].set_xticks(range(0, len(qs), 3)); ax[0].set_xticklabels(qs[::3], rotation=45)
    ax[0].legend(); ax[0].set_ylabel("Real − predicho")

    for nm, fv, c in (("Baseline v1", base, "#888"), ("Final", fin, "#c2410c")):
        p, y = fv["pred"]["test"], ys["test"].values
        z = np.sqrt(np.abs((y - p) / np.std(y - p)))
        ax[1].scatter(p, z, s=6, alpha=.25, color=c)
        order = np.argsort(p); k = 150
        sm = pd.Series(z[order]).rolling(k, center=True, min_periods=30).mean()
        ax[1].plot(p[order], sm, color=c, lw=2.5, label=f"{nm}  (ρ={fv['metrics']['test']['rho_abs_resid_vs_fit']:+.2f})")
    ax[1].set_title("Scale-Location — test"); ax[1].set_xlabel("ŷ (log1p RUB)")
    ax[1].set_ylabel("√|residuo estandarizado|"); ax[1].legend()

    names = [n for n, *_ in variants[:5]]
    ax[2].bar(range(len(names)), [results[n]["metrics"]["test"]["rmse_log"] for n in names], color="#c2410c")
    for i, n in enumerate(names):
        v = results[n]["metrics"]["test"]["rmse_log"]; ax[2].text(i, v + .002, f"{v:.4f}", ha="center", fontsize=9)
    ax[2].set_xticks(range(len(names))); ax[2].set_xticklabels(["Baseline", "Control:\nlineal", "+ tiempo", "+ spline\nárea", "+ metro\n(FINAL)"], fontsize=9)
    ax[2].set_title("RMSE_log en test (ablación acumulativa)"); ax[2].set_ylim(0, .18)
    plt.tight_layout(); plt.savefig(FIG_DIR / "07_before_after.png", dpi=130); plt.close()

    # --- Reporte --------------------------------------------------------------
    payload = {"variantes": results, "lambda_boxcox_mle_train": lam_mle, "segmentos": seg,
               "sensibilidad_sin_top5pct_yhat": lux, "sesgo_por_trimestre": bias_q,
               "alpha_final": fin["alpha"], "smearing_duan": fin["smear"]}
    (OUT_DIR / "ablation_report.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    L = ["# Reporte de ablación y diagnóstico (Etapa 03/04)\n",
         "Ridge sobre `log1p(monthly_rent_rub)`; alpha elegido solo con *valid*. "
         "Test no interviene en ninguna decisión.\n",
         "## 1. Ablación acumulativa\n",
         "| Variante | α | RMSE valid | RMSE test | Sesgo valid | Sesgo test | MAPE test % | ρ(\\|r\\|,ŷ) test |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n, r in results.items():
        v, t = r["metrics"]["valid"], r["metrics"]["test"]
        L.append(f"| {n} | {r['alpha']} | {v['rmse_log']:.4f} | {t['rmse_log']:.4f} | {v['bias_log']:+.3f} | "
                 f"{t['bias_log']:+.3f} | {t['mape_pct']:.2f} | {t['rho_abs_resid_vs_fit']:+.2f} |")
    L += [f"\nλ de Box-Cox por máxima verosimilitud (solo train): **{lam_mle:.3f}** (≈ log). "
          "Un λ más agresivo (−0.2) reduce ρ (ver fila de control) a cambio de algo más de RMSE: "
          "es un intercambio de precisión por homocedasticidad, no una mejora neta.\n",
          "## 2. Sesgo por segmento — test (modelo final vs baseline)\n"]
    for key, title in (("por_distancia_centro", "Distancia al centro (quintiles de train)"),
                       ("por_cuantil_yhat", "Cuantil de ŷ (quintiles de train)")):
        L += [f"**{title}**\n", "| Segmento | n | RMSE baseline | RMSE final | Desv. resid. baseline | Desv. resid. final |",
              "|---|---:|---:|---:|---:|---:|"]
        for b, f in zip(seg["test"][key]["baseline"], seg["test"][key]["final"]):
            L.append(f"| {f['segmento']} | {f['n']} | {b['rmse_log']:.4f} | {f['rmse_log']:.4f} | "
                     f"{b['std_resid']:.4f} | {f['std_resid']:.4f} |")
        L.append("")
    L += ["## 3. Sensibilidad «sin propiedades de lujo» (solo diagnóstico)\n",
          "Se excluye el 5 % con mayor ŷ. **No** se propone como cambio del modelo: filtrar por una "
          "función del objetivo introduce sesgo de selección y no resuelve la causa.\n",
          "| Partición | Modelo | RMSE (todas) | RMSE (sin top 5 %) | ρ (todas) | ρ (sin top 5 %) |", "|---|---|---:|---:|---:|---:|"]
    for part in ("valid", "test"):
        for nm in ("baseline", "final"):
            a, b = lux[part][nm]["todas"], lux[part][nm]["sin_top"]
            L.append(f"| {part} | {nm} | {a['rmse_log']:.4f} | {b['rmse_log']:.4f} | "
                     f"{a['rho_abs_resid_vs_fit']:+.2f} | {b['rho_abs_resid_vs_fit']:+.2f} |")
    L += ["", "![antes/después](figures/07_before_after.png)"]
    (OUT_DIR / "ablation_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\nEscrito: reports/ablation_report.{json,md}, reports/figures/07_before_after.png")



# =========================================================================== #
# PARTE B — Diagnóstico completo del modelo ganador (valid vs test)           #
# =========================================================================== #
DIAG_DIR = Path("reports/diagnostics")
SOURCE_CSV = Path("data/rentals_clean.csv")
SPLIT_NAMES = ("train", "valid", "test")
COLORS = {"train": "#6b6560", "valid": "#c2410c", "test": "#1f2937"}


def config_from_report():
    """Reconstruye EXACTAMENTE el split usado por el 03 leyendo sus fechas de corte."""
    rep = Path("reports/preprocessing_report.json")
    if rep.exists():
        sc = json.loads(rep.read_text(encoding="utf-8")).get("split_cutoffs")
        if sc:
            return p3.Config(split_mode="fixed", train_end=sc["split_train_end"], valid_end=sc["split_valid_end"])
    return p3.Config()


def _read_table(stem: Path) -> pd.DataFrame:
    pq, csv = stem.with_suffix(".parquet"), stem.with_suffix(".csv")
    path = pq if pq.exists() else csv
    if not path.exists():
        raise FileNotFoundError(f"No existe {pq} ni {csv}. Corre antes 03_preprocessing.py")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, index_col=0)


def load_real_pipeline(bundle_path: Path, data_dir: Path = Path("data/processed/splits")) -> dict:
    """Carga lo mismo que usa 04: preprocesador (fit en train) + modelo ganador (fit SOLO en train)."""
    bundle = joblib.load(bundle_path)
    pre = joblib.load(bundle["preprocessor"])
    # Protección: si el 03 se corrió DESPUÉS del 04, el modelo del bundle se entrenó con otro split/preprocesador
    # y el número de features puede coincidir (no habría error): el diagnóstico sería una mezcla silenciosa.
    t_bundle = Path(bundle_path).stat().st_mtime
    for dep in (Path(bundle["preprocessor"]), Path("reports/preprocessing_report.json")):
        if dep.exists() and dep.stat().st_mtime > t_bundle + 1:
            print("\n" + "!" * 78 + f"\n[AVISO] {dep} es MÁS NUEVO que {bundle_path}.\n"
                  "Corriste 03 después de 04: el modelo NO corresponde a estos splits. Vuelve a correr\n"
                  "04_model_selection.py (y 05) antes de interpretar estas métricas.\n" + "!" * 78 + "\n")
            break
    X = {s: _read_table(data_dir / f"X_{s}") for s in SPLIT_NAMES}
    yd = {s: _read_table(data_dir / f"y_{s}") for s in SPLIT_NAMES}
    ylog = {s: pd.Series(np.log1p(yd[s]["monthly_rent_rub"].astype(float).to_numpy()), index=yd[s].index)
            for s in SPLIT_NAMES}
    Xt = {s: np.asarray(pre.transform(X[s])) for s in SPLIT_NAMES}
    dates = {s: None for s in SPLIT_NAMES}
    if SOURCE_CSV.exists():
        d = pd.read_csv(SOURCE_CSV, usecols=["date_posted"], parse_dates=["date_posted"])["date_posted"]
        for s in SPLIT_NAMES:
            if set(X[s].index).issubset(set(d.index)):
                dates[s] = d.loc[X[s].index].reset_index(drop=True)
    pred = {s: bundle["model"].predict(Xt[s]) for s in SPLIT_NAMES}
    return {"bundle": bundle, "pre": pre, "model": bundle["model"], "smear": float(bundle["smearing_factor"]),
            "X": X, "Xt": Xt, "ylog": ylog, "pred": pred, "dates": dates, "names": list(pre.get_feature_names_out())}


# --------------------------------------------------------------------------- #
# Métricas                                                                    #
# --------------------------------------------------------------------------- #
def full_metrics(y_log, p_log, smear, n_feat, Xt=None, dates=None) -> dict:
    y_log, p_log = np.asarray(y_log, float), np.asarray(p_log, float)
    e = y_log - p_log
    n = len(e)
    y, yh = np.expm1(y_log), np.exp(p_log) * smear - 1.0          # RUB, con smearing de Duan
    er = y - yh
    pe = er / y                                                    # error porcentual con signo
    sst_l, sst_r = np.sum((y_log - y_log.mean()) ** 2), np.sum((y - y.mean()) ** 2)
    r2_l, r2_r = 1 - np.sum(e ** 2) / sst_l, 1 - np.sum(er ** 2) / sst_r
    z = (e - e.mean()) / e.std(ddof=1)
    m = {
        # --- escala log (la del modelo) ---
        "MSE_log": np.mean(e ** 2), "RMSE_log": np.sqrt(np.mean(e ** 2)), "MAE_log": np.mean(np.abs(e)),
        "MedAE_log": np.median(np.abs(e)), "R2_log": r2_l,
        "R2_ajustado_log": 1 - (1 - r2_l) * (n - 1) / (n - n_feat - 1),
        "Sesgo_log (media real-pred)": e.mean(), "Desv_residuos_log": e.std(ddof=1),
        "Asimetria_residuos": stats.skew(e), "Exceso_curtosis_residuos": stats.kurtosis(e),
        "Max_|z|_residuo": np.max(np.abs(z)),
        # --- escala RUB (retransformada) ---
        "MSE_RUB": np.mean(er ** 2), "RMSE_RUB": np.sqrt(np.mean(er ** 2)), "MAE_RUB": np.mean(np.abs(er)),
        "R2_RUB": r2_r, "Sesgo_RUB (media real-pred)": er.mean(),
        "MAPE_%": np.mean(np.abs(pe)) * 100,
        "MSPE (≈MAPE², error % al cuadrado) [%²]": np.mean((pe * 100) ** 2),
        "RMSPE_% (raíz del MSPE)": np.sqrt(np.mean((pe * 100) ** 2)),
        "sMAPE_%": np.mean(2 * np.abs(er) / (np.abs(y) + np.abs(yh))) * 100,
        "Aciertos_±10%": np.mean(np.abs(pe) <= 0.10) * 100, "Aciertos_±20%": np.mean(np.abs(pe) <= 0.20) * 100,
        # --- supuestos del modelo lineal ---
        "Spearman(|resid|, ŷ) [heteroced.]": stats.spearmanr(p_log, np.abs(e))[0],
    }
    jb = stats.jarque_bera(e)
    m["Jarque-Bera p-valor"] = jb.pvalue
    if Xt is not None:  # Breusch-Pagan (LM) con los regresores transformados
        A = np.c_[np.ones(n), Xt]
        g = e ** 2
        coef = np.linalg.lstsq(A, g, rcond=None)[0]
        r2g = 1 - np.sum((g - A @ coef) ** 2) / np.sum((g - g.mean()) ** 2)
        df_bp = max(np.linalg.matrix_rank(A) - 1, 1)
        m["Breusch-Pagan p-valor"] = stats.chi2.sf(n * r2g, df_bp)
    if dates is not None:  # Durbin-Watson ordenando por fecha (autocorrelación temporal)
        ee = e[np.argsort(np.asarray(dates), kind="stable")]
        m["Durbin-Watson (≈2 = sin autocorr.)"] = np.sum(np.diff(ee) ** 2) / np.sum(ee ** 2)
    m["n"] = n
    return {k: float(v) for k, v in m.items()}


def traffic_light(M: dict) -> list[tuple]:
    """Reglas PRÁCTICAS (heurísticas, no estándares): (criterio, valid, test, regla, estado)."""
    def st(v, ok, warn, absval=True, lower_better=True):
        x = abs(v) if absval else v
        if lower_better:
            return "OK" if x <= ok else ("ATENCIÓN" if x <= warn else "PROBLEMA")
        return "OK" if x >= ok else ("ATENCIÓN" if x >= warn else "PROBLEMA")
    rows = []
    def add(label, key_fn, rule, ok, warn, **kw):
        v, t = key_fn("valid"), key_fn("test")
        worst = max([st(v, ok, warn, **kw), st(t, ok, warn, **kw)], key=["OK", "ATENCIÓN", "PROBLEMA"].index)
        rows.append((label, v, t, rule, worst))
    add("Sobreajuste: RMSE_log (valid | test) / train", lambda s: M[s]["RMSE_log"] / M["train"]["RMSE_log"], "≤1.10 OK, ≤1.25 atención", 1.10, 1.25)
    add("Sesgo medio (log)", lambda s: M[s]["Sesgo_log (media real-pred)"], "|sesgo| ≤0.02 OK, ≤0.05 atención", 0.02, 0.05)
    add("R² (log)", lambda s: M[s]["R2_log"], "≥0.90 OK, ≥0.80 atención", 0.90, 0.80, absval=False, lower_better=False)
    add("MAPE %", lambda s: M[s]["MAPE_%"], "≤10 OK, ≤15 atención", 10, 15)
    add("Asimetría de residuos", lambda s: M[s]["Asimetria_residuos"], "|skew| ≤0.5 OK, ≤1 atención", 0.5, 1.0)
    add("Exceso de curtosis", lambda s: M[s]["Exceso_curtosis_residuos"], "|kurt| ≤1 OK, ≤3 atención", 1.0, 3.0)
    add("Heterocedasticidad Spearman(|r|,ŷ)", lambda s: M[s]["Spearman(|resid|, ŷ) [heteroced.]"], "|ρ| ≤0.10 OK, ≤0.25 atención", 0.10, 0.25)
    if "Durbin-Watson (≈2 = sin autocorr.)" in M["valid"]:
        add("Autocorrelación (Durbin-Watson)", lambda s: M[s]["Durbin-Watson (≈2 = sin autocorr.)"] - 2, "|DW−2| ≤0.5 OK, ≤1 atención", 0.5, 1.0)
    return rows


# --------------------------------------------------------------------------- #
# Fórmula del modelo                                                          #
# --------------------------------------------------------------------------- #
def coefficient_table(P: dict, model=None) -> pd.DataFrame:
    model = model or P["model"]
    pre, names = P["pre"], P["names"]
    coef = np.asarray(model.coef_, float).ravel()
    sd = P["Xt"]["train"].std(axis=0)
    rows = []
    for name, sl in pre.output_indices_.items():
        if name == "remainder":
            continue
        tr = pre.named_transformers_[name]
        scaler = tr.named_steps.get("scaler") if hasattr(tr, "named_steps") else None
        scale = getattr(scaler, "scale_", None)
        for j, idx in enumerate(range(sl.start, sl.stop)):
            per_unit = pct = np.nan
            if scale is not None and name in ("numeric", "onehot"):   # interpretable por unidad original
                per_unit = coef[idx] / scale[j]
                pct = (np.exp(per_unit) - 1) * 100
            rows.append({"bloque": name, "termino": names[idx], "coeficiente": coef[idx],
                         "desv_est_train": sd[idx], "efecto_estandarizado": coef[idx] * sd[idx],
                         "coef_por_unidad_original": per_unit, "efecto_pct_por_unidad": pct})
    df = pd.DataFrame(rows)
    df.attrs["intercepto"] = float(np.ravel(model.intercept_)[0])
    return df


def formula_text(tbl: pd.DataFrame, title: str, smear: float) -> str:
    b0 = tbl.attrs["intercepto"]
    L = [f"{'=' * 78}", title, f"{'=' * 78}",
         "(One-hot con drop='first': la categoría base —la omitida de cada variable— queda en el intercepto.)",
         "ŷ_log = log1p(renta_estimada) = b0 + Σ ( coeficiente_i × x_i )",
         "donde x_i son las columnas YA TRANSFORMADAS por el preprocesador de la etapa 03",
         "(imputación + escalado + splines + one-hot + target encoding).",
         "", f"  b0 (intercepto) = {b0:+.6f}"]
    for bloque, g in tbl.groupby("bloque", sort=False):
        L.append(f"\n  --- bloque «{bloque}» ({len(g)} términos) ---")
        for _, r in g.iterrows():
            L.append(f"  {r.coeficiente:+12.6f} × {r.termino}")
    L += ["", "Retransformación a RUB (corrección de Duan):",
          f"  renta_RUB = exp(ŷ_log) × S − 1,   con S = {smear:.6f}",
          "", "Lectura de efectos (solo bloques numeric/onehot; ver coeficientes.csv):",
          "  efecto_pct_por_unidad = (exp(coef/escala) − 1) × 100  → % de cambio de la renta por unidad",
          "  original, manteniendo lo demás fijo. CUIDADO: total_area, log_metro y to_center_km entran a la vez",
          "  como término lineal Y como spline, así que su efecto total está repartido entre ambos y el %",
          "  del término lineal es solo una parte. Para splines y tiempo NO hay efecto por unidad: la",
          "  contribución es una curva (ver gráficas) y los coeficientes sueltos no se interpretan."]
    mx = tbl.coeficiente.abs().max()
    if mx > 5:
        L += ["", f"AVISO DE COLINEALIDAD: |coef| máximo = {mx:.1f}. Hay coeficientes grandes que se cancelan",
              "entre sí (p. ej. log_metro aparece en el bloque numérico y en splines). La fórmula es exacta",
              "para predecir, pero los coeficientes individuales NO son interpretables; usa la fórmula",
              "del Ridge compañero (más abajo) para explicar efectos."]
    return "\n".join(L)


def ridge_companion(P: dict):
    """Ridge sobre la MISMA matriz (alpha elegido en valid): coeficientes estables e interpretables."""
    best = None
    for a in (0.01, 0.1, 1.0, 10.0, 100.0):
        m = Ridge(alpha=a).fit(P["Xt"]["train"], P["ylog"]["train"].to_numpy())
        rm = float(np.sqrt(np.mean((P["ylog"]["valid"].to_numpy() - m.predict(P["Xt"]["valid"])) ** 2)))
        if best is None or rm < best[0]:
            best = (rm, a, m)
    return best[2], best[1]


# --------------------------------------------------------------------------- #
# Gráficas                                                                    #
# --------------------------------------------------------------------------- #
def _roll(x, y, k=120):
    o = np.argsort(x)
    return x[o], pd.Series(y[o]).rolling(k, center=True, min_periods=30).mean().to_numpy()


def make_figures(P: dict, M: dict, tbl: pd.DataFrame, out: Path, lc_reps: int = 3) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.grid": True, "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})
    out.mkdir(parents=True, exist_ok=True)
    files = []
    R = {s: np.asarray(P["ylog"][s]) - P["pred"][s] for s in SPLIT_NAMES}
    idx = {s: np.asarray(P["ylog"][s].index) for s in SPLIT_NAMES}

    def pair(fname, title, draw, size=(13, 4.8)):
        fig, axs = plt.subplots(1, 2, figsize=size)
        for ax, s in zip(axs, ("valid", "test")):
            draw(ax, s)
        fig.suptitle(title, fontsize=13, y=1.02)
        fig.tight_layout()
        p = out / fname
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        files.append(p)

    def hist(ax, s):
        r = R[s]; mu, sd = r.mean(), r.std(ddof=1)
        ax.hist(r, bins=40, density=True, color="#e0a48a", edgecolor="white", label="Histograma de residuos")
        x = np.linspace(r.min(), r.max(), 300)
        ax.plot(x, stats.norm.pdf(x, mu, sd), "k--", lw=2, label=f"Normal (μ={mu:.3f}, σ={sd:.3f})")
        ax.axvline(0, color="#2563eb", ls=":", lw=2, label="Cero (sin sesgo)")
        ax.axvline(mu, color="#c2410c", lw=2, label=f"Media real ({mu:+.3f})")
        ax.set_title(f"{s} — skew={stats.skew(r):.3f} | exceso curtosis={stats.kurtosis(r):.3f}")
        ax.set_xlabel("Residuo = real − predicho (log1p RUB)"); ax.set_ylabel("Densidad"); ax.legend(fontsize=8)
    pair("01_histograma_residuos_valid_vs_test.png", "Distribución de residuos (escala log)", hist)

    def qq(ax, s):
        z = (R[s] - R[s].mean()) / R[s].std(ddof=1)
        (osm, osr), (sl, ic, rr) = stats.probplot(z, dist="norm")
        ax.scatter(osm, osr, s=8, alpha=.5, color=COLORS["valid" if s == "valid" else "test"])
        lim = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
        ax.plot(lim, lim, color="#d62728", lw=2, label="Normal perfecta (y = x)")
        ax.set_title(f"{s} — QQ-plot (R² del ajuste = {rr ** 2:.4f})")
        ax.set_xlabel("Cuantiles teóricos (Normal)"); ax.set_ylabel("Cuantiles de residuos estandarizados"); ax.legend(fontsize=8)
    pair("02_qqplot_valid_vs_test.png", "QQ-plot de residuos: colas pesadas se ven como curvas lejos de la diagonal", qq)

    def scale_loc(ax, s):
        f, r = P["pred"][s], R[s]; z = np.sqrt(np.abs((r - r.mean()) / r.std(ddof=1)))
        ax.scatter(f, z, s=8, alpha=.3, color="#555")
        xs, ys = _roll(f, z); ax.plot(xs, ys, color="#d62728", lw=2.5, label="Tendencia media (horizontal = homocedástico)")
        for i in np.argsort(-z)[:3]:
            ax.annotate(str(idx[s][i]), (f[i], z[i]), fontsize=8, xytext=(3, 3), textcoords="offset points")
        ax.set_title(f"{s} — Scale-Location (ρ={M[s]['Spearman(|resid|, ŷ) [heteroced.]']:+.2f})")
        ax.set_xlabel("Valores ajustados ŷ (log1p RUB)"); ax.set_ylabel("√|residuo estandarizado|"); ax.legend(fontsize=8)
    pair("03_scale_location_valid_vs_test.png", "Scale-Location (homocedasticidad)", scale_loc)

    def rvf(ax, s):
        f, r = P["pred"][s], R[s]
        ax.scatter(f, r, s=8, alpha=.3, color=COLORS["valid" if s == "valid" else "test"])
        xs, ys = _roll(f, r); ax.plot(xs, ys, color="#d62728", lw=2.5, label="Tendencia media")
        ax.axhline(0, color="k", ls="--"); ax.set_title(f"{s} — Residuo vs ajustado")
        ax.set_xlabel("ŷ (log1p RUB)"); ax.set_ylabel("Residuo (real − predicho)"); ax.legend(fontsize=8)
    pair("04_residuo_vs_ajustado_valid_vs_test.png", "Residuos vs valores ajustados (linealidad / patrones)", rvf)

    def rvp(ax, s):
        y, f = np.asarray(P["ylog"][s]), P["pred"][s]
        ax.scatter(f, y, s=8, alpha=.3, color=COLORS["valid" if s == "valid" else "test"])
        lim = [min(y.min(), f.min()), max(y.max(), f.max())]; ax.plot(lim, lim, color="#d62728", lw=2, label="Predicción perfecta")
        ax.set_title(f"{s} — Real vs predicho (R²={M[s]['R2_log']:.3f}, RMSE={M[s]['RMSE_log']:.4f})")
        ax.set_xlabel("Predicho (log1p RUB)"); ax.set_ylabel("Real (log1p RUB)"); ax.legend(fontsize=8)
    pair("05_real_vs_predicho_valid_vs_test.png", "Real vs predicho", rvp)

    if all(P["dates"][s] is not None for s in ("valid", "test")):
        def tdrift(ax, s):
            d = pd.DataFrame({"m": pd.to_datetime(P["dates"][s]).dt.to_period("M").astype(str), "r": R[s]})
            g = d.groupby("m").r.agg(["mean", "std"])
            ax.errorbar(range(len(g)), g["mean"], yerr=g["std"], fmt="o-", color="#c2410c", capsize=4, label="media ± 1σ del residuo")
            ax.axhline(0, color="k", ls="--"); ax.set_xticks(range(len(g))); ax.set_xticklabels(g.index, rotation=45)
            ax.set_title(f"{s} — sesgo por mes (deriva temporal)"); ax.set_ylabel("Residuo medio (log)"); ax.legend(fontsize=8)
        pair("06_residuos_en_el_tiempo_valid_vs_test.png", "Deriva temporal: ¿el sesgo crece con el tiempo?", tdrift)

    # --- Curva de aprendizaje ESTILO ORIGINAL: train vs CV (KFold 5 sin mezclar sobre train) ----
    from sklearn.model_selection import KFold, learning_curve
    n_lc, tr_lc, cv_lc = learning_curve(clone(P["model"]), P["Xt"]["train"], np.asarray(P["ylog"]["train"]),
                                        cv=KFold(5), train_sizes=np.linspace(0.15, 1.0, 8),
                                        scoring="neg_root_mean_squared_error", n_jobs=1)
    tr_m, cv_m = -tr_lc.mean(axis=1), -cv_lc.mean(axis=1)
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    ax[0].plot(n_lc, tr_m, "o-", color="#1f77b4", lw=2.5, label="RMSE Entrenamiento (Train)")
    ax[0].plot(n_lc, cv_m, "s-", color="#c2410c", lw=2.5, label="RMSE Validación Cruzada (CV)")
    ax[0].fill_between(n_lc, tr_m, cv_m, color="#c2410c", alpha=.10)
    ax[0].set_title("Curva de Aprendizaje (Diagnóstico de Overfitting)"); ax[0].set_xlabel("Cantidad de filas de entrenamiento")
    ax[0].set_ylabel("Error RMSE (escala log)"); ax[0].legend()
    ax[0].text(0.02, 0.04, "CV = KFold(5) sobre train (mezcla periodos): mide ajuste 'dentro del tiempo'.\n"
               "La brecha fuera de tiempo (train→valid/test) se ve en la figura 07b.", transform=ax[0].transAxes, fontsize=8, color="#555")
    ax[1].bar(["Fit (Entrenamiento)", "Valid", "Test"], vals_lc := [M[s]["RMSE_log"] for s in SPLIT_NAMES],
              color=["#6b6560", "#c2410c", "#2a2a2a"], width=.5)
    for i, v in enumerate(vals_lc):
        ax[1].text(i, v + max(vals_lc) * .01, f"{v:.4f}", ha="center", fontweight="bold")
    ax[1].set_ylim(0, max(vals_lc) * 1.25); ax[1].set_title("Comparación de Error RMSE_log entre Particiones"); ax[1].set_ylabel("RMSE (log1p(RUB))")
    fig.tight_layout(); p = out / "07_curva_aprendizaje_estilo_original.png"; fig.savefig(p, dpi=130); plt.close(fig); files.append(p)

    # --- Curva de aprendizaje FUERA DE TIEMPO (train / valid / test) + barras RMSE ---
    Xtr, ytr = P["Xt"]["train"], np.asarray(P["ylog"]["train"]); n = len(ytr)
    sizes = np.unique((np.linspace(0.10, 1.0, 8) * n).astype(int))
    rng = np.random.default_rng(42)
    curves = {s: [] for s in SPLIT_NAMES}
    for sz in sizes:
        acc = {s: [] for s in SPLIT_NAMES}
        for _ in range(1 if sz == n else lc_reps):
            sub = rng.choice(n, sz, replace=False)
            m = clone(P["model"]).fit(Xtr[sub], ytr[sub])
            acc["train"].append(np.sqrt(np.mean((ytr[sub] - m.predict(Xtr[sub])) ** 2)))
            for s in ("valid", "test"):
                acc[s].append(np.sqrt(np.mean((np.asarray(P["ylog"][s]) - m.predict(P["Xt"][s])) ** 2)))
        for s in SPLIT_NAMES:
            curves[s].append((np.mean(acc[s]), np.std(acc[s])))
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    for s, mk in (("train", "o"), ("valid", "s"), ("test", "^")):
        mu = np.array([c[0] for c in curves[s]]); sd = np.array([c[1] for c in curves[s]])
        ax[0].plot(sizes, mu, marker=mk, color=COLORS[s], lw=2.2, label=f"RMSE {s}")
        ax[0].fill_between(sizes, mu - sd, mu + sd, color=COLORS[s], alpha=.12)
    ax[0].set_title("Curva fuera de tiempo: entrena con submuestras de train, evalúa en valid y test (posteriores)"); ax[0].set_xlabel("Filas de entrenamiento")
    ax[0].set_ylabel("RMSE (log1p RUB)"); ax[0].legend()
    vals = [M[s]["RMSE_log"] for s in SPLIT_NAMES]
    ax[1].bar(["Fit (train)", "Valid", "Test"], vals, color=[COLORS[s] for s in SPLIT_NAMES], width=.5)
    for i, v in enumerate(vals):
        ax[1].text(i, v + max(vals) * .01, f"{v:.4f}", ha="center", fontweight="bold")
    ax[1].set_ylim(0, max(vals) * 1.25); ax[1].set_title("RMSE_log por partición (modelo ajustado SOLO con train)"); ax[1].set_ylabel("RMSE (log1p RUB)")
    fig.tight_layout(); p = out / "07b_curva_aprendizaje_fuera_de_tiempo.png"; fig.savefig(p, dpi=130); plt.close(fig); files.append(p)

    # --- Coeficientes más influyentes -------------------------------------------
    top = tbl.reindex(tbl.efecto_estandarizado.abs().sort_values(ascending=False).index).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top.termino, top.efecto_estandarizado, color=np.where(top.efecto_estandarizado >= 0, "#c2410c", "#2563eb"))
    ax.axvline(0, color="k"); ax.set_xlabel("coeficiente × desv. estándar (efecto en log-renta por 1σ del término)")
    ax.set_title("Top 20 términos por efecto estandarizado (modelo ganador)")
    fig.tight_layout(); p = out / "08_coeficientes_top20.png"; fig.savefig(p, dpi=130); plt.close(fig); files.append(p)
    return files


# --------------------------------------------------------------------------- #
# Orquestación de la parte B                                                  #
# --------------------------------------------------------------------------- #
def run_diagnostics(bundle_path: Path, out: Path, lc_reps: int) -> None:
    P = load_real_pipeline(bundle_path)
    name = P["bundle"].get("model_name", type(P["model"]).__name__)
    nf = P["Xt"]["train"].shape[1]
    tot = sum(len(P["ylog"][s]) for s in SPLIT_NAMES)
    print("Split detectado (leído de data/processed/splits):")
    for s in SPLIT_NAMES:
        d = P["dates"][s]
        rng_txt = f"  [{d.min().date()} → {d.max().date()}]" if d is not None else ""
        print(f"  {s:5s} n={len(P['ylog'][s]):6d} ({100 * len(P['ylog'][s]) / tot:5.2f} %){rng_txt}")
    M = {s: full_metrics(P["ylog"][s], P["pred"][s], P["smear"], nf, P["Xt"][s], P["dates"][s]) for s in SPLIT_NAMES}
    out.mkdir(parents=True, exist_ok=True)

    # 1) Tabla de métricas
    df = pd.DataFrame(M)[list(SPLIT_NAMES)]
    df.to_csv(out / "metricas_comparativas.csv", encoding="utf-8-sig")
    tl = traffic_light(M)
    L = [f"# Diagnóstico completo — modelo ganador: **{name}** ({nf} features)\n",
         "Split: " + " / ".join(f"{s} n={len(P['ylog'][s])} ({100 * len(P['ylog'][s]) / tot:.1f} %)" for s in SPLIT_NAMES) + "\n",
         "Modelo ajustado SOLO con *train*; *valid* y *test* son datos no vistos (el 05 reentrena con train+valid, "
         "por eso sus cifras de valid/test pueden diferir ligeramente).\n",
         "## 1. Métricas por partición\n", "| Métrica | train | valid | test |", "|---|---:|---:|---:|"]
    for k, r in df.iterrows():
        f = (lambda v: f"{v:,.0f}") if k in ("n", "MSE_RUB", "RMSE_RUB", "MAE_RUB", "Sesgo_RUB (media real-pred)") else (lambda v: f"{v:.4g}" if abs(v) < 1e-3 else f"{v:.4f}")
        L.append(f"| {k} | {f(r['train'])} | {f(r['valid'])} | {f(r['test'])} |")
    L += ["\n> **MAPE²:** el cuadrado literal del MAPE no tiene interpretación estándar; se reporta el **MSPE** "
          "(media del error porcentual al cuadrado, en %²) y su raíz RMSPE, que son su equivalente correcto.\n",
          "## 2. Semáforo de calidad (reglas prácticas, NO estándares universales)\n",
          "| Criterio | valid | test | Regla | Estado |", "|---|---:|---:|---|---|"]
    for lab, v, t, rule, est in tl:
        L.append(f"| {lab} | {v:.4f} | {t:.4f} | {rule} | **{est}** |")
    L += ["\n**Cómo leerlo:** con n ≈ 2 000 los tests de Jarque-Bera y Breusch-Pagan rechazan casi siempre "
          "(son muy sensibles); guíate por magnitud (skew, curtosis, ρ) y por las gráficas, no solo por el p-valor.\n",
          "Valid y test deberían dar métricas parecidas entre sí y cercanas a train; una brecha grande entre "
          "train y valid/test indica sobreajuste, y un sesgo creciente por mes indica deriva temporal.\n"]
    (out / "metricas_comparativas.md").write_text("\n".join(L), encoding="utf-8")

    # 2) Fórmula ganadora + Ridge compañero
    tbl = coefficient_table(P)
    tbl.to_csv(out / "coeficientes_ganador.csv", index=False, encoding="utf-8-sig")
    txt = formula_text(tbl, f"FÓRMULA DEL MODELO GANADOR: {name}", P["smear"])
    rid, a = ridge_companion(P)
    pr_v, pr_t = (np.sqrt(np.mean((np.asarray(P["ylog"][s]) - rid.predict(P["Xt"][s])) ** 2)) for s in ("valid", "test"))
    tbl_r = coefficient_table(P, rid)
    tbl_r.to_csv(out / "coeficientes_ridge_companero.csv", index=False, encoding="utf-8-sig")
    txt_r = formula_text(tbl_r, f"FÓRMULA COMPAÑERA INTERPRETABLE: Ridge(alpha={a})  —  RMSE_log valid {pr_v:.4f} | test {pr_t:.4f} "
                                f"(ganador: valid {M['valid']['RMSE_log']:.4f} | test {M['test']['RMSE_log']:.4f})", P["smear"])
    (out / "formula_ganadora.txt").write_text(txt + "\n\n\n" + txt_r, encoding="utf-8")

    # 3) Gráficas
    files = make_figures(P, M, tbl, out, lc_reps)

    # 4) Consola
    print("\n" + "=" * 78 + f"\nMÉTRICAS — {name}\n" + "=" * 78)
    show = ["RMSE_log", "MSE_log", "MAE_log", "R2_log", "R2_ajustado_log", "Sesgo_log (media real-pred)", "RMSE_RUB", "MAE_RUB",
            "MAPE_%", "MSPE (≈MAPE², error % al cuadrado) [%²]", "R2_RUB", "Aciertos_±10%", "Asimetria_residuos",
            "Exceso_curtosis_residuos", "Spearman(|resid|, ŷ) [heteroced.]"]
    print(df.loc[show].to_string(float_format=lambda v: f"{v:,.4f}"))
    print("\nSEMÁFORO (reglas prácticas):")
    for lab, v, t, rule, est in tl:
        print(f"  [{est:9s}] {lab:40s} valid {v:8.4f} | test {t:8.4f}   ({rule})")
    print("\n" + txt + "\n\n" + txt_r)
    print("\nEscrito en", out, ":", ", ".join(sorted(p.name for p in files)),
          "+ metricas_comparativas.{csv,md}, formula_ganadora.txt, coeficientes_*.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description="07 — Ablación + diagnóstico completo del modelo ganador")
    ap.add_argument("--bundle", default="artifacts/best_model_04.joblib", help="Salida del 04 (modelo ajustado en train)")
    ap.add_argument("--out", default="reports/diagnostics")
    ap.add_argument("--lc-reps", type=int, default=3, help="Repeticiones por tamaño en la curva de aprendizaje")
    ap.add_argument("--skip-ablation", action="store_true", help="Omitir la ablación (Parte A, ~2-3 min)")
    ap.add_argument("--skip-diagnostics", action="store_true", help="Omitir el diagnóstico del modelo ganador (Parte B)")
    a = ap.parse_args()
    if not a.skip_diagnostics:
        run_diagnostics(Path(a.bundle), Path(a.out), a.lc_reps)
    if not a.skip_ablation:
        run_ablation()


if __name__ == "__main__":
    main()