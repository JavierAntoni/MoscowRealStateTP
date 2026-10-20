#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script generador del Informe Académico Exhaustivo de Análisis de Datos y Modelado.
Lee los artefactos de /reports y /reports/diagnostics, filtra redundancias y
construye el documento Markdown completo para defensa de examen.
"""

import json
from pathlib import Path
import pandas as pd


def resolve_reports_dir() -> Path:
    """Detecta automáticamente la carpeta reports/ sin importar desde dónde se ejecute."""
    cwd = Path.cwd()
    if (cwd / "diagnostics").exists() and (cwd / "preprocessing_report.json").exists():
        return cwd
    if (cwd / "reports").exists():
        return cwd / "reports"
    return cwd


def fmt_num(val, decimals=4):
    if pd.isna(val) or val is None:
        return "—"
    if isinstance(val, (int, float)):
        return f"{val:,.{decimals}f}"
    return str(val)


def df_to_md_table(df: pd.DataFrame) -> str:
    """Convierte un DataFrame a tabla Markdown sin depender de tabulate."""
    headers = list(df.columns)
    header_line = "| " + " | ".join(str(h) for h in headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = [str(row[col]) for col in headers]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header_line, sep_line] + rows)


def build_report():
    reports_dir = resolve_reports_dir()
    diag_dir = reports_dir / "diagnostics"

    # 1. Carga de archivos JSON
    with open(reports_dir / "preprocessing_report.json", "r", encoding="utf-8") as f:
        preproc = json.load(f)

    with open(reports_dir / "training_mlflow_report.json", "r", encoding="utf-8") as f:
        mlflow_rep = json.load(f)

    with open(reports_dir / "ablation_report.json", "r", encoding="utf-8") as f:
        ablation = json.load(f)

    # 2. Carga de archivos CSV de diagnóstico
    df_coef = pd.read_csv(diag_dir / "coeficientes_ganador.csv")
    df_vif = pd.read_csv(diag_dir / "vif_variables.csv")
    df_metrics = pd.read_csv(diag_dir / "metricas_comparativas.csv")
    df_metrics.rename(columns={df_metrics.columns[0]: "Metrica"}, inplace=True)

    # Consolidación sin redundancia: unir coeficientes con Tolerancia (1/VIF)
    df_master = df_coef.merge(
        df_vif[["termino", "tolerancia (1/VIF)"]],
        on="termino",
        how="left"
    )

    # Preparar tabla de Efectos Directamente Interpretables (Bloques numeric y onehot)
    df_interp = df_master[df_master["coef_por_unidad_original"].notna()].copy()
    df_interp_view = pd.DataFrame({
        "Bloque": df_interp["bloque"],
        "Término": df_interp["termino"].apply(lambda x: f"`{x}`"),
        "Coef. Escala Transf.": df_interp["coeficiente"].apply(lambda x: f"{x:+.6f}"),
        "DE (Train)": df_interp["desv_est_train"].apply(lambda x: f"{x:.4f}"),
        "Efecto Estandarizado": df_interp["efecto_estandarizado"].apply(lambda x: f"{x:+.6f}"),
        "Coef. Unidad Orig.": df_interp["coef_por_unidad_original"].apply(lambda x: f"{x:+.6f}"),
        "Efecto % por Unidad": df_interp["efecto_pct_por_unidad"].apply(lambda x: f"{x:+.2f}%"),
        "VIF": df_interp["VIF"].apply(lambda x: f"{x:.2f}")
    })

    # Preparar tabla maestra completa (57 términos ordenados por |efecto_estandarizado|)
    df_master["abs_efecto"] = df_master["efecto_estandarizado"].abs()
    df_master_sorted = df_master.sort_values("abs_efecto", ascending=False)
    df_all_terms_view = pd.DataFrame({
        "#": range(1, len(df_master_sorted) + 1),
        "Bloque": df_master_sorted["bloque"],
        "Término": df_master_sorted["termino"].apply(lambda x: f"`{x}`"),
        "Coeficiente": df_master_sorted["coeficiente"].apply(lambda x: f"{x:+.6f}"),
        "Efecto Estand. (1σ)": df_master_sorted["efecto_estandarizado"].apply(lambda x: f"{x:+.6f}"),
        "VIF": df_master_sorted["VIF"].apply(lambda x: f"{x:.2f}"),
        "Tolerancia (1/VIF)": df_master_sorted["tolerancia (1/VIF)"].apply(lambda x: f"{x:.4f}"),
        "R² Auxiliar": df_master_sorted["R2_auxiliar"].apply(lambda x: f"{x:.4f}"),
        "Estado VIF": df_master_sorted["estado_vif"]
    })

    # Preparar tabla de las 27 métricas comparativas
    def format_metric_row(row):
        m = row["Metrica"]
        vals = [row["train"], row["valid"], row["test"]]
        if m == "n":
            return [f"{int(v):,}" for v in vals]
        if "RUB" in m and "R2" not in m:
            return [f"{v:,.2f}" for v in vals]
        if "p-valor" in m:
            return [f"{v:.4e}" if v < 0.001 else f"{v:.4f}" for v in vals]
        if "%" in m:
            return [f"{v:.2f}%" for v in vals]
        return [f"{v:.6f}" for v in vals]

    metrics_rows = []
    for _, r in df_metrics.iterrows():
        t_val, v_val, te_val = format_metric_row(r)
        metrics_rows.append({
            "Métrica de Evaluación / Diagnóstico": f"**{r['Metrica']}**",
            "Train (n = 16,012)": t_val,
            "Valid (n = 1,995)": v_val,
            "Test (n = 1,993)": te_val
        })
    df_metrics_view = pd.DataFrame(metrics_rows)

    # Preparar tabla de Ablación
    abl_rows = []
    for var_name, var_data in ablation["variantes"].items():
        m_tr = var_data["metrics"]["train"]
        m_va = var_data["metrics"]["valid"]
        m_te = var_data["metrics"]["test"]
        abl_rows.append({
            "Variante Experimental": f"**{var_name}**",
            "α (Ridge)": var_data["alpha"],
            "Features": var_data["n_features"],
            "RMSE_log (Train)": f"{m_tr['rmse_log']:.4f}",
            "RMSE_log (Valid)": f"{m_va['rmse_log']:.4f}",
            "RMSE_log (Test)": f"{m_te['rmse_log']:.4f}",
            "Sesgo_log (Test)": f"{m_te['bias_log']:+.4f}",
            "MAPE % (Test)": f"{m_te['mape_pct']:.2f}%",
            "MAE RUB (Test)": f"{m_te['mae_rub']:,.2f}",
            "ρ(|resid|, ŷ) Test": f"{m_te['rho_abs_resid_vs_fit']:.4f}"
        })
    # Agregar fila del modelo ganador final de MLflow para contraste directo
    abl_rows.append({
        "Variante Experimental": "**4. Ganador MLflow (`LinearRegression` + `district_adjust`)**",
        "α (Ridge)": "0.0 (OLS)",
        "Features": preproc["n_features_out"],
        "RMSE_log (Train)": "0.0836",
        "RMSE_log (Valid)": f"{mlflow_rep['valid_rmse_log']:.4f}",
        "RMSE_log (Test)": f"{mlflow_rep['test_rmse_log']:.4f}",
        "Sesgo_log (Test)": "+0.0233",
        "MAPE % (Test)": f"{mlflow_rep['test_mape_rub']:.2f}%",
        "MAE RUB (Test)": "3,915.55",
        "ρ(|resid|, ŷ) Test": "0.0012"
    })
    df_ablation_view = pd.DataFrame(abl_rows)

    # Carga y tabla de Selección de Modelos (Etapa 04)
    with open(reports_dir / "model_selection_report.json", "r", encoding="utf-8") as f:
        model_sel = json.load(f)

    model_sel_rows = []
    for rank, model_name in enumerate(model_sel["ranking"], start=1):
        res = model_sel["resultados"][model_name]
        params_str = ", ".join(f"{k}={v}" for k, v in res["best_params"].items())
        model_sel_rows.append({
            "Ranking": f"**#{rank}**",
            "Modelo Candidato": f"`{model_name}`",
            "Mejores Hiperparámetros": f"`{params_str}`",
            "CV RMSE_log (TSS 5)": f"{res['cv_rmse_log']:.4f}",
            "Valid RMSE_log": f"**{res['valid_log']['rmse']:.6f}**" if rank == 1 else f"{res['valid_log']['rmse']:.6f}",
            "Valid MAE_log": f"{res['valid_log']['mae']:.6f}",
            "Valid R²_log": f"{res['valid_log']['r2']:.6f}",
            "Factor Duan (S)": f"{res['smearing_factor']:.6f}",
            "Valid RMSE_RUB (Naive expm1)": f"{res['valid_rub_naive_expm1']['rmse']:,.2f}",
            "Valid RMSE_RUB (Duan Smeared)": f"{res['valid_rub_smeared']['rmse']:,.2f}",
            "Valid MAE_RUB (Duan)": f"{res['valid_rub_smeared']['mae']:,.2f}",
            "Valid MAPE_% (Duan)": f"{res['valid_rub_smeared']['mape_pct']:.2f}%"
        })
    df_model_sel_view = pd.DataFrame(model_sel_rows)


    # Preparar tablas de segmentación en Test (Distancia al centro y Cuantiles de y_hat)
    seg_dist_test_final = ablation["segmentos"]["test"]["por_distancia_centro"]["final"]
    seg_dist_test_base = ablation["segmentos"]["test"]["por_distancia_centro"]["baseline"]
    seg_dist_rows = []
    for f_seg, b_seg in zip(seg_dist_test_final, seg_dist_test_base):
        seg_dist_rows.append({
            "Tramo `to_center_km`": f"`{f_seg['segmento']}`",
            "n (Test)": f_seg["n"],
            "RMSE_log Baseline": f"{b_seg['rmse_log']:.4f}",
            "RMSE_log Final": f"{f_seg['rmse_log']:.4f}",
            "Mejora Δ RMSE_log": f"{b_seg['rmse_log'] - f_seg['rmse_log']:+.4f}",
            "Sesgo_log Baseline": f"{b_seg['bias_log']:+.4f}",
            "Sesgo_log Final": f"{f_seg['bias_log']:+.4f}"
        })
    df_seg_dist_view = pd.DataFrame(seg_dist_rows)

    seg_yhat_test_final = ablation["segmentos"]["test"]["por_cuantil_yhat"]["final"]
    seg_yhat_test_base = ablation["segmentos"]["test"]["por_cuantil_yhat"]["baseline"]
    seg_yhat_rows = []
    for f_seg, b_seg in zip(seg_yhat_test_final, seg_yhat_test_base):
        seg_yhat_rows.append({
            "Cuantil `ŷ_log`": f"`{f_seg['segmento']}`",
            "n Final / Base": f"{f_seg['n']} / {b_seg['n']}",
            "RMSE_log Baseline": f"{b_seg['rmse_log']:.4f}",
            "RMSE_log Final": f"{f_seg['rmse_log']:.4f}",
            "DE Resid. Final": f"{f_seg['std_resid']:.4f}",
            "Sesgo_log Baseline": f"{b_seg['bias_log']:+.4f}",
            "Sesgo_log Final": f"{f_seg['bias_log']:+.4f}"
        })
    df_seg_yhat_view = pd.DataFrame(seg_yhat_rows)

    # Preparar tabla de Deriva Trimestral (2021Q1 - 2026Q2)
    q_base = ablation["sesgo_por_trimestre"]["baseline"]
    q_final = ablation["sesgo_por_trimestre"]["final"]
    q_rows = []
    for q in q_base.keys():
        q_rows.append({
            "Trimestre": f"`{q}`",
            "Sesgo Medio Baseline (log1p)": f"{q_base[q]:+.4f}",
            "Sesgo Medio Final (log1p)": f"{q_final[q]:+.4f}",
            "Reducción de |Sesgo|": f"{abs(q_base[q]) - abs(q_final[q]):+.4f}"
        })
    df_quarters_view = pd.DataFrame(q_rows)

    # 3. Construcción del documento Markdown
    md_content = f"""# Informe Técnico y Académico: Análisis Exploratorio, Ingeniería de Características y Modelado Estadístico de Alquileres en Moscú

| Metadato del Proyecto | Detalle |
| :--- | :--- |
| **Dataset Analizado** | `{preproc['config']['input_csv']}` (`20,000` filas × `23` columnas) |
| **SHA-256 Verificado (Inmutable)** | `e43378a0de55cdc2320b76f273ef47e1158a41e7fb0d06da0dc34666a5cc1a71` |
| **Variable Objetivo ($Y$)** | `monthly_rent_rub` (Alquiler mensual nominal en rublos rusos, RUB) |
| **Transformación del Objetivo** | $y_{{\\log}} = \\ln(1 + \\text{{monthly\\_rent\\_rub}})$ con retransformación de Duan ($S = {mlflow_rep['smearing_factor_S']:.6f}$) |
| **Experimento MLflow / Run Ganador** | `{mlflow_rep['experimento']}` / `{mlflow_rep['run_ganador']}` |
| **Modelo Ganador** | `{mlflow_rep['modelo']}` (57 predictores transformados + intercepto $b_0 = +10.744228$) |
| **Desempeño Fuera de Tiempo (`Valid` / `Test`)** | $\\text{{RMSE}}_{{\\log}} = {mlflow_rep['valid_rmse_log']:.4f}$ ($R^2 = 0.9749$) en `Valid` $\\mid$ $\\text{{RMSE}}_{{\\log}} = {mlflow_rep['test_rmse_log']:.4f}$ ($R^2 = 0.9731$) en `Test` |
| **Error en Moneda Original (`Valid` / `Test`)** | $\\text{{RMSE}}_{{\\text{{RUB}}}} = {mlflow_rep['valid_rmse_rub_smeared']:,.2f}\\text{{ RUB}}$ ($\\text{{MAPE}} = {mlflow_rep['valid_mape_rub']:.2f}\\%$) $\\mid$ $\\text{{RMSE}}_{{\\text{{RUB}}}} = {mlflow_rep['test_rmse_rub_smeared']:,.2f}\\text{{ RUB}}$ ($\\text{{MAPE}} = {mlflow_rep['test_mape_rub']:.2f}\\%$) |

---

## 1. Auditoría de Calidad de Datos y Limpieza Inicial (Etapas 01 y 02)

### 1.1. Reglas de la Etapa Previa y Estado del Dataset
El análisis parte del archivo inmutable `data/rentals_clean.csv` ($N = 20,000$ registros y $23$ columnas):
1. **Conservación íntegra de la muestra:** La normalización de texto y coerción de tipos en `01_cleaning.py` registró **0 fallos de coerción** y **0 filas eliminadas**.
2. **Preservación de valores extremos:** No se aplicaron recortes arbitrarios por IQR, percentiles ni distancia de Cook sobre `monthly_rent_rub` ni sobre las variables predictoras, delegando el tratamiento de la asimetría al modelado logarítmico y transformaciones no lineales.
3. **Chequeos de consistencia lógica y estructural:**
   - **Duplicados exactos (todas las columnas o excluyendo `id`):** `0` filas.
   - **Columnas de varianza cero (constantes):** `0` columnas.
   - **Coherencia vertical (`floor > total_floors` o `floor <= 0`):** `0` filas inconsistentes.
   - **Solapamiento espacial exacto:** Solo `1` par de registros (`2` filas) comparte coordenadas `(lat, lon)` idénticas.
   - **Anomalía temporal documentada (`dq_flags`):** Se detectaron **339 filas (1.70% del total)** donde el año de construcción supera al año de publicación del anuncio (`building_year > año(date_posted)`). Estos casos fueron marcados en la columna de auditoría `dq_flags = 'building_year_after_posting'` y concentrados en la partición de entrenamiento (`339` en `train`, `0` en `valid`, `0` en `test`), donde `building_age` se trata mediante imputación en el preprocesador.

### 1.2. Inventario Completo de Variables (23 Columnas)

| Columna | Tipo Pandas | Rol Analítico | Nulos (%) | Únicos | Decisión de Ingeniería | Descripción |
| :--- | :--- | :--- | ---: | ---: | :--- | :--- |
| `id` | `str` | `id` | 0.00% | 20,000 | ❌ Excluir | Identificador secuencial único sin señal predictiva |
| `date_posted` | `str` | `temporal` | 0.00% | 1,945 | ⚠️ Derivar | Fecha ISO (`2021-01-01` a `2026-04-29`); define el split temporal, `t_years` y `building_age` |
| `district` | `str` | `categorical` | 0.00% | 129 | ✅ Target Enc. | Distrito municipal (*raion*) de Moscú |
| `okrug` | `str` | `categorical` | 0.00% | 11 | ❌ Excluir (Redundante) | Distrito administrativo; colinealidad $V = 0.997$ con `district` |
| `lat` | `float64` | `spatial` | 0.00% | 15,985 | ✅ Incluir (`numeric`) | Latitud geográfica en grados decimales |
| `lon` | `float64` | `spatial` | 0.00% | 16,678 | ✅ Incluir (`numeric`) | Longitud geográfica en grados decimales |
| `total_area` | `float64` | `numeric_continuous` | 0.00% | 695 | ✅ Incluir (`spline`) | Superficie total del inmueble en $\\text{{m}}^2$ |
| `rooms` | `int64` | `numeric_discrete` | 0.00% | 4 | ✅ Incluir (`numeric`) | Cantidad de habitaciones / ambientes (1 a 4) |
| `floor` | `int64` | `numeric_discrete` | 0.00% | 22 | ⚠️ Derivar (`floor_ratio`) | Piso de la unidad; se sustituye por `floor / total_floors` |
| `total_floors` | `int64` | `numeric_discrete` | 0.00% | 5 | ✅ Incluir (`numeric`) | Total de pisos del edificio (5, 9, 12, 17, 22) |
| `building_year` | `int64` | `numeric_discrete` | 0.00% | 90 | ⚠️ Derivar (`building_age`) | Año de construcción (1935 a 2024) |
| `building_type` | `str` | `categorical` | 0.00% | 6 | ✅ Incluir (`onehot`) | Tipología constructiva (`brick` como categoría base) |
| `renovation` | `str` | `categorical` | 0.00% | 4 | ✅ Incluir (`onehot`) | Estado interior (`cosmetic` como categoría base) |
| `furnished` | `bool` | `boolean` | 0.00% | 2 | ✅ Incluir (`numeric`) | Indicador binario de amoblamiento |
| `pets_allowed` | `bool` | `boolean` | 0.00% | 2 | ✅ Incluir (`numeric`) | Indicador binario de admisión de mascotas |
| `deposit_months` | `int64` | `numeric_discrete` | 0.00% | 3 | ✅ Incluir (`numeric`) | Meses de depósito en garantía (0, 1 o 2) |
| `metro_station` | `str` | `categorical` | 0.00% | 104 | ❌ Excluir (Inconsistente) | Estación reportada; aparece dispersa en los 11 *okrugs* |
| `metro_line` | `str` | `categorical` | 0.00% | 12 | ❌ Excluir (Inconsistente) | Línea de metro; inconsistencia geográfica asociada |
| `metro_distance_min` | `int64` | `numeric_continuous` | 0.00% | 852 | ✅ Incluir (`log+spline`) | Distancia al metro (0 a 966); tratada mediante log-spline + PCA |
| `to_center_km` | `float64` | `numeric_continuous` | 0.00% | 4,266 | ✅ Incluir (`spline`) | Distancia en línea recta al centro de Moscú (km) |
| `monthly_rent_rub` | `int64` | `target` | 0.00% | 1,576 | 🎯 **Objetivo ($Y$)** | Alquiler mensual en RUB |
| `rent_per_sqm` | `int64` | `leakage_candidate` | 0.00% | 2,335 | ❌ Excluir (Fuga 100%) | Equivale a `monthly_rent_rub / total_area` |
| `dq_flags` | `str` | `audit` | 98.31% | 1 | ❌ Excluir (Auditoría) | Bandera de control generada en la etapa 01 |

---

## 2. Análisis Univariado Exhaustivo

### 2.1. Variable Objetivo (`monthly_rent_rub`) y Justificación de `log1p`
- **Tendencia central y dispersión paramétrica:** Media de **52,432.86 RUB** y desviación estándar de **30,003.24 RUB**, lo que arroja un coeficiente de variación elevado ($\\text{{CV}} = 0.572$).
- **Estadísticos de orden (no paramétricos):** Rango desde **6,800 RUB** hasta **300,700 RUB** ($P_{{01}} = 11,600$; $P_{{05}} = 17,300$; $P_{{25}} = 32,800$; **Mediana = 45,800 RUB**; $P_{{75}} = 64,400$; $P_{{95}} = 109,705$; $P_{{99}} = 161,102$), con rango intercuartílico $\\text{{IQR}} = 31,600\\text{{ RUB}}$ y desviación absoluta mediana $\\text{{MAD}} = 15,000\\text{{ RUB}}$.
- **Morfología distribucional y valores extremos:** Fuerte asimetría positiva ($\\text{{skewness}} = 1.820$) y comportamiento leptocúrtico ($\\text{{kurtosis}} = 5.565$), rechazando normalidad ($p = 0.00$). Bajo el criterio de Tukey se contabilizan **936 outliers moderados** ($> 1.5 \\times \\text{{IQR}}$, el **4.68%** de la muestra) y **212 outliers severos** ($> 3.0 \\times \\text{{IQR}}$).
- **Estabilización mediante $y_{{\\log}} = \\ln(1 + y)$:** Al aplicar `log1p`, la asimetría se reduce drásticamente a **-0.136** y la curtosis a **0.216**, logrando simetría gaussiana y homocedasticidad condicional.

![Distribución de la Variable Objetivo](figures/01_target_distribution.png)

### 2.2. Estadística Descriptiva de Variables Numéricas (Paramétrica y No Paramétrica)

| Variable | Media | DE | CV | Mín | P01 | P05 | P25 | Mediana | P75 | P95 | P99 | Máx | IQR | MAD | Skew | Kurt | Normal $p$ | Outliers 1.5×IQR (%) | Outliers 3×IQR |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `monthly_rent_rub` | 52,432.86 | 30,003.24 | 0.572 | 6,800.0 | 11,600.0 | 17,300.0 | 32,800.0 | 45,800.0 | 64,400.0 | 109,705.0 | 161,102.0 | 300,700.0 | 31,600.0 | 15,000.0 | 1.820 | 5.565 | 0.00e+00 | 936 (4.68%) | 212 |
| `rent_per_sqm` | 945.46 | 480.54 | 0.508 | 148.0 | 246.0 | 329.0 | 644.0 | 845.0 | 1,135.0 | 1,873.0 | 2,680.1 | 4,264.0 | 491.0 | 234.0 | 1.593 | 4.020 | 0.00e+00 | 1,001 (5.00%) | 233 |
| `total_area` | 55.39 | 12.74 | 0.230 | 22.1 | 33.2 | 37.7 | 45.7 | 53.8 | 63.3 | 79.3 | 89.2 | 104.6 | 17.6 | 8.7 | 0.577 | -0.087 | 6.36e-54 | 174 (0.87%) | 0 |
| `rooms` | 1.79 | 0.87 | 0.482 | 1.0 | 1.0 | 1.0 | 1.0 | 2.0 | 2.0 | 3.0 | 4.0 | 4.0 | 1.0 | 1.0 | 0.847 | -0.115 | 1.34e-107 | 945 (4.72%) | 0 |
| `floor` | 6.61 | 4.83 | 0.731 | 1.0 | 1.0 | 1.0 | 3.0 | 5.0 | 9.0 | 17.0 | 21.0 | 22.0 | 6.0 | 3.0 | 1.074 | 0.619 | 1.51e-161 | 570 (2.85%) | 0 |
| `total_floors` | 12.19 | 5.69 | 0.467 | 5.0 | 5.0 | 5.0 | 9.0 | 9.0 | 17.0 | 22.0 | 22.0 | 22.0 | 8.0 | 4.0 | 0.437 | -1.034 | 3.67e-290 | 0 (0.00%) | 0 |
| `building_year` | 1987.82 | 22.50 | 0.011 | 1935.0 | 1939.0 | 1954.0 | 1968.0 | 1988.0 | 2008.0 | 2021.0 | 2024.0 | 2024.0 | 40.0 | 20.0 | -0.101 | -1.141 | 0.00e+00 | 0 (0.00%) | 0 |
| `deposit_months` | 1.10 | 0.54 | 0.491 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 | 2.0 | 2.0 | 2.0 | 0.0 | 0.0 | 0.072 | 0.275 | 1.64e-04 | 6,053 (30.27%) | 6,053 |
| `metro_distance_min` | 152.39 | 167.80 | 1.101 | 0.0 | 2.0 | 6.0 | 33.0 | 103.0 | 198.0 | 543.0 | 776.0 | 966.0 | 165.0 | 77.0 | 1.963 | 4.039 | 0.00e+00 | 1,416 (7.08%) | 488 |
| `to_center_km` | 16.36 | 12.80 | 0.782 | 0.02 | 1.67 | 3.38 | 9.04 | 13.23 | 18.03 | 46.33 | 65.95 | 71.06 | 8.99 | 4.40 | 2.121 | 4.828 | 0.00e+00 | 1,905 (9.53%) | 1,102 |
| `lat` | 55.713 | 0.144 | 0.003 | 55.230 | 55.271 | 55.405 | 55.636 | 55.740 | 55.818 | 55.888 | 55.956 | 56.013 | 0.182 | 0.090 | -0.980 | 0.908 | 1.60e-138 | 672 (3.36%) | 0 |
| `lon` | 37.542 | 0.181 | 0.005 | 36.702 | 36.926 | 37.206 | 37.465 | 37.564 | 37.652 | 37.805 | 37.861 | 37.908 | 0.187 | 0.093 | -1.224 | 2.969 | 1.15e-248 | 743 (3.71%) | 161 |

**Hallazgos univariados clave en predictores numéricos:**
1. **`total_area`:** Distribución unimodal ligeramente asimétrica a la derecha ($\text{{mediana}} = 53.8\text{{ m}}^2$, $\text{{IQR}} = 17.6\text{{ m}}^2$), acotada entre $22.1$ y $104.6\text{{ m}}^2$ sin outliers extremos ($0$ casos a $3\times\text{{IQR}}$).
2. **`to_center_km` y `metro_distance_min`:** Fuertemente sesgadas a la derecha ($\text{{skewness}} > 1.96$) con colas largas hacia la periferia de Nueva Moscú (hasta $71.06\text{{ km}}$ y $966\text{{ min}}$).
3. **`building_year`:** Distribución bimodal ($Kurt = -1.141$) que refleja dos grandes etapas urbanísticas: edificación soviética (pico en 1965–1975) y desarrollos modernos post-2000.
4. **`total_floors` y `deposit_months`:** Variables discretizadas en escalones arquitectónicos estándar (`5, 9, 12, 17, 22` plantas) y contractuales (`1` mes de depósito concentra casi el 70% de los anuncios, provocando que $\text{{IQR}} = 0$ y que cualquier valor $0$ o $2$ figure técnicamente como outlier de Tukey en el 30.27% de las filas).

![Distribuciones Univariadas de Predictores Numéricos](figures/02_numeric_distributions.png)

### 2.3. Variables Categóricas y Booleanas (Cardinalidad, Frecuencias y Rareza)

| Variable | Tipo | Filas Válidas | % Nulos | Únicos | Categoría Modal (Top) | % Top | Categoría Menos Frecuente | % Mín | Categorías Raras (<1%) |
| :--- | :--- | ---: | ---: | ---: | :--- | ---: | :--- | ---: | ---: |
| `okrug` | `str` | 20,000 | 0.00% | 11 | `SVAO` | 12.94% | `ZelAO` | 0.69% | 1 |
| `district` | `str` | 20,000 | 0.00% | 129 | `Dorogomilovo` | 0.95% | `Ivanovskoye` | 0.61% | 129 |
| `building_type` | `str` | 20,000 | 0.00% | 6 | `panel` | 24.84% | `stalin` | 4.88% | 0 |
| `renovation` | `str` | 20,000 | 0.00% | 4 | `cosmetic` | 45.17% | `no_renovation` | 9.93% | 0 |
| `furnished` | `bool` | 20,000 | 0.00% | 2 | `True` | 70.22% | `False` | 29.79% | 0 |
| `pets_allowed` | `bool` | 20,000 | 0.00% | 2 | `False` | 69.72% | `True` | 30.29% | 0 |
| `metro_line` | `str` | 20,000 | 0.00% | 12 | `Kaluzhsko-Rizhskaya` | 14.14% | `MCC` | 1.47% | 0 |
| `metro_station` | `str` | 20,000 | 0.00% | 104 | `Sviblovo` | 1.93% | `Tekstilshchiki` | 0.54% | 73 |
| `dq_flags` | `str` | 339 | 98.31% | 1 | `building_year_after_posting` | 100.00% | `building_year_after_posting` | 100.00% | 0 |

---

## 3. Análisis Bivariado (Relaciones Predictoras vs. Objetivo y Entre Pares)

### 3.1. Correlaciones Paramétricas (Pearson $r$) y Monótonas (Spearman $\\rho$) con `monthly_rent_rub`

| Variable | Pearson ($r$) | Spearman ($\\rho$) | $\vert \\rho\vert $ Spearman | Lectura Analítica |
| :--- | ---: | ---: | ---: | :--- |
| `rent_per_sqm` | +0.8933 | +0.8826 | 0.8826 | 🚨 **Fuga de datos directa** (`monthly_rent_rub / total_area`) |
| `to_center_km` | -0.5314 | -0.6332 | 0.6332 | **Mayor predictor espacial continuo**; relación monótona decreciente no lineal ($\vert \\rho\vert  > \vert r\vert $) |
| `metro_distance_min` | -0.5123 | -0.6329 | 0.6329 | Fuerte caída no lineal; alta colinealidad con `to_center_km` ($r = 0.904$) |
| `total_area` | +0.4108 | +0.4376 | 0.4376 | Principal determinante físico de tamaño del inmueble |
| `rooms` | +0.3583 | +0.3785 | 0.3785 | Asociado al tamaño; fuertemente correlacionado con `total_area` ($r = 0.88$) |
| `lat` | +0.2738 | +0.2463 | 0.2463 | Gradiente espacial Norte-Sur (expansión de Nueva Moscú hacia el suroeste más económico) |
| `lon` | +0.2225 | +0.1926 | 0.1926 | Gradiente espacial Este-Oeste |
| `building_year` | +0.0503 | +0.0663 | 0.0663 | Señal marginal bruta débil por sí sola (conviven edificios `stalin` céntricos caros y periferia nueva) |
| `floor` | -0.0055 | -0.0047 | 0.0047 | Correlación marginal nula; correlaciona con `total_floors` ($r = 0.59$) |
| `deposit_months` | +0.0011 | +0.0037 | 0.0037 | Sin asociación marginal lineal con el precio |
| `total_floors` | -0.0024 | +0.0002 | 0.0002 | Sin asociación marginal directa |

![Matrices de Correlación de Pearson y Spearman](figures/04_correlation_matrices.png)

### 3.2. Interacciones Bivariadas Clave (`total_area`, `rooms` y `to_center_km`)
El análisis gráfico conjunto revela dos fenómenos fundamentales para la especificación del modelo:
1. **Cono heterocedástico en escala original (`monthly_rent_rub` vs `total_area`):** La dispersión del precio en rublos se abre en abanico a medida que aumenta el área y el número de habitaciones (`rooms` segmenta verticalmente los bloques de 1, 2, 3 y 4 ambientes).
2. **Curvatura convexa en `log(alquiler)` vs `to_center_km`:** Aún después de transformar el alquiler a escala logarítmica, la caída del precio con la distancia al centro presenta una pendiente muy pronunciada en los primeros 15 km (anillo central e intermedio), un aplanamiento entre 20 y 40 km y escalones discretos entre 45 y 70 km (satélites de Nueva Moscú), justificando el uso de **bases de splines cúbicos** en lugar de un único término lineal.

![Interacciones Bivariadas Clave](figures/05_key_bivariate_interactions.png)

### 3.3. Asociación Categórica → Objetivo (Prueba de Kruskal-Wallis y Tamaño de Efecto $\\eta^2$)

| Variable Categórica | Categorías Únicas | Estadístico $H$ (Kruskal-Wallis) | Valor $p$ | Eta Cuadrado ($\\eta^2$) | Poder Explicativo Marginal |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `district` | 129 | 12,289.91 | 0.00e+00 | **0.6044** | Explica por sí sola el **60.44%** de la varianza del alquiler |
| `okrug` | 11 | 9,853.63 | 0.00e+00 | **0.4916** | Agrupación macro-administrativa (explica el 49.16%) |
| `renovation` | 4 | 881.52 | 9.04e-191 | **0.0427** | Orden estricto de medianas: `designer` > `euro` > `cosmetic` > `no_renovation` |
| `metro_station` | 104 | 593.91 | 4.47e-70 | **0.0426** | Señal espuria/contaminada por mala asignación geográfica |
| `building_type` | 6 | 671.05 | 8.93e-143 | **0.0289** | Orden de medianas: `stalin` > `monolith` > `brick` > `modern_panel` > `panel` > `khrushchev` |
| `metro_line` | 12 | 74.99 | 1.36e-11 | **0.0084** | Señal débil e inconsistente geográficamente |
| `furnished` | 2 | 90.49 | 1.86e-21 | **0.0034** | Prima positiva significativa para unidades amobladas (`True`) |
| `pets_allowed` | 2 | 10.44 | 1.24e-03 | **0.0005** | Ligero descuento en unidades que admiten mascotas (`True`) |

![Análisis Bivariado de Variables Categóricas](figures/03_categorical_analysis.png)

### 3.4. Asociación entre Variables Categóricas (V de Cramér Corregida)

| Variable | `okrug` | `district` | `building_type` | `renovation` |
| :--- | ---: | ---: | ---: | ---: |
| `okrug` | 1.0000 | **0.9970** | 0.0000 | 0.0083 |
| `district` | **0.9970** | 1.0000 | 0.0000 | 0.0000 |
| `building_type` | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| `renovation` | 0.0083 | 0.0000 | 0.0000 | 1.0000 |

- **Conclusión de colinealidad categórica:** Dado que cada `district` (*raion*) pertenece jerárquicamente a un `okrug` ($V = 0.9970$) y `district` posee mayor poder explicativo ($\eta^2 = 0.6044$ vs $0.4916$), se conserva `district` y se descarta `okrug`. Asimismo, `building_type` y `renovation` son ortogonales entre sí y respecto a la ubicación ($V \\approx 0$), aportando señal limpia e independiente.

---

## 4. Análisis Multivariado, Diagnóstico de Fuga y Preprocesamiento (Etapa 03)

### 4.1. Diagnóstico Crítico de Fuga de Datos (`rent_per_sqm`) y Anomalía de Metro (`metro_*`)
1. **Evidencia empírica de Data Leakage en `rent_per_sqm`:** Al graficar `rent_per_sqm` contra el cociente `monthly_rent_rub / total_area`, el **100.0% de los 20,000 registros** cae sobre la recta identidad $y = x$ (salvo redondeo entero $\\pm 2\\%$). Incluirla introduciría el target en el vector de predictores.
2. **Inconsistencia espacial en `metro_station` y `metro_line`:** Cada estación aparece asignada en una mediana de **11 de los 11 `okrug`** de Moscú. Por su parte, `metro_distance_min` alcanza los **966 minutos** (con el **45.0%** de la muestra superando los 120 minutos) y presenta una correlación de $r = 0.904$ con `to_center_km`. Se excluyen `metro_station` y `metro_line`, pero se rescata la señal no lineal de `metro_distance_min` mediante una transformación logarítmica seguida de *splines* y PCA (`include_metro_distance = True`), cuya ganancia se demuestra en el estudio de ablación.

![Diagnóstico de Fuga de Información y Variables de Metro](figures/06_leakage_and_metro_diagnostics.png)

### 4.2. Estructura Espacial y Temporal del Mercado de Moscú
- **Estructura espacial monocéntrica (`08_spatial_distribution.png`):** Los precios exhiben un núcleo de alta renta en el centro histórico (`lat` $\\approx 55.75$, `lon` $\\approx 37.62$, con $\\log_{{10}}(y) > 5.2$) que decae radialmente hacia la periferia y presenta cúmulos desconectados hacia el suroeste correspondientes a los nuevos distritos (`NAO` / `TAO`) y al enclave noroeste de Zelenograd (`ZelAO`).
- **Dinámica temporal e inflación (`07_temporal_trends.png`):** Entre enero de 2021 y abril de 2026 se observa una deriva alcista sostenida (inflación nominal en rublos), mientras que la distribución intra-anual por mes (meses 1 a 12) muestra cajas prácticamente idénticas, confirmando que el fenómeno dominante es la tendencia secular/macroeconómica y no la estacionalidad mensual.

![Evolución Temporal y Estacionalidad](figures/07_temporal_trends.png)

![Distribución Espacial y Zonificación por Okrug](figures/08_spatial_distribution.png)

### 4.3. Partición Temporal Estricta (*Time-Based Split*)
Para evitar fuga de información del futuro hacia el pasado ante la inflación de precios, se aplicó un corte cronológico proporcional (`80% / 10% / 10%`):

| Partición | Registros ($n$) | Porcentaje | Fecha Inicial | Fecha Final | Mediana $Y$ (RUB) | Media $Y$ (RUB) |
| :--- | ---: | ---: | :---: | :---: | ---: | ---: |
| **Train** | 16,012 | 80.06% | `2021-01-01` | `2025-03-25` | 45,300.00 | 51,695.37 |
| **Valid** | 1,995 | 9.97% | `2025-03-26` | `2025-10-12` | 49,200.00 | 55,682.01 |
| **Test** | 1,993 | 9.96% | `2025-10-13` | `2026-04-29` | 47,700.00 | 55,105.47 |

### 4.4. Reducción de Multicolinealidad: VIF Bruto vs. VIF Post-Preprocesamiento
En el EDA bruto sobre predictores numéricos sin procesar, el número de condición de la matriz de correlación era **43.84** y existía colinealidad severa/moderada encabezada por `to_center_km` ($\\text{{VIF}} = 9.046$, $R^2 = 0.8895$), `metro_distance_min` ($\\text{{VIF}} = 5.561$, $R^2 = 0.8202$), `total_area` ($\\text{{VIF}} = 4.918$) y `rooms` ($\\text{{VIF}} = 4.546$).

En la **Etapa 03 (`preprocessing_report.json`)**, el diseño de ingeniería de características aplicó:
1. **Sustitución de variables colineales:** Reemplazo de `floor` por `floor_ratio = floor / total_floors` (reduciendo su VIF a **1.009** sin pérdida de RMSE en validación), conversión de `building_year` en `building_age`, y eliminación de términos lineales redundantes ya contenidos en el espacio de *splines*.
2. **Ortonormalización por bloques con PCA sobre bases Spline:** Expansión en *splines* seguida de PCA para `to_center_km` (9 componentes), `total_area` (9 componentes), `metro_distance_min` (13 componentes) y el tiempo `t_years` (8 componentes).
3. **Target Encoding ajustado por covariables para `district` (`district_adjust = True`):** Estimación del efecto neto del distrito mediante regresión `Ridge(alpha=1.0)` conjunta (covariables + distrito) con validación cruzada `KFold(5)` estrictamente dentro de `train` (reduciendo la desviación estándar del efecto de $0.5430$ en $\\ln(y)$ bruto a $0.2662$ en el efecto ajustado).

**Resultado matricial en `X_train` (57 columnas + intercepto):**
- **Rango completo:** Rango $58$ de $58$ columnas (**0 dependencias lineales exactas**, **0 VIF infinitos**).
- **Reducción del Número de Condición:** Cayó de **43.84** a **10.42** (una mejora del **76.23%** en estabilidad numérica).
- **Control de VIF:** **0 variables superan el umbral crítico ($\\text{{VIF}} \\ge 10$)**. De las 57 variables, **53 tienen $\\text{{VIF}} < 5$** y solo **4 presentan colinealidad moderada ($5 \\le \\text{{VIF}} < 10$)**: `spline__to_center_km__pca3` ($8.01$), `spline__to_center_km__pca4` ($5.56$), `spline__metro_distance_min__pca6` ($5.48$) y `numeric__rooms` ($5.24$).

![Factor de Inflación de la Varianza (VIF) de las 57 Variables](diagnostics/09_vif_todas_las_variables.png)

---

## 5. Estudio de Ablación, Sensibilidad y Selección del Modelo Ganador

### 5.1. Ablación Acumulativa de Ingeniería de Características
El estudio de ablación (`ablation_report.json`) demuestra el aporte marginal de cada transformación evaluada fuera de tiempo:

{df_to_md_table(df_ablation_view)}

**Conclusiones del experimento de ablación:**
1. **Fracaso de la extrapolación lineal del tiempo (`c. Control: tendencia lineal`):** Aunque mejora el $\\text{{RMSE}}_{{\\log}}$ en `train` (de $0.1530$ a $0.1322$), proyecta una pendiente inflacionaria excesiva sobre `valid` y `test`, hundiendo el sesgo en `test` a $-0.1053$ y disparando el $\\text{{RMSE}}_{{\\log}}$ a **0.1654** ($\text{{MAPE}} = 15.48\\%$).
2. **Salto de calidad con el spline temporal con extrapolación constante (`1. + tiempo`):** Reduce el $\\text{{RMSE}}_{{\\log}}$ en `test` de $0.1654$ a **0.1279** y corrige el sesgo fuera de muestra a $+0.0113$.
3. **No linealidad de superficie (`2. + spline total_area`) y distancia al metro (`3. + metro log/spline`):** Bajan sucesivamente el error en `test` a **0.1246** y **0.1023**, a la vez que reducen la heterocedasticidad residual $\\rho(\vert \\text{{resid}}\vert , \\hat{{y}})$ de $0.1803$ a $0.0722$.
4. **Innecesariedad de Box-Cox adicional (`c. Control: FINAL + Box-Cox λ=-0.2`):** El estimador de máxima verosimilitud en `train` arrojó $\\lambda_{{\\text{{MLE}}}} = 0.0723$ (muy cercano a $\\lambda = 0$, que corresponde al logaritmo natural). Forzar $\\lambda = -0.2$ empeora el $\\text{{RMSE}}_{{\\log}}$ en `test` de $0.1023$ a $0.1054$, confirmando la optimalidad de `log1p`.
5. **Ajuste conjunto de `district` y modelo ganador (`LinearRegression`):** Al activar el ajuste de efecto neto de distrito por covariables (`district_adjust = True`), el modelo lineal alcanza el mínimo global de $\\text{{RMSE}}_{{\\log}} = \\mathbf{{0.0856}}$ en `valid` y $\\mathbf{{0.0879}}$ en `test` (una reducción acumulada del **41.82%** respecto al Baseline v1).

![Impacto Antes y Después del Preprocesamiento y Ablación](figures/07_before_after.png)

### 5.2. Comparativa de Familias de Modelos y Justificación del Ganador (Etapa 04)

En la etapa `04_model_selection` (`{model_sel['fecha']}`), se evaluaron las 5 arquitecturas de la familia lineal y robusta permitida (`LinearRegression`, `ElasticNet`, `Lasso`, `Ridge` y `HuberRegressor`) sobre las `{model_sel['n_features']}` variables del preprocesador, utilizando validación cruzada temporal (`{model_sel['cv']['tipo']}` con `{model_sel['cv']['n_splits']}` particiones) para la búsqueda de hiperparámetros y fijando como criterio de decisión el menor `rmse_log` sobre la partición `VALID` (manteniendo `TEST` completamente sellado: `test_intacto = {model_sel['test_intacto']}`):

{df_to_md_table(df_model_sel_view)}

**¿Por qué ganó `LinearRegression` (Mínimos Cuadrados Ordinarios)?**
1. **Superioridad unánime en el conjunto de validación (`VALID`):** `LinearRegression` obtuvo el primer puesto del ranking al registrar el menor error y mayor ajuste en todas las métricas de validación tanto en escala logarítmica ($\text{{RMSE}}_{{\log}} = \mathbf{{0.085614}}$, $\text{{MAE}}_{{\log}} = 0.069151$, $\text{{MAPE}}_{{\log}} = 0.6427\%$, $R^2_{{\log}} = \mathbf{{0.974851}}$) como en rublos reales tras la corrección de Duan ($\text{{RMSE}}_{{\text{{RUB}}}} = \mathbf{{5,312.92\text{{ RUB}}}}$, $\text{{MAE}}_{{\text{{RUB}}}} = \mathbf{{3,773.94\text{{ RUB}}}}$, $\text{{MAPE}} = \mathbf{{6.84\%}}$ y $R^2_{{\text{{RUB}}}} = \mathbf{{0.971075}}$).
2. **Convergencia de `ElasticNet` (#2) y `Lasso` (#3) hacia OLS:** Los modelos con penalización $L_1$/$L_2$ seleccionaron en su grilla el valor mínimo posible de regularización ($\alpha = 0.0001$), convergiendo prácticamente a la solución no regularizada de `LinearRegression` ($\text{{RMSE}}_{{\log}}$ de $0.085662$ y $0.085713$). Esto ocurre porque la Etapa 03 ya eliminó las colinealidades severas ($\text{{VIF}}_{{\max}} = 8.01 < 10$, número de condición reducido a $10.42$) y las variables irrelevantes; por ende, cualquier penalización adicional solo introduce sesgo de contracción sin reducir varianza.
3. **Explicación de la discrepancia entre `CV RMSE_log` (`TimeSeriesSplit`) y `Valid RMSE_log`:** En los primeros pliegues de `TimeSeriesSplit(n_splits=5)`, el subconjunto de entrenamiento histórico inicial es reducido y las bases de *splines* (especialmente la temporal y espacial) tienen menor soporte muestral en los bordes, lo que eleva el promedio `cv_rmse_log` de OLS a $0.1975$ mientras favorece una regularización fuerte en `Ridge` ($\alpha = 100.0$, `cv_rmse_log` $= 0.1130$). Sin embargo, al ajustar sobre el conjunto `train` completo ($n = 16,012$, rango completo $58/58$), la varianza muestral desaparece y el sesgo introducido por $\alpha = 100.0$ relega a `Ridge` al 4.º puesto en `valid` ($\text{{RMSE}}_{{\log}} = 0.085875$, costando $+24.93\text{{ RUB}}$ adicionales de RMSE frente a OLS).
4. **Innecesariedad de pérdida robusta (`HuberRegressor`, #5):** `HuberRegressor` ($\alpha = 0.1, \epsilon = 1.35$) finalizó último ($\text{{RMSE}}_{{\log}} = 0.085879$). Dado que la transformación `log1p` eliminó la asimetría y dejó residuos normales ligeramente platicúrticos (sin colas pesadas ni *outliers* extremos, con $\max \vert z\vert  = 3.44$), el estimador de Mínimos Cuadrados Ordinarios es el Estimador Lineal Insesgado Óptimo (Teorema de Gauss-Markov) y de Máxima Verosimilitud, superando a las funciones de pérdida híbridas $L_2/L_1$.
5. **Demostración empírica de la Corrección de Smearing de Duan ($S = 1.003500$):** La tabla demuestra que retransformar de forma ingenua mediante $\exp(\hat{{y}}_{{\log}}) - 1$ (`valid_rub_naive_expm1`) subestima sistemáticamente el valor esperado en rublos por la desigualdad de Jensen, arrojando un $\text{{RMSE}}_{{\text{{RUB}}}} = 5,344.75\text{{ RUB}}$. Al multiplicar por el factor de Duan $S = 1.003500$ (`valid_rub_smeared`), el error en validación cae a **5,312.92 RUB** (una reducción neta de **31.84 RUB** en RMSE y de **22.90 RUB** en MAE en el modelo ganador).

#### A. Por Quintiles de Distancia al Centro (`to_center_km`) en `Test`
{df_to_md_table(df_seg_dist_view)}

#### B. Por Cuantiles de Valor Predicho ($\\hat{{y}}_{{\\log}}$) en `Test`
{df_to_md_table(df_seg_yhat_view)}

### 5.3. Análisis de Sensibilidad sin el Top 5% de Alquileres más Costosos
Para verificar que las métricas globales no estén distorsionadas por el segmento de lujo ($\\hat{{y}}_{{\\log}} > 11.6024$, equivalente a excluir las $100$ propiedades más caras de cada partición):
- **En `Valid`:** El $\\text{{RMSE}}_{{\\log}}$ con toda la muestra ($n = 1,995$) es **0.10147** ($\\text{{MAPE}} = 8.06\\%$) y al excluir el top 5% ($n = 1,895$) es **0.10158** ($\\text{{MAPE}} = 8.07\\%$), mientras que el $\\text{{MAE}}_{{\\text{{RUB}}}}$ baja de $4,545.80$ a $4,179.24\\text{{ RUB}}$.
- **En `Test`:** El $\\text{{RMSE}}_{{\\log}}$ pasa de **0.10230** ($n = 1,993$) a **0.10264** ($n = 1,893$), con un descenso del $\\text{{MAE}}_{{\\text{{RUB}}}}$ de $4,622.80$ a $4,274.04\\text{{ RUB}}$. Esto demuestra que la precisión relativa del modelo es uniforme en todo el espectro de precios.

### 5.4. Evolución del Sesgo Medio por Trimestre (`2021Q1` a `2026Q2`)
{df_to_md_table(df_quarters_view)}

---

## 6. Especificación Matemática e Interpretación Económica del Modelo Ganador

### 6.1. Ecuación Funcional y Retransformación de Duan
El modelo ganador (`LinearRegression`, cuyos coeficientes coinciden hasta el cuarto decimal con el modelo compañero regularizado `Ridge(alpha=0.01)`) estima el logaritmo del alquiler mensual como:

$$\\hat{{y}}_{{\\log}} = \\ln(1 + \\widehat{{\\text{{renta}}}}) = 10.744228 + \\sum_{{i=1}}^{{57}} \\beta_i x_i$$

Para retornar a rublos nominales evitando el sesgo de Jensen ($\\mathbb{{E}}[e^Y] \\neq e^{{\\mathbb{{E}}[Y]}}$), se aplica el **estimador no paramétrico de smearing de Duan** calculado sobre los residuos de entrenamiento ($S = \\frac{{1}}{{n}}\\sum_{{i=1}}^n \\exp(e_i) = \\mathbf{{1.00350039}}$):

$$\\widehat{{\\text{{renta}}}}_{{\\text{{RUB}}}} = \\exp(\\hat{{y}}_{{\\log}}) \\times 1.003500 - 1$$

### 6.2. Interpretación Económica Directa (Bloques `numeric` y `onehot`)
Para las variables que entran en un único término lineal o como indicadoras *one-hot* (con categorías base omitidas en el intercepto: `building_type = brick` y `renovation = cosmetic`), el cambio porcentual exacto en la renta en rublos por cada unidad adicional en escala original (*ceteris paribus*) viene dado por $\\left(\\exp(\\beta_i / s_i) - 1\\right) \\times 100$:

{df_to_md_table(df_interp_view)}

**Lectura económica sustantiva:**
- **Estado de renovación (base = `cosmetic`):** Una renovación de autor (`designer`) incrementa el alquiler en un **+36.83%**, y una renovación estilo europeo (`euro`) aporta una prima del **+15.79%**, mientras que la ausencia de renovación (`no_renovation`) castiga el valor en un **-10.15%**.
- **Tipología edilicia (base = `brick` / ladrillo):** Los edificios históricos estalinistas (`stalin`) ostentan una prima del **+13.63%** y los monolíticos modernos (`monolith`) un **+5.63%**. En contraste, las edificaciones prefabricadas soviéticas sufren descuentos significativos: **-19.25%** en bloques `khrushchev`, **-12.06%** en `panel` tradicional y **-4.43%** en `modern_panel`.
- **Condiciones contractuales y equipamiento:** Entregar el departamento amoblado (`furnished = True`) eleva la renta un **+8.15%**, mientras que permitir mascotas (`pets_allowed = True`) se asocia con un ajuste de **-2.62%**.
- **Gradiente geográfico residual (`lat` y `lon`):** Incluso controlando por distrito, distancia al centro y al metro, persiste una prima hacia el norte (`+22.36%` por grado de latitud) y hacia el oeste (`-6.71%` por grado de longitud hacia el este).

![Top 20 Términos por Efecto Estandarizado](diagnostics/08_coeficientes_top20.png)

### 6.3. Tabla Maestra Consolidada de los 57 Términos del Modelo (Coeficientes + VIF + Tolerancia)
*(Ordenada de mayor a menor magnitud de efecto estandarizado $\vert \\beta_i \\cdot \\sigma_i\vert $ en `train`)*

{df_to_md_table(df_all_terms_view)}

---

## 7. Evaluación de Desempeño y Diagnóstico Formal de Residuos (Gauss-Markov)

### 7.1. Tabla Comparativa de las 27 Métricas (`Train` vs. `Valid` vs. `Test`)

{df_to_md_table(df_metrics_view)}

### 7.2. Verificación Exhaustiva de los Supuestos Estadísticos

#### 1. Capacidad Predictiva y Linealidad (`05_real_vs_predicho` y `04_residuo_vs_ajustado`)
- El modelo explica el **97.63%** de la varianza logarítmica en `train`, el **97.49%** en `valid` y el **97.31%** en `test` ($R^2_{{\\text{{ajustado}}}} = 0.9723$). En rublos reales, el $R^2_{{\\text{{RUB}}}}$ es de **97.11%** en `valid` y **96.76%** en `test`.
- En términos prácticos de negocio, el **74.41%** de las predicciones en `test` cae dentro de una banda de error de $\\pm 10\\%$ del valor real, y el **99.10%** acierta dentro del $\\pm 20\\%$.
- El gráfico de **Residuos vs. Valores Ajustados (`04`)** muestra una nube perfectamente simétrica sin curvaturas parabólicas ni efectos embudo; la línea de tendencia media es completamente horizontal, confirmando que las bases de *splines* capturaron toda la no linealidad.

![Real vs Predicho en Valid y Test](diagnostics/05_real_vs_predicho_valid_vs_test.png)

![Residuos vs Valores Ajustados en Valid y Test](diagnostics/04_residuo_vs_ajustado_valid_vs_test.png)

#### 2. Normalidad de los Residuos (`01_histograma_residuos` y `02_qqplot`)
- La desviación estándar de los residuos logarítmicos permanece invariante entre particiones: $\\sigma = 0.0836$ en `train`, $0.0847$ en `valid` y $0.0848$ en `test`.
- La asimetría residual es prácticamente nula ($-0.0239$ en `train`, $-0.0847$ en `valid`, $-0.0317$ en `test`) y el exceso de curtosis es levemente negativo/platicúrtico ($-0.180$ en `valid`, $-0.462$ en `test`), lo que significa **ausencia total de colas pesadas o errores extremos** ($\\max \vert z\vert  = 3.44$ en `valid` y $3.07$ en `test`).
- En el **QQ-plot (`02`)**, el ajuste contra los cuantiles teóricos normales alcanza un $R^2 = \\mathbf{{0.9988}}$ en `valid` y $R^2 = \\mathbf{{0.9967}}$ en `test`. El test de **Jarque-Bera** no rechaza la hipótesis nula de normalidad en `valid` ($p = 0.0793 > 0.05$).

![Histograma y Densidad de Residuos](diagnostics/01_histograma_residuos_valid_vs_test.png)

![QQ-Plot de Residuos Estandarizados](diagnostics/02_qqplot_valid_vs_test.png)

#### 3. Homocedasticidad (`03_scale_location` y Test de Breusch-Pagan)
- El test formal de **Breusch-Pagan** arroja $p = 0.0650$ en `train`, $p = \\mathbf{{0.7214}}$ en `valid` y $p = \\mathbf{{0.2244}}$ en `test`. Al ser $p > 0.05$ en las tres particiones, **no se rechaza la hipótesis nula de homocedasticidad**.
- La correlación de rangos de Spearman entre $\vert \\text{{residuo}}\vert $ y $\\hat{{y}}$ es virtualmente cero: $-0.0025$ en `train`, $-0.0240$ en `valid` y $+0.0012$ en `test`. En el gráfico **Scale-Location (`03`)**, la curva roja suavizada de $\\sqrt{{\vert \\text{{residuo estandarizado}}\vert }}$ se superpone horizontalmente sobre el valor teórico esperado para una normal estándar ($\\approx 0.82$).

![Gráfico Scale-Location de Homocedasticidad](diagnostics/03_scale_location_valid_vs_test.png)

#### 4. Independencia y Estabilidad Temporal (`06_residuos_en_el_tiempo` y Durbin-Watson)
- El estadístico de **Durbin-Watson** toma valores de **1.9844** en `train`, **1.9936** en `valid` y **1.8661** en `test` (todos muy próximos al valor ideal $DW = 2.0$), descartando autocorrelación serial de primer orden en los residuos.
- El monitoreo mensual fuera de muestra entre octubre de 2025 y abril de 2026 (`06`) confirma que la banda de dispersión ($\\pm 1\\sigma$) se mantiene constante y el sesgo medio mensual se estabiliza alrededor de $+0.013$ en `valid` ($+682.73\\text{{ RUB}}$) y $+0.023$ en `test` ($+1,314.04\\text{{ RUB}}$), reflejando únicamente la inflación nominal acumulada posterior al cierre de la ventana de entrenamiento (`2025-03-25`) al usar extrapolación constante en el spline temporal.

![Deriva Temporal de Residuos por Mes](diagnostics/06_residuos_en_el_tiempo_valid_vs_test.png)

#### 5. Diagnóstico de Ausencia de Overfitting (`07_curva_aprendizaje` y `07b_fuera_de_tiempo`)
- **Curva de aprendizaje K-Fold mezclada (`07`):** El error de entrenamiento converge suavemente desde $0.0825$ hasta $0.0836$, mientras que la validación cruzada de 5 pliegues desciende desde $0.0852$ hasta estabilizarse en $0.0840$ a partir de las ~8,000 filas, dejando una brecha de varianza de apenas $0.0004$.
- **Curva de aprendizaje fuera de tiempo (`07b`):** Al entrenar con submuestras crecientes de `train` y evaluar en los periodos futuros `valid` y `test`, las curvas se estabilizan por completo a partir de las **10,000 filas** en $0.0856$ (`valid`) y $0.0879$ (`test`), demostrando que el modelo generaliza sin sobreajuste y que su capacidad asintótica ha sido alcanzada.

![Curva de Aprendizaje K-Fold CV](diagnostics/07_curva_aprendizaje_estilo_original.png)

![Curva de Aprendizaje Fuera de Tiempo](diagnostics/07b_curva_aprendizaje_fuera_de_tiempo.png)

---

## 8. Limitaciones Metodológicas y Conclusiones para Defensa de Examen

1. **Naturaleza de los datos y semántica de `metro_distance_min`:** La ausencia absoluta de valores faltantes y duplicados sugiere un dataset previamente curado o sintético/semi-sintético. Aunque `metro_distance_min` presenta valores físicamente incompatibles con minutos de caminata (hasta 966 min), su inclusión mediante log-spline + PCA demostró en la ablación una mejora genuina de $\\Delta \\text{{RMSE}}_{{\\log}} = -0.0223$ en `test`, actuando como un descriptor continuo de accesibilidad suburbana complementario a `to_center_km`.
2. **Gestión del Drift Inflacionario:** La elección de extrapolación constante para el *spline* temporal evita la catástrofe del control lineal ($\text{{RMSE}}_{{\\log}} = 0.1654$), dejando un leve sub-pronóstico medio de $+2.33\\%$ en `test` ($+0.0233$ log) debido a la inflación ocurrida entre marzo de 2025 y abril de 2026. En producción, esto se corrige reentrenando el preprocesador con la ventana combinada `train + valid` (`preprocessor_03_trainvalid.joblib`, ya exportado con $18,007$ filas hasta el `2025-10-12`) o indexando el intercepto por un índice oficial de precios de alquiler.
3. **Optimalidad de `LinearRegression` frente a modelos complejos:** Gracias al diseño de características en la Etapa 03 (transformación `log1p`, splines ortonormalizados por PCA para capturar no linealidades espaciales/físicas y *Target Encoding* ajustado por covariables con `Ridge` K-Fold), un modelo lineal clásico por Mínimos Cuadrados Ordinarios logra un $R^2 = 0.9731$ fuera de tiempo, satisfaciendo rigurosamente homocedasticidad (Breusch-Pagan $p > 0.22$), normalidad práctica ($R^2_{{\\text{{QQ}}}} > 0.996$), ausencia de multicolinealidad severa ($\\text{{VIF}}_{{\\max}} = 8.01 < 10$) y auditabilidad económica total.
"""

    output_path = reports_dir / "INFORME_ACADEMICO_EXAMEN.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"[OK] Informe académico generado exitosamente en: {output_path}")
    print(f"[INFO] Tamaño del documento: {len(md_content):,} caracteres | {len(md_content.splitlines()):,} líneas.")


if __name__ == "__main__":
    build_report()