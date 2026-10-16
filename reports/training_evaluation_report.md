# Informe de Entrenamiento, Ajuste, Evaluación y Tracking con MLflow — Alquileres en Moscú (Etapa 05)

| Metadato | Valor |
|---|---|
| Fecha de generación | `2026-10-16 19:23:49` |
| Script generador | `src/05_training_evaluation.py` |
| Experimento en MLflow | `moscow_rentals_linear_regression` (ID: `3`) |
| URI de Tracking MLflow | `sqlite:///E:/FACULTAD UTN FRT/5-QUINTO AÑO/introduccion a analisis de datos/1er parcial/mlflow.db` |
| Partición temporal (`Train` / `Val` / `Test`) | `14009` / `2991` / `3000` filas (`155` columnas) |
| Modelo Principal Académico (`LinearRegression`) | `linear_regression_ols__log1p_rub` (`Val RMSE = 10176.42 RUB`, `Test RMSE = 9689.34 RUB`, Run ID: `b9c484bf20f44fc2a471c14bc4e66937`) |
| Mejor Modelo Lineal Global (según `Val RMSE`) | `linear_regression_ols__log1p_rub` (`Val RMSE = 10176.42 RUB`, `Test RMSE = 9689.34 RUB`, Run ID: `b9c484bf20f44fc2a471c14bc4e66937`) |
| Verificación de Inmutabilidad SHA-256 | `VERIFICADA (0 archivos modificados en etapas 01–04)` |

---

## 1. Resumen Ejecutivo

En esta quinta etapa del proyecto se ejecutó el protocolo completo de entrenamiento, ajuste de hiperparámetros sobre partición temporal de validación, auditoría algebraica de coeficientes y evaluación final *holdout* (`Test` 2025-2026) para los 5 modelos candidatos definidos en `models/model_candidates.json`:

1. **Superioridad del régimen logarítmico (`log1p_rub` vs `raw_rub`):** Entrenar sobre `monthly_rent_rub_log1p` ($\ln(1 + y)$) y retransformar a rublos mediante $\exp(\hat{z}) - 1$ mejora sustancialmente la generalización en `Validation` y `Test` frente al ajuste directo en rublos crudos (`raw_rub`), al corregir la asimetría positiva original (`1.821` $\to$ `-0.133`) y atenuar el apalancamiento de los alquileres *premium* de la cola derecha.
2. **Desempeño del Modelo Principal Académico (`LinearRegression`):** En su mejor régimen (`log1p_rub`), `LinearRegression` alcanza en `Validation` un `RMSE` de **`10176.42` RUB** (`MAE = 6829.16` RUB, `R² = 0.8987`) y en `Test` un `RMSE` de **`9689.34` RUB** (`MAE = 6507.81` RUB, `R² = 0.9032`), superando ampliamente a `DummyRegressor`.
3. **Comparación con Alternativas Regularizadas (`Ridge`, `Lasso`, `ElasticNet`):** El mejor modelo global en validación es **`LinearRegression` (`log1p_rub`)** con hiperparámetros `{'fit_intercept': True}`, logrando `Val RMSE = 10176.42 RUB` y `Test RMSE = 9689.34 RUB`. Las diferencias de error entre OLS y las variantes regularizadas son marginales (ver Sección 4), por lo que esta elección no es concluyente solo con estas métricas.

---

## 2. Verificación de Integridad e Inmutabilidad de Archivos de Entrada (SHA-256)

Todos los artefactos provenientes de las etapas 01 a 04 fueron leídos en modo **solo lectura**. El cálculo criptográfico SHA-256 antes y después de la ejecución confirma que no se alteró ningún byte de las entradas ni se repitió el preprocesamiento:

| Archivo de Entrada | SHA-256 Verificado | Estado Post-Ejecución |
|---|---|---|
| `models/model_candidates.json` | `4849d982e0bc6831d9eae0f03537c2022715f6615e4588110a7793394611a369` | `INMUTABLE (IDÉNTICO)` |
| `models/preprocessing/preprocessing_metadata.json` | `7494e23644a66f56c5577759da44d9ce913dce6d9b89d81dde5f635dc3f08983` | `INMUTABLE (IDÉNTICO)` |
| `data/processed/X_train_linear.csv` | `96ca915ea960e0a36debe438018adce5d76415ca501438a69127c27f13a656da` | `INMUTABLE (IDÉNTICO)` |
| `data/processed/X_val_linear.csv` | `d7f03d0ac46986b53aefe3a09ca2f3033fe14d68e396298825b9dc6988d69dea` | `INMUTABLE (IDÉNTICO)` |
| `data/processed/X_test_linear.csv` | `551be1911c5d700d1ff3fa08019bdffa91a15b7c0b391d9d751e839d08ee06e5` | `INMUTABLE (IDÉNTICO)` |
| `data/processed/y_train.csv` | `fbf8eb503bfca159e7967c4c15cc3febdc5a6edc5a8271e2401b6be8f0bc1145` | `INMUTABLE (IDÉNTICO)` |
| `data/processed/y_val.csv` | `3982d8dd5aa092be5f61cb4bd583f24fd2b412835f7a2d410ac4a34b352b35b6` | `INMUTABLE (IDÉNTICO)` |
| `data/processed/y_test.csv` | `c7dfff904f05fec4963d04e160addfbc0d7f8aa07752ff46f14bf32c146c0f70` | `INMUTABLE (IDÉNTICO)` |
| `models/preprocessing/preprocessor_linear.joblib` | `663d3ef160bf4f09d8ee09e89fe34e2d3bd5ea714f2a6d5f0dca4b34c394a976` | `INMUTABLE (IDÉNTICO)` |
| `models/preprocessing/target_log_transformer.joblib` | `a272c046080c7902d2fc6463b763e7eefe22f1d4b7006758527857e48b9a1940` | `INMUTABLE (IDÉNTICO)` |
| `data/rentals_clean.csv` | `e43378a0de55cdc2320b76f273ef47e1158a41e7fb0d06da0dc34666a5cc1a71` | `INMUTABLE (IDÉNTICO)` |
| `reports/eda_report.md` | `991059b28265d495ca45b721ad67ddb0679d98c3773772b26cbdb300c2e7b704` | `INMUTABLE (IDÉNTICO)` |
| `reports/preprocessing_report.md` | `0909b221af3658640233023ca10dde892324d8df77dbfb812478e86fdf83f590` | `INMUTABLE (IDÉNTICO)` |
| `reports/model_selection_report.md` | `2ab3ad553f0a99ad2b8e0ff07f01dc7d3133cba27849d71576e06360af300171` | `INMUTABLE (IDÉNTICO)` |
| `data/processed/X_train_tree.csv` | `6c203b47cfbb2a1d3a6170b8a1c44a6339c3a5779bf26952344ac5207de243b1` | `INMUTABLE (IDÉNTICO)` |
| `data/processed/X_val_tree.csv` | `7dc43bbde177fa9a258701a4df6ad5251ee7a62eaef76e35faaf4b1b192c5879` | `INMUTABLE (IDÉNTICO)` |
| `data/processed/X_test_tree.csv` | `ec5bbea5af84c0d04277d803c5b087d0c0ab2feb3982fcbfc2994ec4b54b8aef` | `INMUTABLE (IDÉNTICO)` |
| `models/preprocessing/preprocessor_tree.joblib` | `331b2ce1f850af5a65b3ae66ed5edeecad883536b19756dcf87806353a6020da` | `INMUTABLE (IDÉNTICO)` |

---

## 3. Protocolo Experimental Sin Fuga Temporal (Zero Data Leakage) y Ajuste de Hiperparámetros

Siguiendo el diseño temporal estricto de la Etapa 03:
- **Ajuste (`.fit()`):** Ejecutado exclusivamente sobre `X_train_linear.csv` (`14009` registros entre `2021-01-01` y `2024-09-06`).
- **Selección de hiperparámetros (`search_grid`):** Evaluada exclusivamente sobre `X_val_linear.csv` (`2991` registros entre `2024-09-07` y `2025-06-29`) minimizando `val_rmse_rub`.
- **Evaluación final (*Holdout*):** `X_test_linear.csv` (`3000` registros entre `2025-06-30` y `2026-04-29`) permaneció completamente aislado durante la búsqueda y solo se evaluó una vez fijados los modelos ganadores.

### 3.1. Mejores Configuraciones en la Búsqueda de Hiperparámetros (Top 12 sobre Validación)

| Modelo | Modo Target | Hiperparámetros Evaluados | Train RMSE (RUB) | Val RMSE (RUB) | Val MAE (RUB) | Val R² | Norma L2 $\Vert\hat{\beta}\Vert_2$ |
|---|---|---|---:|---:|---:|---:|---:|
| `LinearRegression` | `log1p_rub` | `{"fit_intercept": true}` | `8682.34` | **`10176.42`** | `6829.16` | `0.8987` | `5.3398e+00` |
| `Ridge` | `log1p_rub` | `{"alpha": 0.01, "fit_intercept": true, "random_state": 42}` | `8682.51` | **`10176.93`** | `6829.07` | `0.8986` | `5.2536e+00` |
| `Ridge` | `log1p_rub` | `{"alpha": 0.1, "fit_intercept": true, "random_state": 42}` | `8687.43` | **`10183.92`** | `6830.52` | `0.8985` | `4.6932e+00` |
| `ElasticNet` | `log1p_rub` | `{"alpha": 1e-05, "fit_intercept": true, "l1_ratio": 0.1, "max_iter": 10000, "random_state": 42}` | `8691.03` | **`10187.62`** | `6831.97` | `0.8984` | `4.4985e+00` |
| `ElasticNet` | `log1p_rub` | `{"alpha": 1e-05, "fit_intercept": true, "l1_ratio": 0.3, "max_iter": 10000, "random_state": 42}` | `8692.41` | **`10188.39`** | `6832.33` | `0.8984` | `4.4414e+00` |
| `ElasticNet` | `log1p_rub` | `{"alpha": 1e-05, "fit_intercept": true, "l1_ratio": 0.5, "max_iter": 10000, "random_state": 42}` | `8694.04` | **`10189.55`** | `6832.91` | `0.8984` | `4.3791e+00` |
| `ElasticNet` | `log1p_rub` | `{"alpha": 1e-05, "fit_intercept": true, "l1_ratio": 0.7, "max_iter": 10000, "random_state": 42}` | `8695.76` | **`10190.74`** | `6833.37` | `0.8984` | `4.3239e+00` |
| `ElasticNet` | `log1p_rub` | `{"alpha": 1e-05, "fit_intercept": true, "l1_ratio": 0.9, "max_iter": 10000, "random_state": 42}` | `8697.52` | **`10191.87`** | `6833.43` | `0.8983` | `4.2751e+00` |
| `Lasso` | `log1p_rub` | `{"alpha": 1e-05, "fit_intercept": true, "max_iter": 10000, "random_state": 42}` | `8698.70` | **`10192.98`** | `6833.92` | `0.8983` | `4.2395e+00` |
| `Ridge` | `log1p_rub` | `{"alpha": 1.0, "fit_intercept": true, "random_state": 42}` | `8750.85` | **`10233.41`** | `6861.26` | `0.8975` | `3.4242e+00` |
| `ElasticNet` | `log1p_rub` | `{"alpha": 0.0001, "fit_intercept": true, "l1_ratio": 0.1, "max_iter": 10000, "random_state": 42}` | `8790.34` | **`10261.09`** | `6882.95` | `0.8970` | `3.1362e+00` |
| `ElasticNet` | `log1p_rub` | `{"alpha": 0.0001, "fit_intercept": true, "l1_ratio": 0.3, "max_iter": 10000, "random_state": 42}` | `8827.89` | **`10298.07`** | `6908.75` | `0.8962` | `2.9532e+00` |

---

## 4. Comparativa Integral de Modelos en `Train`, `Validation` y `Test` (Escala RUB)

La siguiente tabla resume las 10 corridas principales registradas en MLflow (5 candidatos $\times$ 2 regímenes de la variable objetivo), ordenadas por desempeño en validación (`Val RMSE`):

| Modelo | Modo Target | Mejores Hiperparámetros | Train RMSE | Val RMSE | Test RMSE | Val MAE | Test MAE | Val MedAE | Test MedAE | Val MAPE | Test MAPE | Val R² | Test R² | MLflow Run ID |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `LinearRegression` | `log1p_rub` | `{"fit_intercept": true}` | `8682.34` | **`10176.42`** | **`9689.34`** | `6829.16` | `6507.81` | `4481.50` | `4196.75` | `11.93%` | `11.41%` | `0.8987` | `0.9032` | `b9c484bf20f44fc2a471c14bc4e66937` |
| `Ridge` | `log1p_rub` | `{"alpha": 0.01, "fit_intercept": true, "random_state": 42}` | `8682.51` | **`10176.93`** | **`9688.94`** | `6829.07` | `6506.99` | `4482.34` | `4197.21` | `11.93%` | `11.41%` | `0.8986` | `0.9032` | `3a1d1c6505b749de97bc83f58552b2b9` |
| `ElasticNet` | `log1p_rub` | `{"alpha": 1e-05, "fit_intercept": true, "l1_ratio": 0.1, "max_iter": 10000, "random_state": 42}` | `8691.03` | **`10187.62`** | **`9690.32`** | `6831.97` | `6509.06` | `4471.60` | `4201.60` | `11.94%` | `11.42%` | `0.8984` | `0.9032` | `2e3c580a0b624cda9a95c914e067f877` |
| `Lasso` | `log1p_rub` | `{"alpha": 1e-05, "fit_intercept": true, "max_iter": 10000, "random_state": 42}` | `8698.70` | **`10192.98`** | **`9694.01`** | `6833.92` | `6517.54` | `4458.51` | `4254.22` | `11.95%` | `11.44%` | `0.8983` | `0.9031` | `7181d778724c4d7f9a3be3be26a6885d` |
| `LinearRegression` | `raw_rub` | `{"fit_intercept": true}` | `10886.78` | **`12633.46`** | **`12163.28`** | `8630.79` | `8340.32` | `6249.46` | `6126.65` | `17.48%` | `17.00%` | `0.8438` | `0.8475` | `69e17014dc384e0aaa78ebd4bc6f6a89` |
| `Lasso` | `raw_rub` | `{"alpha": 1e-05, "fit_intercept": true, "max_iter": 10000, "random_state": 42}` | `10886.78` | **`12633.46`** | **`12163.28`** | `8630.79` | `8340.32` | `6249.41` | `6126.64` | `17.48%` | `17.00%` | `0.8438` | `0.8475` | `4e495b837dbe49d6a507024d19fb6aae` |
| `Ridge` | `raw_rub` | `{"alpha": 0.01, "fit_intercept": true, "random_state": 42}` | `10886.79` | **`12633.96`** | **`12163.46`** | `8631.16` | `8340.35` | `6232.38` | `6121.91` | `17.48%` | `17.00%` | `0.8438` | `0.8475` | `e484a11e35ff44f184d03dfeceddae55` |
| `ElasticNet` | `raw_rub` | `{"alpha": 1e-05, "fit_intercept": true, "l1_ratio": 0.9, "max_iter": 10000, "random_state": 42}` | `10886.80` | **`12634.16`** | **`12163.54`** | `8631.30` | `8340.37` | `6230.51` | `6126.33` | `17.48%` | `17.00%` | `0.8438` | `0.8475` | `919149de627a461b9485058b56d70d84` |
| `DummyRegressor` | `raw_rub` | `{"strategy": "mean"}` | `29203.80` | **`32322.62`** | **`31453.34`** | `22314.54` | `22216.39` | `16662.68` | `16512.68` | `50.13%` | `50.61%` | `-0.0224` | `-0.0196` | `5df02b2319944699ad89309fdf80b00e` |
| `DummyRegressor` | `log1p_rub` | `{"strategy": "median"}` | `29888.89` | **`33854.20`** | **`32944.61`** | `22572.08` | `22279.13` | `15400.00` | `15000.00` | `44.45%` | `44.48%` | `-0.1216` | `-0.1186` | `0b483c946a744cdf8901dd4f99b86ee9` |

### 4.1. Métricas en Escala Logarítmica y Diagnóstico del Factor de Corrección de Duan (`log1p_rub`)

Cuando el modelo se entrena sobre $z = \ln(1 + y)$, la predicción ingenua $\hat{y} = \exp(\hat{z}) - 1$ estima la mediana condicional geométrica en lugar de la media condicional $\mathbb{E}[Y|X]$. El estimador no paramétrico de Duan ($\hat{\alpha}_{\text{Duan}} = \frac{1}{N_{\text{train}}}\sum_{i=1}^{N_{\text{train}}} \exp(e_i)$) corrige dicho sesgo multiplicativo:

| Modelo (`log1p_rub`) | Train RMSE (log) | Val RMSE (log) | Test RMSE (log) | Val R² (log) | Test R² (log) | Factor de Duan ($\hat{\alpha}$) | Val RMSE `expm1` (RUB) | Val RMSE con Duan (RUB) | Test RMSE `expm1` (RUB) | Test RMSE con Duan (RUB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `LinearRegression` | `0.1264` | `0.1411` | `0.1364` | `0.9317` | `0.9359` | `1.008038` | `10176.42` | `10368.35` | `9689.34` | `9854.23` |
| `Ridge` | `0.1264` | `0.1411` | `0.1364` | `0.9317` | `0.9359` | `1.008038` | `10176.93` | `10368.79` | `9688.94` | `9853.75` |
| `ElasticNet` | `0.1265` | `0.1413` | `0.1365` | `0.9316` | `0.9358` | `1.008058` | `10187.62` | `10378.73` | `9690.32` | `9854.52` |
| `Lasso` | `0.1266` | `0.1414` | `0.1367` | `0.9315` | `0.9357` | `1.008079` | `10192.98` | `10383.13` | `9694.01` | `9857.61` |
| `DummyRegressor` | `0.5422` | `0.5458` | `0.5439` | `-0.0215` | `-0.0185` | `1.142339` | `33854.20` | `32322.62` | `32944.61` | `31453.34` |

---

## 5. Auditoría Matemática de Multicolinealidad y Estabilidad de Coeficientes (Defensa Académica)

### 5.1. Diagnóstico Espectral de la Matriz de Diseño `X_train_linear`

- **Dimensiones de `X_train_linear`:** `14009` filas $\times$ `155` columnas.
- **Rango algebraico efectivo (`np.linalg.matrix_rank`):** **`155`** (Deficiencia de rango exacta = **`0` columnas linealmente dependientes**).
- **Número de condición (`np.linalg.cond`):** **`2.9453e+02`** ($\sigma_{\max} = 2.0978e+02$, $\sigma_{\min} = 7.1225e-01$).

**Diagnóstico de rango (`155` columnas, rango `155`):** la matriz tiene rango completo. La etapa 03 eliminó las dependencias exactas con `OneHotEncoder(drop="first")`, retirando `okrug` (suma exacta de sus `district`) y `building_age` (combinación lineal de `building_year` y el tiempo). Persiste casi-colinealidad: `district` explica ~99% de la varianza de `to_center_km` (medido sobre el dataset), por lo que el coeficiente de la distancia no es interpretable por separado de las dummies de distrito.

### 5.2. Comparación Cuantitativa del Vector de Coeficientes $\hat{\beta}$ entre Modelos

| Modelo | Modo Target | Norma L2 $\Vert\hat{\beta}\Vert_2$ | Norma L1 $\Vert\hat{\beta}\Vert_1$ | Máximo Absoluto $\max_j \vert\hat{\beta}_j\vert$ | Coeficientes Nulos (`n_zero` / `%`) | $\Vert\hat{\beta}_{\text{num}}\Vert_2$ (14 cols) | $\Vert\hat{\beta}_{\text{okrug}}\Vert_2$ (0 cols) | $\Vert\hat{\beta}_{\text{district}}\Vert_2$ (128 cols) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `DummyRegressor` | `raw_rub` | `0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `155` / `155` (`100.0%`) | `0.0000e+00` | `0.0000e+00` | `0.0000e+00` |
| `DummyRegressor` | `log1p_rub` | `0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `155` / `155` (`100.0%`) | `0.0000e+00` | `0.0000e+00` | `0.0000e+00` |
| `LinearRegression` | `raw_rub` | `2.5374e+05` | `2.2349e+06` | `7.3895e+04` | `0` / `155` (`0.0%`) | `1.8530e+04` | `0.0000e+00` | `2.5182e+05` |
| `LinearRegression` | `log1p_rub` | `5.3398e+00` | `4.6801e+01` | `1.2334e+00` | `0` / `155` (`0.0%`) | `3.9312e-01` | `0.0000e+00` | `5.3045e+00` |
| `Ridge` | `raw_rub` | `2.5241e+05` | `2.2214e+06` | `7.3850e+04` | `0` / `155` (`0.0%`) | `1.8542e+04` | `0.0000e+00` | `2.5047e+05` |
| `Ridge` | `log1p_rub` | `5.2536e+00` | `4.6228e+01` | `1.1986e+00` | `0` / `155` (`0.0%`) | `3.9311e-01` | `0.0000e+00` | `5.2177e+00` |
| `Lasso` | `raw_rub` | `2.5374e+05` | `2.2349e+06` | `7.3895e+04` | `0` / `155` (`0.0%`) | `1.8530e+04` | `0.0000e+00` | `2.5182e+05` |
| `Lasso` | `log1p_rub` | `4.2395e+00` | `3.9062e+01` | `8.6346e-01` | `0` / `155` (`0.0%`) | `3.9845e-01` | `0.0000e+00` | `4.1945e+00` |
| `ElasticNet` | `raw_rub` | `2.5189e+05` | `2.2162e+06` | `7.3831e+04` | `0` / `155` (`0.0%`) | `1.8547e+04` | `0.0000e+00` | `2.4996e+05` |
| `ElasticNet` | `log1p_rub` | `4.4985e+00` | `4.1385e+01` | `9.1041e-01` | `0` / `155` (`0.0%`) | `3.9665e-01` | `0.0000e+00` | `4.4562e+00` |

**Conclusión de la auditoría numérica:**
- Número de condición `2.95e+02`: OLS es numéricamente estable (rango completo). Norma $\Vert\hat{\beta}\Vert_2$ en `log1p_rub`: OLS = `5.340` vs Ridge = `5.254`.
- Las variantes regularizadas cambian muy poco el error predictivo (ver Sección 4): las diferencias entre OLS, Ridge, Lasso y ElasticNet son marginales y no deben leerse como superioridad de un modelo.
- `Lasso` y `ElasticNet` anulan coeficientes solo si el alpha elegido es lo bastante alto (ver columna `n_zero`); con alphas muy pequeños se comportan casi como OLS.

---

## 6. Interpretación Económica e Inmobiliaria de las Variables Clave

En el régimen `log1p_rub`, cada coeficiente $\hat{\beta}_j$ representa una **semielasticidad**: un incremento de una unidad en la variable preprocesada $X_j$ (escala `RobustScaler` / IQR para numéricas, o cambio $0 \to 1$ para binarias y dummies) produce una variación porcentual aproximada de $(\exp(\hat{\beta}_j) - 1) \times 100\%$ en el alquiler mensual en rublos:

### 6.1. Comparación de Coeficientes en Variables Estructurales, Espaciales y Temporales (`log1p_rub`)

| Variable (`feature`) | `LinearRegression` (OLS) | `Ridge` (L2) | `Lasso` (L1) | `ElasticNet` (L1+L2) |
|---|---:|---:|---:|---:|
| `num__total_area` | `+3.61459e-01` | `+0.36143` | `+0.36053` | `+0.36109` |
| `num__to_center_km` | `-1.52254e-03` | `-0.00436` | `-0.05334` | `-0.03458` |
| `num__posting_trend` | `+1.36963e-01` | `+0.13698` | `+0.13709` | `+0.13709` |
| `num__rooms` | `-6.08015e-02` | `-0.06077` | `-0.05960` | `-0.06033` |
| `num__building_year` | `-2.77426e-03` | `-0.00278` | `-0.00271` | `-0.00283` |
| `num__area_per_room` | `-3.47624e-02` | `-0.03475` | `-0.03406` | `-0.03454` |
| `num__floor_ratio` | `+5.23635e-03` | `+0.00527` | `+0.00490` | `+0.00550` |
| `bin__furnished` | `+8.01739e-02` | `+0.08016` | `+0.07990` | `+0.08001` |
| `cat__renovation_designer` | `+3.13498e-01` | `+0.31352` | `+0.31375` | `+0.31366` |
| `cat__renovation_euro` | `+1.44605e-01` | `+0.14461` | `+0.14464` | `+0.14466` |
| `cat__renovation_no_renovation` | `-1.10248e-01` | `-0.11023` | `-0.10981` | `-0.11002` |

### 6.2. Top 15 Variables de Mayor Impacto Económico (Modelo Regularizado de Referencia en `log1p_rub`)

| Variable (`feature`) | Familia | Coeficiente Regularizado ($\hat{\beta}_j$) | Impacto Porcentual Aprox. $(\exp(\hat{\beta}_j)-1)\%$ | Coeficiente en OLS ($\hat{\beta}_j^{\text{OLS}}$) |
|---|---|---:|---:|---:|
| `cat__district_Novofyodorovskoye` | `ohe_district` | `-1.23344` | `-70.87%` | `-1.23344e+00` |
| `cat__district_Kievsky` | `ohe_district` | `-1.23154` | `-70.82%` | `-1.23154e+00` |
| `cat__district_Rogovskoye` | `ohe_district` | `-1.22850` | `-70.73%` | `-1.22850e+00` |
| `cat__district_Klyonovskoye` | `ohe_district` | `-1.22377` | `-70.59%` | `-1.22377e+00` |
| `cat__district_Mikhaylovo-Yartsevskoye` | `ohe_district` | `-1.16986` | `-68.96%` | `-1.16986e+00` |
| `cat__district_Shchapovskoye` | `ohe_district` | `-1.16322` | `-68.75%` | `-1.16322e+00` |
| `cat__district_Voronovskoye` | `ohe_district` | `-1.15868` | `-68.61%` | `-1.15868e+00` |
| `cat__district_Pervomayskoye` | `ohe_district` | `-1.11651` | `-67.26%` | `-1.11651e+00` |
| `cat__district_Krasnopakhorskoye` | `ohe_district` | `-1.04992` | `-65.00%` | `-1.04992e+00` |
| `cat__district_Troitsk` | `ohe_district` | `-0.87023` | `-58.11%` | `-8.70227e-01` |
| `cat__district_Tverskoy` | `ohe_district` | `+0.86052` | `+136.44%` | `+8.60520e-01` |
| `cat__district_Khamovniki` | `ohe_district` | `+0.83806` | `+131.19%` | `+8.38065e-01` |
| `cat__district_Arbat` | `ohe_district` | `+0.82224` | `+127.56%` | `+8.22240e-01` |
| `cat__district_Marushkinskoye` | `ohe_district` | `-0.78470` | `-54.37%` | `-7.84701e-01` |
| `cat__district_Yakimanka` | `ohe_district` | `+0.78451` | `+119.13%` | `+7.84514e-01` |

**Lectura económica de los resultados:**
1. **Gradiente espacial (`cat__district_*`, `num__to_center_km`):** la señal geográfica está en las dummies de distrito (Sección 6.2). `num__to_center_km` es casi redundante con `district` (~99% de su varianza la explica el distrito), por lo que su coeficiente individual no debe interpretarse por separado ni como "efecto de la distancia".
2. **Escala física del inmueble (`num__total_area`, `num__rooms`, `num__area_per_room`):** La superficie total (`num__total_area`) es el impulsor físico dominante con elasticidad positiva; en los modelos con penalización $L_1$ (`Lasso` / `ElasticNet`), parte de la colinealidad entre `total_area`, `rooms` y `area_per_room` se consolida en `total_area` y `rooms`.
3. **Calidad constructiva e interior (`cat__renovation_*`, `bin__furnished`):** Los acabados `designer` y `euro` aportan una prima sustancial frente a `cosmetic` y `no_renovation`, al igual que la condición de inmueble amoblado (`bin__furnished`).
4. **Tendencia temporal (`num__posting_trend`):** el mercado se estancó desde 2024 (precio mediano por m² plano), por lo que la tendencia se congela en el último valor visto en `Train` y no se extrapola a `Val`/`Test`. Revisar el sesgo mensual en la figura 04.

---

## 7. Diagnóstico Gráfico de Residuos y Estabilidad Temporal (`reports/figures/modeling/`)

| Archivo Generado | Hallazgo Analítico Principal |
|---|---|
| `reports/figures/modeling/01_model_comparison_val_test.png` | Muestra visualmente la caída de `RMSE` y `MAE` y el salto en `R²` al pasar de `DummyRegressor` a los 4 modelos lineales, así como la ganancia sistemática del régimen `log1p_rub` frente a `raw_rub`. |
| `reports/figures/modeling/02_ols_vs_regularized_coefficients.png` | Compara los coeficientes de `LinearRegression` frente a `Ridge`, `Lasso` y `ElasticNet`; con la matriz de rango completo las diferencias entre modelos son pequeñas. |
| `reports/figures/modeling/03_residuals_diagnostics.png` | `raw_rub` muestra heterocedasticidad en abanico y puede predecir alquileres negativos. `log1p_rub` reduce la cola derecha del Q-Q, pero empeora la cola izquierda y, en escala RUB, el abanico persiste: no se demuestra corrección de heterocedasticidad ni normalidad. Para inferencia usar errores robustos (HC3) y un test formal (Breusch-Pagan). |
| `reports/figures/modeling/04_temporal_error_drift_test.png` | Evalúa el error absoluto medio (`MAE`) y el sesgo monetario ($\hat{y} - y$) mes a mes entre junio de 2025 y abril de 2026 (`Test`), permitiendo detectar sesgo sistemático por mes; la tendencia congelada (`num__posting_trend`) evita extrapolar inflación que ya no continúa. |

---

## 8. Conclusión Final y Recomendación Académica

1. **Entrega Académica Principal (`models/trained/primary_linear_regression.joblib`):** Se selecciona **`LinearRegression` entrenado en el régimen `log1p_rub`** como modelo principal para cumplir estrictamente la exigencia académica del proyecto (`Val RMSE = 10176.42 RUB`, `Test RMSE = 9689.34 RUB`, `Test R² = 0.9032`).
2. **Mejor Modelo Lineal Global (`models/trained/best_linear_model.joblib`):** Para interpretación económica directa de los coeficientes individuales $\hat{\beta}_j$ y despliegue robusto ante colinealidad, se respalda con **`LinearRegression` (`log1p_rub`, `{'fit_intercept': True}`)**, que alcanza el menor error en validación (`Val RMSE = 10176.42 RUB`, `Test RMSE = 9689.34 RUB`) con diferencias de error marginales respecto a `LinearRegression`; la elección entre ambos no es concluyente solo con estas métricas.
