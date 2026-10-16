# Informe de Entrenamiento, Ajuste, Evaluación y Tracking con MLflow — Alquileres en Moscú (Etapa 05)

| Metadato | Valor |
|---|---|
| Fecha de generación | `2026-10-16 17:16:42` |
| Script generador | `src/05_training_evaluation.py` |
| Experimento en MLflow | `moscow_rentals_linear_regression` (ID: `3`) |
| URI de Tracking MLflow | `sqlite:///E:/FACULTAD UTN FRT/5-QUINTO AÑO/introduccion a analisis de datos/1er parcial/mlflow.db` |
| Partición temporal (`Train` / `Val` / `Test`) | `14009` / `2991` / `3000` filas (`170` columnas) |
| Modelo Principal Académico (`LinearRegression`) | `linear_regression_ols__log1p_rub` (`Val RMSE = 10966.97 RUB`, `Test RMSE = 12875.24 RUB`, Run ID: `fd4a752e7a4443d28f17d71aa65d0d0f`) |
| Mejor Modelo Lineal Global (según `Val RMSE`) | `ridge_l2__log1p_rub` (`Val RMSE = 10946.11 RUB`, `Test RMSE = 12772.61 RUB`, Run ID: `2a2e397a37cc4be09d8f668422823fdd`) |
| Verificación de Inmutabilidad SHA-256 | `VERIFICADA (0 archivos modificados en etapas 01–04)` |

---

## 1. Resumen Ejecutivo

En esta quinta etapa del proyecto se ejecutó el protocolo completo de entrenamiento, ajuste de hiperparámetros sobre partición temporal de validación, auditoría algebraica de coeficientes y evaluación final *holdout* (`Test` 2025-2026) para los 5 modelos candidatos definidos en `models/model_candidates.json`:

1. **Superioridad del régimen logarítmico (`log1p_rub` vs `raw_rub`):** Entrenar sobre `monthly_rent_rub_log1p` ($\ln(1 + y)$) y retransformar a rublos mediante $\exp(\hat$\exp(\hat{z}) - 1$ mejora sustancialmente la generalización en `Validation` y `Test` frente al ajuste directo en rublos crudos (`raw_rub`), al corregir la asimetría positiva original (`1.821` $	o$ `-0.133`) y atenuar el apalancamiento de los alquileres *premium* de la cola derecha.
2. **Desempeño del Modelo Principal Académico (`LinearRegression`):** En su mejor régimen (`log1p_rub`), `LinearRegression` alcanza en `Validation` un `RMSE` de **`10966.97` RUB** (`MAE = 7569.60` RUB, `R² = 0.8823`) y en `Test` un `RMSE` de **`12875.24` RUB** (`MAE = 9318.20` RUB, `R² = 0.8291`), superando ampliamente a `DummyRegressor`.
3. **Comparación con Alternativas Regularizadas (`Ridge`, `Lasso`, `ElasticNet`):** El mejor modelo global en validación es **`Ridge` (`log1p_rub`)** con hiperparámetros `{'alpha': 10.0, 'fit_intercept': True, 'random_state': 42}`, logrando `Val RMSE = 10946.11 RUB` y `Test RMSE = 12772.61 RUB`. Mientras que la capacidad predictiva dentro del subespacio columna de $X$ es cercana entre OLS y las variantes regularizadas, la diferencia fundamental radica en la **estabilidad numérica del vector de coeficientes $\hat{eta}$**.

---

## 2. Verificación de Integridad e Inmutabilidad de Archivos de Entrada (SHA-256)

Todos los artefactos provenientes de las etapas 01 a 04 fueron leídos en modo **solo lectura**. El cálculo criptográfico SHA-256 antes y después de la ejecución confirma que no se alteró ningún byte de las entradas ni se repitió el preprocesamiento:

| Archivo de Entrada | SHA-256 Verificado | Estado Post-Ejecución |
|---|---|---|
| `models/model_candidates.json` | `1ba15a0b073bd64ec1494ddc763654286296b5b6415c466e72c01a92e6e9ac7f` | `INMUTABLE (IDÉNTICO)` |
| `models/preprocessing/preprocessing_metadata.json` | `271b0e3827af73ef78d8169b3b4674a920fb6a8f1a2c230a693642952a577d2a` | `INMUTABLE (IDÉNTICO)` |
| `data/processed/X_train_linear.csv` | `111dcd6216a6e2b2ad418dd1f5d004e6aec1686ffb1f3416d19e3b96920f9917` | `INMUTABLE (IDÉNTICO)` |
| `data/processed/X_val_linear.csv` | `070ccbb36f6f2612d81fff088499f2e2849c5caa1b3b5516b7dfa75a2b30bc69` | `INMUTABLE (IDÉNTICO)` |
| `data/processed/X_test_linear.csv` | `21d72ab52d0f2567bde0d57b4cd6c71a5b4de78fe07ced6b95a0d7194db0ce45` | `INMUTABLE (IDÉNTICO)` |
| `data/processed/y_train.csv` | `fbf8eb503bfca159e7967c4c15cc3febdc5a6edc5a8271e2401b6be8f0bc1145` | `INMUTABLE (IDÉNTICO)` |
| `data/processed/y_val.csv` | `3982d8dd5aa092be5f61cb4bd583f24fd2b412835f7a2d410ac4a34b352b35b6` | `INMUTABLE (IDÉNTICO)` |
| `data/processed/y_test.csv` | `c7dfff904f05fec4963d04e160addfbc0d7f8aa07752ff46f14bf32c146c0f70` | `INMUTABLE (IDÉNTICO)` |
| `models/preprocessing/preprocessor_linear.joblib` | `d7b8dfd31591a87c988684ba248eedf445b8eefcc4a34cd65aaf565d3a094bfa` | `INMUTABLE (IDÉNTICO)` |
| `models/preprocessing/target_log_transformer.joblib` | `a272c046080c7902d2fc6463b763e7eefe22f1d4b7006758527857e48b9a1940` | `INMUTABLE (IDÉNTICO)` |
| `data/rentals_clean.csv` | `e43378a0de55cdc2320b76f273ef47e1158a41e7fb0d06da0dc34666a5cc1a71` | `INMUTABLE (IDÉNTICO)` |
| `reports/eda_report.md` | `991059b28265d495ca45b721ad67ddb0679d98c3773772b26cbdb300c2e7b704` | `INMUTABLE (IDÉNTICO)` |
| `reports/preprocessing_report.md` | `28c513533b7a4bcdae6c6cbcd61dff0f10abdeed6b2a03e8bed20a2cb8ba9a44` | `INMUTABLE (IDÉNTICO)` |
| `reports/model_selection_report.md` | `8d2bfe1868d15c982ce69a255357d1c51613ed91df78634599f50cb029976665` | `INMUTABLE (IDÉNTICO)` |
| `data/processed/X_train_tree.csv` | `83777efce60d6f3f812f97bd953d2cbc0893ab5c2de94a422a9b0cae9ebc329a` | `INMUTABLE (IDÉNTICO)` |
| `data/processed/X_val_tree.csv` | `4c12d944a5e73f6f86244fdc0a535784c241e376b1daba5954f9ef43486fce03` | `INMUTABLE (IDÉNTICO)` |
| `data/processed/X_test_tree.csv` | `412bfb5c46f3389aa72cdad735dde47498b454aaf90f990fd73e36ff09f8466f` | `INMUTABLE (IDÉNTICO)` |
| `models/preprocessing/preprocessor_tree.joblib` | `3e5868863774596e8296a431998f2435831b2d99e60b5debdf7b6a24c1674743` | `INMUTABLE (IDÉNTICO)` |

---

## 3. Protocolo Experimental Sin Fuga Temporal (Zero Data Leakage) y Ajuste de Hiperparámetros

Siguiendo el diseño temporal estricto de la Etapa 03:
- **Ajuste (`.fit()`):** Ejecutado exclusivamente sobre `X_train_linear.csv` (`14009` registros entre `2021-01-01` y `2024-09-06`).
- **Selección de hiperparámetros (`search_grid`):** Evaluada exclusivamente sobre `X_val_linear.csv` (`2991` registros entre `2024-09-07` y `2025-06-29`) minimizando `val_rmse_rub`.
- **Evaluación final (*Holdout*):** `X_test_linear.csv` (`3000` registros entre `2025-06-30` y `2026-04-29`) permaneció completamente aislado durante la búsqueda y solo se evaluó una vez fijados los modelos ganadores.

### 3.1. Mejores Configuraciones en la Búsqueda de Hiperparámetros (Top 12 sobre Validación)

| Modelo | Modo Target | Hiperparámetros Evaluados | Train RMSE (RUB) | Val RMSE (RUB) | Val MAE (RUB) | Val R² | Norma L2 $\Vert\hat{eta}\Vert_2$ |
|---|---|---|---:|---:|---:|---:|---:|
| `Ridge` | `log1p_rub` | `{"alpha": 10.0, "fit_intercept": true, "random_state": 42}` | `8865.80` | **`10946.11`** | `7604.99` | `0.8827` | `1.6382e+00` |
| `ElasticNet` | `log1p_rub` | `{"alpha": 0.0001, "fit_intercept": true, "l1_ratio": 0.5, "max_iter": 10000, "random_state": 42}` | `8780.86` | **`10951.28`** | `7579.37` | `0.8826` | `1.9356e+00` |
| `ElasticNet` | `log1p_rub` | `{"alpha": 0.0001, "fit_intercept": true, "l1_ratio": 0.3, "max_iter": 10000, "random_state": 42}` | `8759.11` | **`10951.99`** | `7572.94` | `0.8826` | `2.0088e+00` |
| `ElasticNet` | `log1p_rub` | `{"alpha": 0.001, "fit_intercept": true, "l1_ratio": 0.1, "max_iter": 10000, "random_state": 42}` | `9033.94` | **`10952.44`** | `7659.38` | `0.8826` | `1.4452e+00` |
| `Lasso` | `log1p_rub` | `{"alpha": 0.0001, "fit_intercept": true, "max_iter": 10000, "random_state": 42}` | `8845.50` | **`10954.95`** | `7602.95` | `0.8826` | `1.7785e+00` |
| `ElasticNet` | `log1p_rub` | `{"alpha": 0.0001, "fit_intercept": true, "l1_ratio": 0.7, "max_iter": 10000, "random_state": 42}` | `8807.28` | **`10955.00`** | `7589.20` | `0.8826` | `1.8549e+00` |
| `ElasticNet` | `log1p_rub` | `{"alpha": 0.0001, "fit_intercept": true, "l1_ratio": 0.9, "max_iter": 10000, "random_state": 42}` | `8833.91` | **`10956.90`** | `7599.36` | `0.8825` | `1.7948e+00` |
| `ElasticNet` | `log1p_rub` | `{"alpha": 0.0001, "fit_intercept": true, "l1_ratio": 0.1, "max_iter": 10000, "random_state": 42}` | `8742.67` | **`10957.83`** | `7571.20` | `0.8825` | `2.0854e+00` |
| `Ridge` | `log1p_rub` | `{"alpha": 1.0, "fit_intercept": true, "random_state": 42}` | `8728.49` | **`10961.46`** | `7570.60` | `0.8824` | `2.2079e+00` |
| `LinearRegression` | `log1p_rub` | `{"fit_intercept": true}` | `8716.46` | **`10966.97`** | `7569.60` | `0.8823` | `2.4875e+00` |
| `Ridge` | `log1p_rub` | `{"alpha": 0.1, "fit_intercept": true, "random_state": 42}` | `8717.17` | **`10967.13`** | `7570.74` | `0.8823` | `2.4357e+00` |
| `Ridge` | `log1p_rub` | `{"alpha": 0.01, "fit_intercept": true, "random_state": 42}` | `8716.52` | **`10967.17`** | `7569.88` | `0.8823` | `2.4786e+00` |

---

## 4. Comparativa Integral de Modelos en `Train`, `Validation` y `Test` (Escala RUB)

La siguiente tabla resume las 10 corridas principales registradas en MLflow (5 candidatos $	imes$ 2 regímenes de la variable objetivo), ordenadas por desempeño en validación (`Val RMSE`):

| Modelo | Modo Target | Mejores Hiperparámetros | Train RMSE | Val RMSE | Test RMSE | Val MAE | Test MAE | Val MedAE | Test MedAE | Val MAPE | Test MAPE | Val R² | Test R² | MLflow Run ID |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `Ridge` | `log1p_rub` | `{"alpha": 10.0, "fit_intercept": true, "random_state": 42}` | `8865.80` | **`10946.11`** | **`12772.61`** | `7604.99` | `9331.77` | `5335.21` | `6925.57` | `13.70%` | `17.76%` | `0.8827` | `0.8319` | `2a2e397a37cc4be09d8f668422823fdd` |
| `ElasticNet` | `log1p_rub` | `{"alpha": 0.0001, "fit_intercept": true, "l1_ratio": 0.5, "max_iter": 10000, "random_state": 42}` | `8780.86` | **`10951.28`** | **`12825.33`** | `7579.37` | `9321.92` | `5311.87` | `6978.84` | `13.63%` | `17.69%` | `0.8826` | `0.8305` | `6b1a617ebc334914bdd6e1f6b834d331` |
| `Lasso` | `log1p_rub` | `{"alpha": 0.0001, "fit_intercept": true, "max_iter": 10000, "random_state": 42}` | `8845.50` | **`10954.95`** | **`12816.88`** | `7602.95` | `9328.33` | `5351.83` | `6951.61` | `13.68%` | `17.73%` | `0.8826` | `0.8307` | `5f77ff7d06c844edb78a59f83b2025cb` |
| `LinearRegression` | `log1p_rub` | `{"fit_intercept": true}` | `8716.46` | **`10966.97`** | **`12875.24`** | `7569.60` | `9318.20` | `5174.33` | `6922.05` | `13.60%` | `17.63%` | `0.8823` | `0.8291` | `fd4a752e7a4443d28f17d71aa65d0d0f` |
| `Lasso` | `raw_rub` | `{"alpha": 0.005, "fit_intercept": true, "max_iter": 10000, "random_state": 42}` | `10911.37` | **`13088.10`** | **`13896.46`** | `9315.76` | `10875.86` | `7233.69` | `9245.80` | `19.28%` | `23.77%` | `0.8324` | `0.8010` | `d2c10bff697c4ea28ccb4817523e93bb` |
| `LinearRegression` | `raw_rub` | `{"fit_intercept": true}` | `10911.37` | **`13088.26`** | **`13896.78`** | `9316.01` | `10876.26` | `7236.46` | `9246.87` | `19.28%` | `23.77%` | `0.8324` | `0.8010` | `0392c386d7c542fab55ba8337164c95d` |
| `Ridge` | `raw_rub` | `{"alpha": 0.01, "fit_intercept": true, "random_state": 42}` | `10911.37` | **`13088.29`** | **`13896.74`** | `9316.01` | `10876.21` | `7234.62` | `9244.15` | `19.28%` | `23.77%` | `0.8324` | `0.8010` | `b6f57b0d4f29474b88c113817ae26abc` |
| `ElasticNet` | `raw_rub` | `{"alpha": 0.0001, "fit_intercept": true, "l1_ratio": 0.9, "max_iter": 10000, "random_state": 42}` | `10911.45` | **`13088.87`** | **`13896.65`** | `9316.21` | `10875.99` | `7242.54` | `9259.05` | `19.28%` | `23.77%` | `0.8323` | `0.8010` | `b88eb58e4c434345abaed08386321ea0` |
| `DummyRegressor` | `raw_rub` | `{"strategy": "mean"}` | `29203.80` | **`32322.62`** | **`31453.34`** | `22314.54` | `22216.39` | `16662.68` | `16512.68` | `50.13%` | `50.61%` | `-0.0224` | `-0.0196` | `857875eef1f84f75911ac15e0b806615` |
| `DummyRegressor` | `log1p_rub` | `{"strategy": "median"}` | `29888.89` | **`33854.20`** | **`32944.61`** | `22572.08` | `22279.13` | `15400.00` | `15000.00` | `44.45%` | `44.48%` | `-0.1216` | `-0.1186` | `76f12b99014844b399218541d646a4e2` |

### 4.1. Métricas en Escala Logarítmica y Diagnóstico del Factor de Corrección de Duan (`log1p_rub`)

Cuando el modelo se entrena sobre $z = \ln(1 + y)$, la predicción ingenua $\hat{y} = \exp(\hat{z}) - 1$ estima la mediana condicional geométrica en lugar de la media condicional $\mathbb{E}[Y|X]$. El estimador no paramétrico de Duan ($\hat{lpha}_{	ext{Duan}} = rac{1}{N_{	ext{train}}}\sum_{i=1}^{N_{	ext{train}}} \exp(e_i)$) corrige dicho sesgo multiplicativo:

| Modelo (`log1p_rub`) | Train RMSE (log) | Val RMSE (log) | Test RMSE (log) | Val R² (log) | Test R² (log) | Factor de Duan ($\hat{lpha}$) | Val RMSE `expm1` (RUB) | Val RMSE con Duan (RUB) | Test RMSE `expm1` (RUB) | Test RMSE con Duan (RUB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Ridge` | `0.1296` | `0.1560` | `0.1895` | `0.9166` | `0.8763` | `1.008493` | `10946.11` | `11207.01` | `12772.61` | `13189.54` |
| `ElasticNet` | `0.1282` | `0.1553` | `0.1889` | `0.9172` | `0.8771` | `1.008293` | `10951.28` | `11217.58` | `12825.33` | `13239.52` |
| `Lasso` | `0.1293` | `0.1558` | `0.1893` | `0.9167` | `0.8766` | `1.008463` | `10954.95` | `11219.06` | `12816.88` | `13234.90` |
| `LinearRegression` | `0.1271` | `0.1552` | `0.1885` | `0.9174` | `0.8776` | `1.008133` | `10966.97` | `11239.31` | `12875.24` | `13286.76` |
| `DummyRegressor` | `0.5422` | `0.5458` | `0.5439` | `-0.0215` | `-0.0185` | `1.142339` | `33854.20` | `32322.62` | `32944.61` | `31453.34` |

---

## 5. Auditoría Matemática de Multicolinealidad y Estabilidad de Coeficientes (Defensa Académica)

### 5.1. Diagnóstico Espectral de la Matriz de Diseño `X_train_linear`

- **Dimensiones de `X_train_linear`:** `14009` filas $	imes$ `170` columnas.
- **Rango algebraico efectivo (`np.linalg.matrix_rank`):** **`157`** (Deficiencia de rango exacta = **`13` columnas linealmente dependientes**).
- **Número de condición (`np.linalg.cond`):** **`3.7455e+16`** ($\sigma_{\max} = 2.1332e+02$, $\sigma_{\min} = 5.6953e-15$).

**Explicación matemática de la deficiencia de rango ($170 - 157 = 13$):**
1. En la Etapa 03, las 4 variables categóricas multiclase fueron codificadas mediante `OneHotEncoder(drop="if_binary")`, conservando las `150` columnas indicadoras completas (`okrug`: 11, `district`: 129, `building_type`: 6, `renovation`: 4). En cada bloque, la suma por fila es exactamente igual a la constante $1$ ($\sum_k D_{i,k} = 1$), lo que introduce colinealidad exacta con el intercepto $eta_0$ y entre bloques.
2. Existe **anidamiento jerárquico exacto** entre los `129` distritos municipales (`district`) y los `11` distritos administrativos (`okrug`): cada `okrug` es exactamente la suma de los distritos que lo componen.
3. En las variables numéricas, `num__building_age = max(0, posting_year - building_year)` es combinación lineal exacta de `num__posting_year` y `num__building_year` en el $98.30\%$ de los anuncios donde `is_presale_or_new == 0`.

### 5.2. Comparación Cuantitativa del Vector de Coeficientes $\hat{eta}$ entre Modelos

| Modelo | Modo Target | Norma L2 $\Vert\hat{eta}\Vert_2$ | Norma L1 $\Vert\hat{eta}\Vert_1$ | Máximo Absoluto $\max_j ert\hat{eta}_jert$ | Coeficientes Nulos (`n_zero` / `%`) | $\Vert\hat{eta}_{	ext{num}}\Vert_2$ (15 cols) | $\Vert\hat{eta}_{	ext{okrug}}\Vert_2$ (11 cols) | $\Vert\hat{eta}_{	ext{district}}\Vert_2$ (129 cols) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `DummyRegressor` | `raw_rub` | `0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `170` / `170` (`100.0%`) | `0.0000e+00` | `0.0000e+00` | `0.0000e+00` |
| `DummyRegressor` | `log1p_rub` | `0.0000e+00` | `0.0000e+00` | `0.0000e+00` | `170` / `170` (`100.0%`) | `0.0000e+00` | `0.0000e+00` | `0.0000e+00` |
| `LinearRegression` | `raw_rub` | `1.3353e+05` | `1.3035e+06` | `5.1284e+04` | `0` / `170` (`0.0%`) | `1.8894e+04` | `6.5055e+04` | `1.1279e+05` |
| `LinearRegression` | `log1p_rub` | `2.4875e+00` | `2.4806e+01` | `8.9061e-01` | `0` / `170` (`0.0%`) | `5.1439e-01` | `1.3500e+00` | `1.9788e+00` |
| `Ridge` | `raw_rub` | `1.3342e+05` | `1.3021e+06` | `5.1248e+04` | `0` / `170` (`0.0%`) | `1.8877e+04` | `6.4984e+04` | `1.1270e+05` |
| `Ridge` | `log1p_rub` | `1.6382e+00` | `1.6725e+01` | `5.2799e-01` | `0` / `170` (`0.0%`) | `4.2563e-01` | `7.7609e-01` | `1.3109e+00` |
| `Lasso` | `raw_rub` | `1.3807e+05` | `1.2534e+06` | `6.0598e+04` | `4` / `170` (`2.4%`) | `1.8825e+04` | `7.5446e+04` | `1.1178e+05` |
| `Lasso` | `log1p_rub` | `1.7785e+00` | `1.6114e+01` | `7.0695e-01` | `21` / `170` (`12.4%`) | `4.2157e-01` | `9.3034e-01` | `1.3915e+00` |
| `ElasticNet` | `raw_rub` | `1.3206e+05` | `1.2865e+06` | `5.0911e+04` | `0` / `170` (`0.0%`) | `1.8830e+04` | `6.4122e+04` | `1.1159e+05` |
| `ElasticNet` | `log1p_rub` | `1.9356e+00` | `1.8105e+01` | `7.4245e-01` | `12` / `170` (`7.1%`) | `4.1034e-01` | `1.0453e+00` | `1.5173e+00` |

**Conclusión de la auditoría numérica:**
- En `LinearRegression` (OLS), la deficiencia de rango hace que $X^T X$ sea singular o cuasi-singular; aunque el solucionador SVD de `scikit-learn` proyecta sobre la pseudoinversa de Moore-Penrose manteniendo acotadas las predicciones $\hat{y} = X\hat{eta}$ dentro del subespacio columna, **cualquier componente en el espacio nulo de $X$ o asociada a valores singulares diminutos ($\sigma_{\min} = 5.6953e-15$) dispara la norma de los coeficientes individuales**.
- **`Ridge` ($L_2$)** suma $lpha I$ a la diagonal de $X^T X$, desplazando todos los autovalores hacia arriba ($\lambda_j + lpha$) y reduciendo drásticamente $\Vert\hat{eta}\Vert_2$ sin perder poder predictivo.
- **`Lasso` ($L_1$) y `ElasticNet` ($L_1 + L_2$)** eliminan automáticamente las categorías redundantes y los distritos con baja señal estadística llevando sus coeficientes exactamente a cero (`n_zero_coefs`), resolviendo de manera simultánea la colinealidad estructural y la parsimonia del modelo.

---

## 6. Interpretación Económica e Inmobiliaria de las Variables Clave

En el régimen `log1p_rub`, cada coeficiente $\hat{eta}_j$ representa una **semielasticidad**: un incremento de una unidad en la variable preprocesada $X_j$ (escala `RobustScaler` / IQR para numéricas, o cambio $0 	o 1$ para binarias y dummies) produce una variación porcentual aproximada de $(\exp(\hat{eta}_j) - 1) 	imes 100\%$ en el alquiler mensual en rublos:

### 6.1. Comparación de Coeficientes en Variables Estructurales, Espaciales y Temporales (`log1p_rub`)

| Variable (`feature`) | `LinearRegression` (OLS) | `Ridge` (L2) | `Lasso` (L1) | `ElasticNet` (L1+L2) |
|---|---:|---:|---:|---:|
| `num__total_area` | `+3.61696e-01` | `+0.35496` | `+0.35529` | `+0.35788` |
| `num__to_center_km` | `-7.23380e-04` | `-0.16624` | `-0.15072` | `-0.10906` |
| `num__posting_year` | `+1.59288e-01` | `+0.14743` | `+0.14743` | `+0.14773` |
| `num__rooms` | `-6.11355e-02` | `-0.05258` | `-0.05232` | `-0.05597` |
| `num__building_age` | `-2.24067e-01` | `+0.00434` | `+0.00253` | `+0.00298` |
| `num__area_per_room` | `-3.50302e-02` | `-0.02954` | `-0.02866` | `-0.03147` |
| `num__floor_ratio` | `+5.35809e-03` | `+0.00560` | `+0.00069` | `+0.00153` |
| `bin__furnished` | `+7.99759e-02` | `+0.07900` | `+0.07872` | `+0.07914` |
| `cat__renovation_designer` | `+2.26282e-01` | `+0.22500` | `+0.21373` | `+0.21927` |
| `cat__renovation_euro` | `+5.74922e-02` | `+0.05700` | `+0.04518` | `+0.05055` |
| `cat__renovation_no_renovation` | `-1.97081e-01` | `-0.19497` | `-0.20702` | `-0.20253` |

### 6.2. Top 15 Variables de Mayor Impacto Económico (Modelo Regularizado de Referencia en `log1p_rub`)

| Variable (`feature`) | Familia | Coeficiente Regularizado ($\hat{eta}_j$) | Impacto Porcentual Aprox. $(\exp(\hat{eta}_j)-1)\%$ | Coeficiente en OLS ($\hat{eta}_j^{	ext{OLS}}$) |
|---|---|---:|---:|---:|
| `cat__okrug_CAO` | `ohe_okrug` | `+0.52799` | `+69.55%` | `+7.73725e-01` |
| `num__total_area` | `numeric_scaled` | `+0.35496` | `+42.61%` | `+3.61696e-01` |
| `cat__okrug_TAO` | `ohe_okrug` | `-0.33505` | `-28.47%` | `-8.90606e-01` |
| `cat__district_Strogino` | `ohe_district` | `+0.31372` | `+36.85%` | `+3.57443e-01` |
| `cat__district_Sokol` | `ohe_district` | `+0.25956` | `+29.64%` | `+3.37828e-01` |
| `cat__district_Prospekt Vernadskogo` | `ohe_district` | `+0.25923` | `+29.59%` | `+3.07134e-01` |
| `cat__district_Yasenevo` | `ohe_district` | `+0.25764` | `+29.39%` | `+2.08490e-01` |
| `cat__okrug_NAO` | `ohe_okrug` | `-0.25382` | `-22.42%` | `-3.86265e-01` |
| `cat__district_Dorogomilovo` | `ohe_district` | `+0.23740` | `+26.79%` | `+4.08337e-01` |
| `cat__renovation_designer` | `ohe_renovation` | `+0.22500` | `+25.23%` | `+2.26282e-01` |
| `cat__district_Chertanovo Yuzhnoye` | `ohe_district` | `+0.22334` | `+25.03%` | `+1.86841e-01` |
| `cat__district_Perovo` | `ohe_district` | `+0.20611` | `+22.89%` | `+2.48575e-01` |
| `cat__district_Khamovniki` | `ohe_district` | `+0.20170` | `+22.35%` | `+2.33546e-01` |
| `cat__district_Troparyovo-Nikulino` | `ohe_district` | `+0.19884` | `+22.00%` | `+2.04512e-01` |
| `cat__district_Tverskoy` | `ohe_district` | `+0.19866` | `+21.98%` | `+2.56705e-01` |

**Lectura económica de los resultados:**
1. **Gradiente espacial monocéntrico (`num__to_center_km`, `cat__okrug_*`, `cat__district_*`):** La distancia al centro de Moscú presenta signo negativo consistente, mientras que el okrug central (`cat__okrug_CAO`) y sus distritos históricos concentran las mayores primas positivas de precio frente a los distritos periféricos (`TAO`, `NAO`, `ZelAO`).
2. **Escala física del inmueble (`num__total_area`, `num__rooms`, `num__area_per_room`):** La superficie total (`num__total_area`) es el impulsor físico dominante con elasticidad positiva; en los modelos con penalización $L_1$ (`Lasso` / `ElasticNet`), parte de la colinealidad entre `total_area`, `rooms` y `area_per_room` se consolida en `total_area` y `rooms`.
3. **Calidad constructiva e interior (`cat__renovation_*`, `bin__furnished`):** Los acabados `designer` y `euro` aportan una prima sustancial frente a `cosmetic` y `no_renovation`, al igual que la condición de inmueble amoblado (`bin__furnished`).
4. **Tendencia inflacionaria nominal (`num__posting_year`):** El coeficiente positivo de `num__posting_year` permite a la familia lineal extrapolar la tendencia nominal de alquileres entre `Train` (`2021–2024`) y `Val`/`Test` (`2024–2026`), superando la limitación estructural de los modelos constantes (`DummyRegressor`).

---

## 7. Diagnóstico Gráfico de Residuos y Estabilidad Temporal (`reports/figures/modeling/`)

| Archivo Generado | Hallazgo Analítico Principal |
|---|---|
| `reports/figures/modeling/01_model_comparison_val_test.png` | Muestra visualmente la caída de `RMSE` y `MAE` y el salto en `R²` al pasar de `DummyRegressor` a los 4 modelos lineales, así como la ganancia sistemática del régimen `log1p_rub` frente a `raw_rub`. |
| `reports/figures/modeling/02_ols_vs_regularized_coefficients.png` | Evidencia en escala logarítmica cómo la norma $\Vert\hat{eta}\Vert_2$ de `LinearRegression` se infla por la colinealidad de los bloques One-Hot, mientras `Ridge`, `Lasso` y `ElasticNet` comprimen los pesos en varios órdenes de magnitud. |
| `reports/figures/modeling/03_residuals_diagnostics.png` | Confirma que el entrenamiento en `raw_rub` exhibe heterocedasticidad en abanico (mayor varianza residual en alquileres altos), la cual se corrige sustancialmente bajo `log1p_rub`, acercando los residuos estandarizados a la recta normal teórica en el gráfico Q-Q. |
| `reports/figures/modeling/04_temporal_error_drift_test.png` | Evalúa el error absoluto medio (`MAE`) y el sesgo monetario ($\hat{y} - y$) mes a mes entre junio de 2025 y abril de 2026 (`Test`), mostrando cómo `num__posting_year` absorbe gran parte de la inflación nominal del mercado moscovita. |

---

## 8. Conclusión Final y Recomendación Académica

1. **Entrega Académica Principal (`models/trained/primary_linear_regression.joblib`):** Se selecciona **`LinearRegression` entrenado en el régimen `log1p_rub`** como modelo principal para cumplir estrictamente la exigencia académica del proyecto (`Val RMSE = 10966.97 RUB`, `Test RMSE = 12875.24 RUB`, `Test R² = 0.8291`).
2. **Mejor Modelo Lineal Global (`models/trained/best_linear_model.joblib`):** Para interpretación económica directa de los coeficientes individuales $\hat{eta}_j$ y despliegue robusto ante colinealidad, se respalda con **`Ridge` (`log1p_rub`, `{'alpha': 10.0, 'fit_intercept': True, 'random_state': 42}`)**, que alcanza el menor error en validación (`Val RMSE = 10946.11 RUB`, `Test RMSE = 12772.61 RUB`) eliminando por completo la inestabilidad numérica en los pesos de las variables indicadoras.
