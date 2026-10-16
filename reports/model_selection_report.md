# Informe de Selección de Modelos — Alquileres en Moscú (Etapa 04)

| Metadato | Valor |
|---|---|
| Fecha de generación | `2026-10-16 18:58:36` |
| Script generador | `src/04_model_selection.py` |
| SHA-256 verificado del dataset limpio (`data/rentals_clean.csv`) | `e43378a0de55cdc2320b76f273ef47e1158a41e7fb0d06da0dc34666a5cc1a71` |
| Variable objetivo (`y`) | `monthly_rent_rub` (escala original RUB) y `monthly_rent_rub_log1p` ($\ln(1 + y)$) |
| Partición temporal (`Train` / `Val` / `Test`) | `14009` (`70.05%`) / `2991` (`14.95%`) / `3000` (`15.0%`) |
| Dimensiones de entrada (`linear` / `tree`) | `155` columnas / `170` columnas |
| Modelo principal requerido | `LinearRegression` (exigencia académica de regresión lineal) |
| Archivo de configuración exportado | `models/model_candidates.json` |

---

## 1. Síntesis de Hallazgos del EDA y Preprocesamiento que Guían la Selección

El análisis conjunto de `reports/eda_report.md`, `reports/preprocessing_report.md` y `models/preprocessing/preprocessing_metadata.json` establece las siguientes condiciones técnicas para el modelado:

1. **Asimetría positiva y heterocedasticidad en `monthly_rent_rub`:**
   - En escala original, el alquiler mensual oscila entre `6800` RUB y `300700` RUB, con media de `52432.9` RUB ($\text{DE} = 30003.2$ RUB) y mediana de `45800.0` RUB ($\text{IQR} = 31600.0$ RUB).
   - Presenta una asimetría positiva pronunciada ($\text{Skewness} = 1.820$ global; `1.821` en `Train`) y `936` observaciones (`4.68%`) por encima de $1.5 \times \text{IQR}$ correspondientes al segmento residencial *premium* en el centro de Moscú.
   - Al aplicar la transformación `np.log1p` (`monthly_rent_rub_log1p`), la asimetría se reduce a `-0.136` (`-0.133` en `Train`) y la curtosis a `0.216`, estabilizando la varianza condicional. Por ello, los modelos lineales deben evaluarse en el paso 5 considerando tanto el objetivo en escala logarítmica (con retransformación $\exp(\hat{y}) - 1$ mediante `target_log_transformer.joblib`) como en escala original.
2. **Multicolinealidad entre predictores físicos, espaciales y variables de ingeniería:**
   - El EDA identificó colinealidad entre `total_area` ($\text{VIF} = 4.546$, $R^2 = 0.78$) y `rooms` ($\text{VIF} = 4.545$, $R^2 = 0.78$), así como entre `to_center_km` ($\text{VIF} = 7.777$), `lat` ($\text{VIF} = 2.014$) y `lon` ($\text{VIF} = 2.051$).
   - En la etapa 03, `DomainFeatureEngineer` incorporó ratios y diferencias exactas (`num__area_per_room`, `num__floor_ratio`) que agregan dependencia funcional con sus variables de origen (`total_area`, `rooms`, `floor`, `total_floors`). `building_age` se excluye de la rama lineal por ser combinación de `building_year` y el tiempo.
   - Las categóricas (`okrug` con `0` niveles, `district` con `128`, `building_type` con `5` y `renovation` con `3`) existen en el dataset, pero `okrug` es la suma exacta de sus `district` ($\eta^2 = 0.4916$ vs $0.6044$). La etapa 03 corrigió la rama lineal con `OneHotEncoder(drop="first")` y retirando `okrug`, de modo que la matriz tiene rango completo (136 dummies). Persiste casi-colinealidad entre `to_center_km` y `district` (~99% de la varianza de la distancia la explica el distrito). Esto motiva contrastar `LinearRegression` (OLS) frente a `Ridge` ($L_2$) y `ElasticNet` ($L_1 + L_2$).
3. **Alta dimensionalidad (`155` columnas) y baja frecuencia por distrito:**
   - La matriz `X_*_linear.csv` cuenta con `14` variables numéricas escaladas con `RobustScaler`, `5` binarias y `136` dummies categóricas.
   - Las `128` categorías de `district` presentan frecuencias individuales menores al $1\%$ (entre $0.61\%$ y $0.95\%$). La regularización $L_1$ (`Lasso` y `ElasticNet`) resulta idónea para anular coeficientes de distritos con poca señal estadística y evitar sobreajuste.
4. **Deriva temporal e inflación nominal (Partición Cronológica Estricta):**
   - El dataset abarca de `2021-01-01` a `2026-04-29`. La partición cronológica (`Train`: `2021-01-01` a `2024-09-06`; `Val`: `2024-09-07` a `2025-06-29`; `Test`: `2025-06-30` a `2026-04-29`) exige que los modelos capturen la tendencia inflacionaria sin fuga de información futura. El mercado se estancó desde 2024 (precio mediano por m² plano), por lo que la etapa 03 usa `num__posting_trend`, congelada en el último valor visto en Train, en lugar de extrapolar `posting_year` linealmente.

---

## 2. Resumen de Modelos Seleccionados y Decisión sobre Familias Adicionales

Siguiendo los requisitos académicos del proyecto, se selecciona **`LinearRegression` como modelo principal**, acompañado exclusivamente por tres alternativas lineales regularizadas (**`Ridge`**, **`Lasso`**, **`ElasticNet`**) y un modelo de referencia (**`DummyRegressor`**):

| ID Candidato | Modelo (`scikit-learn`) | Rol en el Proyecto | Matriz de Entrada | Objetivo Recomendado |
|---|---|---|---|---|
| `dummy_baseline` | `DummyRegressor` (`sklearn.dummy.DummyRegressor`) | Referencia Básica (Baseline) | `linear` (`155` cols) | `monthly_rent_rub` (y contraste con `monthly_rent_rub`) |
| `linear_regression_ols` | `LinearRegression` (`sklearn.linear_model.LinearRegression`) | **Modelo Principal (Entrega Académica)** | `linear` (`155` cols) | `monthly_rent_rub_log1p` (y contraste con `monthly_rent_rub`) |
| `ridge_l2` | `Ridge` (`sklearn.linear_model.Ridge`) | Alternativa Lineal Regularizada | `linear` (`155` cols) | `monthly_rent_rub_log1p` (y contraste con `monthly_rent_rub`) |
| `lasso_l1` | `Lasso` (`sklearn.linear_model.Lasso`) | Alternativa Lineal Regularizada | `linear` (`155` cols) | `monthly_rent_rub_log1p` (y contraste con `monthly_rent_rub`) |
| `elasticnet_l1_l2` | `ElasticNet` (`sklearn.linear_model.ElasticNet`) | Alternativa Lineal Regularizada | `linear` (`155` cols) | `monthly_rent_rub_log1p` (y contraste con `monthly_rent_rub`) |

### Justificación de No Incorporación de Modelos Basados en Árboles u Otras Familias
- Aunque la etapa 03 generó matrices específicas sin escalado para árboles (`data/processed/X_*_tree.csv` y `models/preprocessing/preprocessor_tree.joblib`), **no se incluyen modelos de árboles (`RandomForest`, `ExtraTrees`, `LightGBM`, `XGBoost`, `CatBoost`)** en el conjunto de candidatos de la etapa 05 porque:
  1. La entrega académica exige explícitamente un **modelo final de regresión lineal**.
  2. Las directrices del proyecto establecen no incorporar modelos adicionales innecesarios fuera de la familia lineal requerida y su línea base.
  3. Los modelos basados en árboles de decisión no extrapolan tendencias lineales fuera del soporte observado del tiempo (`2021`–`2024` en `Train` frente a `2024`–`2026` en `Val`/`Test`) a menos que se desestacionalice o deflacte externamente el objetivo.

---

## 3. Detalle Técnico de los Modelos Candidatos

### 3.1. `DummyRegressor` (`dummy_baseline`)

- **Clase en scikit-learn:** `sklearn.dummy.DummyRegressor`
- **Rol en el experimento:** `baseline_reference` (Modelo principal académico: `No`)
- **Justificación basada en EDA y Preprocesamiento:** Punto de referencia básico indispensable (sanity check). Predice una constante (mediana de 45,800 RUB o media de 52,432.9 RUB en el conjunto histórico) sin usar predictores. Establece el piso mínimo de error (RMSE y MAE máximos aceptables) que cualquier regresión lineal debe superar en validación temporal.
- **Ventajas:**
  - Costo computacional nulo y ausencia total de sobreajuste.
  - La estrategia 'median' es robusta frente a la asimetría positiva del target (skewness = 1.821 en Train).
  - Permite cuantificar la ganancia predictiva real aportada por las 155 features.
- **Limitaciones:**
  - No captura ninguna relación espacial, física ni temporal (R² <= 0 por definición en media).
  - Incapaz de reflejar la inflación nominal entre los periodos Train (2021-2024) y Val/Test (2024-2026).
- **Configuración base (`default_params`):** `{"strategy": "median"}`
- **Grilla de búsqueda para Validación en Etapa 05 (`search_grid`):** `{"strategy": ["median", "mean"]}`
- **Archivos de entrada asignados para la Etapa 05:**
  - Matrices de predictores (`X`): `data/processed/X_train_linear.csv`, `data/processed/X_val_linear.csv`, `data/processed/X_test_linear.csv`
  - Vectores objetivo (`y`): `data/processed/y_train.csv`, `data/processed/y_val.csv`, `data/processed/y_test.csv` (columnas `monthly_rent_rub` y `monthly_rent_rub_log1p`)
  - Artefactos de transformación: `models/preprocessing/preprocessor_linear.joblib` y `models/preprocessing/target_log_transformer.joblib`

### 3.2. `LinearRegression` (`linear_regression_ols`)

- **Clase en scikit-learn:** `sklearn.linear_model.LinearRegression`
- **Rol en el experimento:** `primary_academic_model` (Modelo principal académico: `Sí`)
- **Justificación basada en EDA y Preprocesamiento:** Modelo principal del proyecto porque la entrega académica exige expresamente un modelo final de regresión lineal. Permite estimar por Mínimos Cuadrados Ordinarios (OLS mediante descomposición SVD) el efecto marginal de variables clave como 'to_center_km' (r = -0.5314), 'total_area' (r = 0.4108) y los distritos ('district', eta² = 0.6044). Se recomienda su ajuste sobre 'monthly_rent_rub_log1p' (asimetría = -0.133) contrastándolo contra la escala original en el paso 5.
- **Ventajas:**
  - Cumple de manera directa el requisito académico obligatorio del proyecto.
  - Máxima interpretabilidad económica de los coeficientes (elasticidades y semielasticidades en escala log1p).
  - Entrenamiento determinista y cerrado sin hiperparámetros de penalización que ajustar.
  - Captura la tendencia temporal mediante 'num__posting_trend', congelada en el último valor visto en Train (no extrapola una inflación que ya no continúa).
- **Limitaciones:**
  - Alta varianza en los estimadores ante multicolinealidad entre predictores físicos ('total_area', 'rooms', 'area_per_room').
  - La rama lineal de la etapa 03 ya no tiene dependencias exactas (drop='first', sin 'okrug' ni 'building_age'), pero persiste casi-colinealidad entre 'to_center_km' y las dummies de 'district' (district explica ~99% de su varianza): los coeficientes de ambos no son interpretables por separado.
  - Sensible a valores extremos en escala original (936 outliers > 1.5 IQR) si no se emplea la transformación log1p.
- **Configuración base (`default_params`):** `{"fit_intercept": true}`
- **Grilla de búsqueda para Validación en Etapa 05 (`search_grid`):** `{"fit_intercept": [true]}`
- **Archivos de entrada asignados para la Etapa 05:**
  - Matrices de predictores (`X`): `data/processed/X_train_linear.csv`, `data/processed/X_val_linear.csv`, `data/processed/X_test_linear.csv`
  - Vectores objetivo (`y`): `data/processed/y_train.csv`, `data/processed/y_val.csv`, `data/processed/y_test.csv` (columnas `monthly_rent_rub` y `monthly_rent_rub_log1p`)
  - Artefactos de transformación: `models/preprocessing/preprocessor_linear.joblib` y `models/preprocessing/target_log_transformer.joblib`

### 3.3. `Ridge` (`ridge_l2`)

- **Clase en scikit-learn:** `sklearn.linear_model.Ridge`
- **Rol en el experimento:** `regularized_linear_alternative` (Modelo principal académico: `No`)
- **Justificación basada en EDA y Preprocesamiento:** Alternativa lineal con regularización L2 diseñada específicamente para mitigar la multicolinealidad documentada en el EDA (VIF = 4.546 en 'total_area' y 4.545 en 'rooms') y en el preprocesamiento (ratios derivados 'area_per_room', 'floor_ratio') y en la casi-colinealidad entre 'to_center_km' y las 129 dummies de 'district').
- **Ventajas:**
  - Estabiliza la inversión de la matriz X^T X añadiendo penalización L2 (alpha * ||w||_2^2).
  - Retiene las 155 variables repartiendo el peso entre predictores correlacionados sin perder información geográfica.
  - Reduce la varianza de las predicciones en los distritos de baja frecuencia (< 1% cada uno).
- **Limitaciones:**
  - No realiza selección de variables (ningún coeficiente se anula exactamente a cero).
  - Requiere calibrar el hiperparámetro 'alpha' sobre el conjunto de validación temporal.
- **Configuración base (`default_params`):** `{"alpha": 1.0, "fit_intercept": true, "random_state": 42}`
- **Grilla de búsqueda para Validación en Etapa 05 (`search_grid`):** `{"alpha": [0.01, 0.1, 1.0, 10.0, 50.0, 100.0]}`
- **Archivos de entrada asignados para la Etapa 05:**
  - Matrices de predictores (`X`): `data/processed/X_train_linear.csv`, `data/processed/X_val_linear.csv`, `data/processed/X_test_linear.csv`
  - Vectores objetivo (`y`): `data/processed/y_train.csv`, `data/processed/y_val.csv`, `data/processed/y_test.csv` (columnas `monthly_rent_rub` y `monthly_rent_rub_log1p`)
  - Artefactos de transformación: `models/preprocessing/preprocessor_linear.joblib` y `models/preprocessing/target_log_transformer.joblib`

### 3.4. `Lasso` (`lasso_l1`)

- **Clase en scikit-learn:** `sklearn.linear_model.Lasso`
- **Rol en el experimento:** `regularized_linear_alternative` (Modelo principal académico: `No`)
- **Justificación basada en EDA y Preprocesamiento:** Alternativa lineal con penalización L1 orientada a la reducción de dimensionalidad y selección automática de variables sobre las 155 columnas de 'X_*_linear.csv', donde 136 provienen de One-Hot Encoding y las 129 categorías de 'district' tienen frecuencias individuales inferiores al 1% (entre 0.61% y 0.95%).
- **Ventajas:**
  - Produce modelos parsimoniosos (sparse) llevando exactamente a cero los coeficientes de distritos o variables redundantes.
  - Anula automáticamente distritos y dummies con poca señal (pocos anuncios por distrito).
  - Facilita identificar el subconjunto mínimo de predictores con mayor impacto real.
- **Limitaciones:**
  - Ante grupos de variables altamente correlacionadas (como 'total_area', 'rooms' y 'area_per_room'), tiende a seleccionar una de forma arbitraria y descartar las demás.
  - Si 'alpha' es demasiado alto, puede subajustar (underfitting) al descartar distritos con primas de precio reales.
- **Configuración base (`default_params`):** `{"alpha": 0.001, "max_iter": 10000, "fit_intercept": true, "random_state": 42}`
- **Grilla de búsqueda para Validación en Etapa 05 (`search_grid`):** `{"alpha": [1e-05, 5e-05, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1.0], "max_iter": [10000]}`
- **Archivos de entrada asignados para la Etapa 05:**
  - Matrices de predictores (`X`): `data/processed/X_train_linear.csv`, `data/processed/X_val_linear.csv`, `data/processed/X_test_linear.csv`
  - Vectores objetivo (`y`): `data/processed/y_train.csv`, `data/processed/y_val.csv`, `data/processed/y_test.csv` (columnas `monthly_rent_rub` y `monthly_rent_rub_log1p`)
  - Artefactos de transformación: `models/preprocessing/preprocessor_linear.joblib` y `models/preprocessing/target_log_transformer.joblib`

### 3.5. `ElasticNet` (`elasticnet_l1_l2`)

- **Clase en scikit-learn:** `sklearn.linear_model.ElasticNet`
- **Rol en el experimento:** `regularized_linear_alternative` (Modelo principal académico: `No`)
- **Justificación basada en EDA y Preprocesamiento:** Compromiso entre Ridge (L2) y Lasso (L1). Permite podar dummies irrelevantes de 'district' mediante la componente L1 mientras mantiene agrupados y estables los predictores físicamente colineales ('total_area', 'rooms', 'area_per_room' y coordenadas espaciales 'lat', 'lon', 'to_center_km') gracias a la componente L2.
- **Ventajas:**
  - Supera la limitación de selección arbitraria de Lasso en presencia de grupos colineales (efecto de agrupación).
  - Combina esparsidad controlada con estabilidad numérica en espacios de 155 dimensiones.
  - Permite explorar de forma continua el balance entre L1 y L2 mediante 'l1_ratio'.
- **Limitaciones:**
  - Mayor espacio de búsqueda en validación al requerir el ajuste conjunto de dos hiperparámetros ('alpha' y 'l1_ratio').
  - Requiere 'max_iter' suficientemente alto para garantizar convergencia por descenso coordenado.
- **Configuración base (`default_params`):** `{"alpha": 0.001, "l1_ratio": 0.5, "max_iter": 10000, "fit_intercept": true, "random_state": 42}`
- **Grilla de búsqueda para Validación en Etapa 05 (`search_grid`):** `{"alpha": [1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0], "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9], "max_iter": [10000]}`
- **Archivos de entrada asignados para la Etapa 05:**
  - Matrices de predictores (`X`): `data/processed/X_train_linear.csv`, `data/processed/X_val_linear.csv`, `data/processed/X_test_linear.csv`
  - Vectores objetivo (`y`): `data/processed/y_train.csv`, `data/processed/y_val.csv`, `data/processed/y_test.csv` (columnas `monthly_rent_rub` y `monthly_rent_rub_log1p`)
  - Artefactos de transformación: `models/preprocessing/preprocessor_linear.joblib` y `models/preprocessing/target_log_transformer.joblib`

---

## 4. Mapeo de Archivos de Entrada y Artefactos para la Etapa 05

### 4.1. Archivos que DEBE utilizar la Etapa 05 (Modelos Lineales Seleccionados)

Todos los modelos seleccionados (`DummyRegressor`, `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`) deben consumir la rama **`linear`** generada en el paso 3:

| Tipo de Recurso | Ruta del Archivo | Uso en la Etapa 05 (`05_training_evaluation.py`) |
|---|---|---|
| **Predictores Entrenamiento (`X_train`)** | `data/processed/X_train_linear.csv` | Ajuste (`.fit()`) de los modelos candidatos (`14009` filas × `155` columnas). |
| **Predictores Validación (`X_val`)** | `data/processed/X_val_linear.csv` | Selección de hiperparámetros (`alpha`, `l1_ratio`, `strategy`) y comparación de candidatos (`2991` filas × `155` columnas). |
| **Predictores Prueba (`X_test`)** | `data/processed/X_test_linear.csv` | Evaluación final *holdout* del modelo lineal seleccionado (`3000` filas × `155` columnas). |
| **Objetivo Entrenamiento (`y_train`)** | `data/processed/y_train.csv` | Contiene `id`, `monthly_rent_rub` (escala RUB) y `monthly_rent_rub_log1p` (escala $\ln(1+y)$). |
| **Objetivo Validación (`y_val`)** | `data/processed/y_val.csv` | Evaluación en ventana temporal intermedia (`2024-09-07` a `2025-06-29`). |
| **Objetivo Prueba (`y_test`)** | `data/processed/y_test.csv` | Evaluación final en ventana futura (`2025-06-30` a `2026-04-29`). |
| **Pipeline Preprocesador Lineal** | `models/preprocessing/preprocessor_linear.joblib` | Artefacto `joblib` ajustado en `X_train` para empaquetar junto al modelo lineal final en MLflow. |
| **Transformador Logarítmico del Target** | `models/preprocessing/target_log_transformer.joblib` | Artefacto `FunctionTransformer(log1p, expm1)` para invertir las predicciones logarítmicas a RUB antes de calcular métricas. |

### 4.2. Archivos Generados por el Paso 3 que NO se utilizan en los Candidatos Lineales

Para mantener trazabilidad completa con `03_preprocessing.py`, se deja constancia de los archivos de la rama `tree` que permanecen en disco sin ser consumidos por los modelos lineales:

- `data/processed/X_train_tree.csv`, `data/processed/X_val_tree.csv`, `data/processed/X_test_tree.csv` y `models/preprocessing/preprocessor_tree.joblib` (matrices con las 15 variables numéricas en sus unidades físicas originales sin `RobustScaler`, destinadas únicamente a algoritmos de partición recursiva por árboles).

---

## 5. Lineamientos de Entrenamiento, Evaluación y Registro en MLflow para la Etapa 05

1. **Prohibición de re-preprocesamiento y prevención de fuga:** El paso 5 debe cargar directamente los archivos `X_*_linear.csv` y `y_*.csv` sin volver a ejecutar `RobustScaler` ni `OneHotEncoder`, y sin usar `X_test_linear.csv` para decidir hiperparámetros.
2. **Comparación en la misma escala monetaria (RUB):** Cuando un modelo se entrene sobre `monthly_rent_rub_log1p`, sus predicciones $\hat{z}$ deben transformarse mediante $\hat{y} = \exp(\hat{z}) - 1$ (`np.expm1` o `target_log_transformer.inverse_transform`) para calcular `RMSE`, `MAE`, `MedAE`, `MAPE` y `R2` en rublos comparables uno a uno con el entrenamiento en escala cruda.
3. **Trazabilidad en MLflow:** Cada ejecución (`run`) en el paso 5 deberá registrar como mínimo los parámetros de `models/model_candidates.json`, el régimen de la variable objetivo (`raw_rub` vs `log1p_rub`), las métricas en `Train`, `Val` y `Test`, y el modelo lineal final serializado.
