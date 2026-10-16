# Informe de Análisis Exploratorio de Datos (EDA) — Alquileres en Moscú (Etapa 02)

| Metadato | Valor |
|---|---|
| Fecha de generación | `2026-10-16 13:58:54` |
| Archivo analizado (solo lectura) | `data\rentals_clean.csv` |
| SHA-256 verificado (inmutable) | `e43378a0de55cdc2320b76f273ef47e1158a41e7fb0d06da0dc34666a5cc1a71` |
| Dimensiones del dataset | `20000` filas × `23` columnas |
| Variable objetivo | `monthly_rent_rub` (alquiler mensual en RUB) |

## 1. Resumen y Contraste de la Limpieza Inicial (`01_cleaning.py`)

- **Informe de limpieza detectado:** `data\cleaning_report.md`.
- **Verificación de integridad SHA-256 contra salida de etapa 01:** `COINCIDE EXACTAMENTE` (`e43378a0de55cdc2320b76f273ef47e1158a41e7fb0d06da0dc34666a5cc1a71`).
- **Contabilidad de registros en etapa 01:** Filas iniciales = `20000`, Filas eliminadas = `0`, Filas finales conservadas = `20000`.
- **Reglas aplicadas en la etapa previa y efectos sobre el EDA:**
  1. Normalización de texto y conversión de tipos sin pérdida de filas (`0` registros eliminados, `0` celdas convertidas a nulo forzoso).
  2. No se aplicaron filtros estadísticos de valores extremos (IQR, percentiles ni distancia de Cook) sobre `monthly_rent_rub` ni sobre predictores físicos.
  3. Se preservaron las anomalías temporales en la columna de auditoría `dq_flags` (`building_year_after_posting` detectado en `339` filas), la cual **no debe utilizarse como feature**.
  4. Se conservaron sin tocar variables redundantes o con potencial de fuga (`rent_per_sqm`) y variables con semántica dudosa (`metro_*`), delegando su diagnóstico riguroso a este EDA.

## 2. Estructura del Dataset y Calidad de Datos

### 2.1. Inventario de Variables, Tipos y Cardinalidad

| Columna | Tipo en Pandas | Rol Analítico | Nulos (%) | Valores Únicos | Descripción |
|---|---|---|---:|---:|---|
| `id` | `str` | `id` | 0.00% | 20,000 | Identificador único del anuncio |
| `date_posted` | `str` | `temporal` | 0.00% | 1,945 | Fecha de publicación del anuncio (ISO YYYY-MM-DD) |
| `district` | `str` | `categorical` | 0.00% | 129 | Distrito municipal (raion) de Moscú |
| `okrug` | `str` | `categorical` | 0.00% | 11 | Distrito administrativo (okrug) de Moscú |
| `lat` | `float64` | `spatial` | 0.00% | 15,985 | Latitud geográfica (grados decimales) |
| `lon` | `float64` | `spatial` | 0.00% | 16,678 | Longitud geográfica (grados decimales) |
| `total_area` | `float64` | `numeric_continuous` | 0.00% | 695 | Superficie total del inmueble (m²) |
| `rooms` | `int64` | `numeric_discrete` | 0.00% | 4 | Cantidad de habitaciones / ambientes |
| `floor` | `int64` | `numeric_discrete` | 0.00% | 22 | Piso en el que se ubica la unidad |
| `total_floors` | `int64` | `numeric_discrete` | 0.00% | 5 | Cantidad total de pisos del edificio |
| `building_year` | `int64` | `numeric_discrete` | 0.00% | 90 | Año de construcción del edificio |
| `building_type` | `str` | `categorical` | 0.00% | 6 | Material o tipología constructiva del edificio |
| `renovation` | `str` | `categorical` | 0.00% | 4 | Estado de renovación / remodelación interior |
| `furnished` | `bool` | `boolean` | 0.00% | 2 | Indica si el inmueble se alquila amoblado |
| `pets_allowed` | `bool` | `boolean` | 0.00% | 2 | Indica si se admiten mascotas |
| `deposit_months` | `int64` | `numeric_discrete` | 0.00% | 3 | Meses de depósito en garantía requeridos |
| `metro_station` | `str` | `categorical` | 0.00% | 104 | Estación de metro asignada al anuncio |
| `metro_line` | `str` | `categorical` | 0.00% | 12 | Línea de metro correspondiente a la estación |
| `metro_distance_min` | `int64` | `numeric_continuous` | 0.00% | 852 | Distancia reportada al metro (en minutos) |
| `to_center_km` | `float64` | `numeric_continuous` | 0.00% | 4,266 | Distancia en línea recta al centro de Moscú (km) |
| `monthly_rent_rub` | `int64` | `target` | 0.00% | 1,576 | Variable objetivo: alquiler mensual en rublos rusos (RUB) |
| `rent_per_sqm` | `int64` | `leakage_candidate` | 0.00% | 2,335 | Precio de alquiler por m² (RUB/m²) |
| `dq_flags` | `str` | `audit` | 98.31% | 1 | Banderas de auditoría generadas en 01_cleaning.py |

### 2.2. Chequeos de Integridad y Consistencia Lógica

- **Duplicados exactos (todas las columnas):** `0` filas.
- **Duplicados excluyendo `id`:** `0` filas.
- **Columnas constantes (varianza cero):** `Ninguna`.
- **Errores de coerción numérica en columnas cuantitativas:** `Ninguno (0 fallos)`.
- **Inconsistencia vertical (`floor > total_floors`):** `0` filas.
- **Pisos no positivos (`floor <= 0`):** `0` filas.
- **Anomalía temporal (`building_year > año(date_posted)`):** `339` filas (1.70% del total).
- **Coordenadas `(lat, lon)` idénticas en registros distintos:** `1` grupo(s) / `2` filas (unidades distintas dentro del mismo edificio o punto geográfico).

## 3. Estadística Descriptiva de Variables Numéricas y Categóricas

### 3.1. Resumen Paramétrico y No Paramétrico (Variables Numéricas)

| variable           |   n_valid |   missing_pct |   coerced_invalid |      mean |       std |    cv |      min |       p01 |       p05 |       p25 |    median |       p75 |        p95 |        p99 |        max |       iqr |   skewness |   kurtosis |   outliers_1_5_iqr |   outliers_1_5_pct |   outliers_3_0_iqr |
|:-------------------|----------:|--------------:|------------------:|----------:|----------:|------:|---------:|----------:|----------:|----------:|----------:|----------:|-----------:|-----------:|-----------:|----------:|-----------:|-----------:|-------------------:|-------------------:|-------------------:|
| monthly_rent_rub   |     20000 |             0 |                 0 | 52432.9   | 30003.2   | 0.572 | 6800     | 11600     | 17300     | 32800     | 45800     | 64400     | 109705     | 161102     | 300700     | 31600     |      1.82  |      5.565 |                936 |               4.68 |                212 |
| rent_per_sqm       |     20000 |             0 |                 0 |   945.461 |   480.537 | 0.508 |  148     |   246     |   329     |   644     |   845     |  1135     |   1873     |   2680.05  |   4264     |   491     |      1.593 |      4.02  |               1001 |               5    |                233 |
| total_area         |     20000 |             0 |                 0 |    55.387 |    12.74  | 0.23  |   22.1   |    33.199 |    37.7   |    45.7   |    53.8   |    63.3   |     79.3   |     89.2   |    104.6   |    17.6   |      0.577 |     -0.087 |                174 |               0.87 |                  0 |
| rooms              |     20000 |             0 |                 0 |     1.794 |     0.865 | 0.482 |    1     |     1     |     1     |     1     |     2     |     2     |      3     |      4     |      4     |     1     |      0.847 |     -0.115 |                945 |               4.72 |                  0 |
| floor              |     20000 |             0 |                 0 |     6.608 |     4.83  | 0.731 |    1     |     1     |     1     |     3     |     5     |     9     |     17     |     21     |     22     |     6     |      1.074 |      0.619 |                570 |               2.85 |                  0 |
| total_floors       |     20000 |             0 |                 0 |    12.191 |     5.692 | 0.467 |    5     |     5     |     5     |     9     |     9     |    17     |     22     |     22     |     22     |     8     |      0.437 |     -1.034 |                  0 |               0    |                  0 |
| building_year      |     20000 |             0 |                 0 |  1987.82  |    22.5   | 0.011 | 1935     |  1939     |  1954     |  1968     |  1988     |  2008     |   2021     |   2024     |   2024     |    40     |     -0.101 |     -1.141 |                  0 |               0    |                  0 |
| deposit_months     |     20000 |             0 |                 0 |     1.101 |     0.541 | 0.491 |    0     |     0     |     0     |     1     |     1     |     1     |      2     |      2     |      2     |     0     |      0.072 |      0.275 |               6053 |              30.27 |               6053 |
| metro_distance_min |     20000 |             0 |                 0 |   152.385 |   167.803 | 1.101 |    0     |     2     |     6     |    33     |   103     |   198     |    543     |    776     |    966     |   165     |      1.963 |      4.039 |               1416 |               7.08 |                488 |
| to_center_km       |     20000 |             0 |                 0 |    16.36  |    12.801 | 0.782 |    0.02  |     1.67  |     3.38  |     9.04  |    13.23  |    18.03  |     46.33  |     65.95  |     71.06  |     8.99  |      2.121 |      4.828 |               1905 |               9.53 |               1102 |
| lat                |     20000 |             0 |                 0 |    55.713 |     0.144 | 0.003 |   55.23  |    55.271 |    55.405 |    55.636 |    55.74  |    55.818 |     55.888 |     55.956 |     56.013 |     0.182 |     -0.98  |      0.908 |                672 |               3.36 |                  0 |
| lon                |     20000 |             0 |                 0 |    37.542 |     0.181 | 0.005 |   36.702 |    36.926 |    37.206 |    37.465 |    37.564 |    37.652 |     37.805 |     37.861 |     37.908 |     0.187 |     -1.224 |      2.969 |                743 |               3.71 |                161 |

### 3.2. Resumen de Cardinalidad, Frecuencias y Categorías Poco Frecuentes

| variable      | dtype   |   n_valid |   missing_pct |   n_unique | top_category                |   top_pct | least_category              |   least_pct |   rare_categories_lt_1pct |
|:--------------|:--------|----------:|--------------:|-----------:|:----------------------------|----------:|:----------------------------|------------:|--------------------------:|
| okrug         | str     |     20000 |          0    |         11 | SVAO                        |     12.94 | ZelAO                       |        0.69 |                         1 |
| district      | str     |     20000 |          0    |        129 | Dorogomilovo                |      0.95 | Ivanovskoye                 |        0.61 |                       129 |
| building_type | str     |     20000 |          0    |          6 | panel                       |     24.84 | stalin                      |        4.88 |                         0 |
| renovation    | str     |     20000 |          0    |          4 | cosmetic                    |     45.17 | no_renovation               |        9.93 |                         0 |
| furnished     | bool    |     20000 |          0    |          2 | True                        |     70.22 | False                       |       29.79 |                         0 |
| pets_allowed  | bool    |     20000 |          0    |          2 | False                       |     69.72 | True                        |       30.29 |                         0 |
| metro_line    | str     |     20000 |          0    |         12 | Kaluzhsko-Rizhskaya         |     14.14 | MCC                         |        1.47 |                         0 |
| metro_station | str     |     20000 |          0    |        104 | Sviblovo                    |      1.93 | Tekstilshchiki              |        0.54 |                        73 |
| dq_flags      | str     |       339 |         98.31 |          1 | building_year_after_posting |    100    | building_year_after_posting |      100    |                         0 |

## 4. Análisis Profundo de la Variable Objetivo (`monthly_rent_rub`)

- **Hecho observado (Rango y Tendencia Central):** `monthly_rent_rub` varía entre **6,800 RUB** y **300,700 RUB**, con una **mediana de 45,800 RUB** ($P_{25} = 32,800$, $P_{75} = 64,400$, $\text{IQR} = 31,600$ RUB) y una **media de 52,433 RUB** ($\text{DE} = 30,003$ RUB).
- **Hecho observado (Asimetría y Cola Derecha):** Presenta asimetría positiva pronunciada ($\text{Skewness} = 1.820$, $\text{Kurtosis} = 5.565$). Existen `936` observaciones (4.68%) por encima del umbral de Tukey ($1.5 \times \text{IQR}$), correspondientes al segmento residencial *premium* y de gran superficie en el centro de Moscú.
- **Efecto de la Transformación Logarítmica ($\ln(1 + y)$):** Al aplicar `np.log1p`, la asimetría se reduce drásticamente a **-0.136** (curtosis = **0.216**), estabilizando la varianza condicional (homocedasticidad).
- **Recomendación para el modelado:** Entrenar modelos paramétricos y basados en gradiente utilizando $\ln(\text{monthly\_rent\_rub})$ como variable respuesta (con corrección de sesgo de retransformación $\exp(\hat{y})$ o factor de Duan/smearing), o emplear funciones de pérdida robustas (Huber, MAE / L1, o Gamma/Tweedie deviance).

## 5. Relaciones con el Objetivo, Asociaciones y Multicolinealidad

### 5.1. Correlaciones Numéricas con `monthly_rent_rub`

| variable           |   pearson_r |   spearman_rho |   abs_spearman |
|:-------------------|------------:|---------------:|---------------:|
| rent_per_sqm       |      0.8933 |         0.8826 |         0.8826 |
| to_center_km       |     -0.5314 |        -0.6332 |         0.6332 |
| metro_distance_min |     -0.5123 |        -0.6329 |         0.6329 |
| total_area         |      0.4108 |         0.4376 |         0.4376 |
| rooms              |      0.3583 |         0.3785 |         0.3785 |
| lat                |      0.2738 |         0.2463 |         0.2463 |
| lon                |      0.2225 |         0.1926 |         0.1926 |
| building_year      |      0.0503 |         0.0663 |         0.0663 |
| floor              |     -0.0055 |        -0.0047 |         0.0047 |
| deposit_months     |      0.0011 |         0.0037 |         0.0037 |
| total_floors       |     -0.0024 |         0.0002 |         0.0002 |

### 5.2. Asociación de Variables Categóricas con `monthly_rent_rub` (Kruskal-Wallis y $\eta^2$)

| variable      |   n_unique |   kruskal_H |   p_value |   eta_squared |
|:--------------|-----------:|------------:|----------:|--------------:|
| district      |        129 |    12289.9  | 0         |        0.6044 |
| okrug         |         11 |     9853.63 | 0         |        0.4916 |
| renovation    |          4 |      881.52 | 9.04e-191 |        0.0427 |
| metro_station |        104 |      593.91 | 4.47e-70  |        0.0426 |
| building_type |          6 |      671.05 | 8.93e-143 |        0.0289 |
| metro_line    |         12 |       74.99 | 1.36e-11  |        0.0084 |
| furnished     |          2 |       90.49 | 1.86e-21  |        0.0034 |
| pets_allowed  |          2 |       10.44 | 0.00124   |        0.0005 |

### 5.3. Multicolinealidad entre Predictores Numéricos (VIF - Excluyendo Fuga)

| variable           |   vif |   r2_with_others |
|:-------------------|------:|-----------------:|
| to_center_km       | 7.777 |           0.8714 |
| metro_distance_min | 5.498 |           0.8181 |
| total_area         | 4.546 |           0.78   |
| rooms              | 4.545 |           0.78   |
| lon                | 2.051 |           0.5124 |
| lat                | 2.014 |           0.5034 |
| total_floors       | 1.545 |           0.3527 |
| floor              | 1.545 |           0.3526 |
| deposit_months     | 1     |           0.0004 |
| building_year      | 1     |           0.0002 |

## 6. Diagnóstico de Fuga de Información (Data Leakage), Anomalías de Dominio y Sesgos

1. **Fuga de Información Directa (`rent_per_sqm`) — CRÍTICO (Hecho Observado):** `rent_per_sqm` presenta una correlación lineal de $r = 0.893$ con el objetivo y equivale exactamente a `monthly_rent_rub / total_area` (dentro de una tolerancia de redondeo del $\pm 2\%$) en el **100.0%** de los registros. **Acción obligatoria:** Eliminar `rent_per_sqm` antes del entrenamiento; no está disponible al momento de predecir el precio de un inmueble nuevo.
2. **Columna de Auditoría (`dq_flags`) y `id` — CRÍTICO:** `id` es un identificador secuencial sin valor predictivo real y `dq_flags` fue creada durante la limpieza. **Acción obligatoria:** Excluir ambas del conjunto de features.
3. **Inconsistencia Semántica Severa en Variables de Metro (`metro_station`, `metro_line`, `metro_distance_min`) — ALTO RIESGO:**
   - **Hecho observado 1:** Cada estación (`metro_station`) aparece distribuida en una mediana de **11 distritos administrativos (`okrug`) distintos** (de los 11 totales de Moscú), lo cual es geográficamente imposible dado que una estación física tiene coordenadas únicas.
   - **Hecho observado 2:** `metro_distance_min` llega hasta **966 minutos (16.1 horas)**, el **45.0%** de los registros supera los 120 minutos, y guarda una colinealidad de $r = 0.904$ con `to_center_km`.
   - **Hipótesis:** En la generación o captura del dataset, `metro_station`/`metro_line` fueron asignadas sin restricción geográfica al polígono del `district`, y `metro_distance_min` actúa como un proxy ruidoso de `to_center_km` (o está expresado en otra unidad/escala).
   - **Recomendación:** Excluir `metro_station` y `metro_line` (o evaluar mediante ablación controlada en validación) para evitar introducir ruido espurio, y analizar si `metro_distance_min` aporta señal incremental real sobre `to_center_km` (VIF elevado).
4. **Dimensión Temporal y Prevención de Fuga del Futuro (`date_posted`: `2021-01-01` a `2026-04-29`):**
   - **Hecho observado:** Los anuncios abarcan más de 5 años (enero 2021 a abril 2026). En mercados inmobiliarios sujetos a inflación y cambios macroeconómicos, el nivel nominal en rublos cambia con el tiempo.
   - **Recomendación de Split (Prevención de Fuga Temporal):** No usar partición aleatoria simple si el objetivo de negocio es predecir alquileres futuros. Implementar un **Time-Based Split** (entrenamiento con datos históricos, validación en ventana intermedia y prueba en los últimos meses de 2025–2026), y ajustar cualquier transformador (imputadores, escaladores, Target Encoding de `district`) **exclusivamente sobre el subconjunto de entrenamiento (`X_train`, `y_train`)** dentro de un `Pipeline` o `ColumnTransformer`.

## 7. Viabilidad para Machine Learning y Familias de Modelos Candidatas

| Familia de Modelos | Ejemplos Concretos | Justificación Basada en Hallazgos del EDA | Preprocesamiento Requerido |
|---|---|---|---|
| **1. Modelos Lineales Regularizados (Baseline Interpretable)** | `Ridge`, `Lasso`, `ElasticNet`, `HuberRegressor` (con `TransformedTargetRegressor` en escala `log1p`) | Permiten auditar elasticidades económicas claras (impacto por m², caída exponencial por km al centro). Sensibles a multicolinealidad (`to_center_km` vs `metro_distance_min`, `total_area` vs `rooms`) y a asimetría del target. | Transformación `log1p` del objetivo, eliminación de variables colineales (VIF > 5), `OneHotEncoder` en categóricas de baja cardinalidad, `TargetEncoder` con CV en `district`, escalado robusto (`RobustScaler`) y términos polinómicos/interacciones (`total_area * to_center_km`). |
| **2. Árboles de Decisión Potenciados por Gradiente (GBDT — Familia Principal)** | `LightGBM`, `CatBoost`, `XGBoost` | Capturan de forma nativa relaciones no lineales (gradiente espacial `lat`/`lon` y `to_center_km`), interacciones de orden superior (`total_area` × `okrug` × `renovation`) y son robustos a escalas dispares y colinealidad moderada. `CatBoost` destaca por el manejo nativo de `district` y `okrug` sin explosión dimensional. | Mínimo escalado requerido; creación de features de dominio (`floor_ratio = floor / total_floors`, `building_age = posting_year - building_year`, coordenadas polares al Kremlin) y función de pérdida `RMSE` sobre `log1p(y)` o `Gamma`/`Huber`. |
| **3. Ensambles de Bagging No Paramétricos** | `RandomForestRegressor`, `ExtraTreesRegressor` | Útiles como punto de comparación robusto frente a varianza y valores extremos sin supuestos paramétricos, aunque no extrapolan tendencias inflacionarias temporales fuera del rango observado. | Codificación ordinal/target de alta cardinalidad (`district`) e indicadores de antigüedad y fecha. |

## 8. Catálogo de Gráficos Generados (`reports/figures/`)

| Archivo | Finalidad Analítica |
|---|---|
| `reports/figures/01_target_distribution.png` | Evaluar asimetría y cola derecha de `monthly_rent_rub` y verificar si la transformación logarítmica estabiliza la varianza. |
| `reports/figures/02_numeric_distributions.png` | Inspeccionar multimodalidad, rangos, discretización y sesgos de las variables numéricas predictoras. |
| `reports/figures/03_categorical_analysis.png` | Evaluar el poder discriminativo de las categorías clave (`okrug`, `renovation`, `building_type`, `furnished`) sobre la mediana del alquiler. |
| `reports/figures/04_correlation_matrices.png` | Detectar relaciones lineales y monótonas con el objetivo, así como colinealidad severa entre predictores. |
| `reports/figures/05_key_bivariate_interactions.png` | Visualizar la no-linealidad del decaimiento del precio con `to_center_km` y la interacción multiplicativa con `total_area`. |
| `reports/figures/06_leakage_and_metro_diagnostics.png` | Demostrar empíricamente la fuga exacta en `rent_per_sqm` y la colinealidad/escala anómala de `metro_distance_min`. |
| `reports/figures/07_temporal_trends.png` | Analizar deriva temporal (drift de precios 2021-2026), volumen mensual de oferta y estacionalidad para definir el esquema de validación. |
| `reports/figures/08_spatial_distribution.png` | Confirmar la estructura monocéntrica de precios en Moscú y validar la coherencia espacial de los distritos (`okrug`). |
