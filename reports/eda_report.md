# Informe de Análisis Exploratorio de Datos (EDA) — Alquileres en Moscú (Etapa 02)

| Metadato | Valor |
|---|---|
| Fecha de generación | `2026-10-20 09:29:25` |
| Archivo analizado (solo lectura) | `data\rentals_clean.csv` |
| SHA-256 verificado (inmutable) | `e43378a0de55cdc2320b76f273ef47e1158a41e7fb0d06da0dc34666a5cc1a71` |
| Dimensiones del dataset | `20000` filas × `23` columnas |
| Variable objetivo | `monthly_rent_rub` (alquiler mensual en RUB) |

## 1. Resumen y Contraste de la Limpieza Inicial (`01_cleaning.py`)

- **Informe de limpieza previo no disponible;** el diagnóstico se basa en la inspección directa de `rentals_clean.csv`.
- **Reglas de la etapa previa y su efecto sobre el EDA:**
  1. Normalización de texto y conversión de tipos sin pérdida de filas (0 registros eliminados).
  2. No se aplicaron filtros estadísticos de valores extremos (IQR, percentiles, Cook) sobre `monthly_rent_rub` ni predictores.
  3. Anomalías preservadas en `dq_flags` (`building_year_after_posting` en `339` filas), **no utilizable como feature**.
  4. Se conservaron variables redundantes (`rent_per_sqm`) y de semántica dudosa (`metro_*`), cuyo diagnóstico se realiza aquí.

## 2. Estructura del Dataset y Calidad de Datos

### 2.1. Inventario de Variables, Tipos y Cardinalidad

| Columna | Tipo en Pandas | Rol Analítico | Nulos (%) | Valores Únicos | Descripción |
|---|---|---|---:|---:|---|
| `id` | `str` | `id` | 0.00% | 20,000 | Identificador único del anuncio |
| `date_posted` | `str` | `temporal` | 0.00% | 1,945 | Fecha de publicación (ISO YYYY-MM-DD) |
| `district` | `str` | `categorical` | 0.00% | 129 | Distrito municipal (raion) de Moscú |
| `okrug` | `str` | `categorical` | 0.00% | 11 | Distrito administrativo (okrug) de Moscú |
| `lat` | `float64` | `spatial` | 0.00% | 15,985 | Latitud geográfica (grados decimales) |
| `lon` | `float64` | `spatial` | 0.00% | 16,678 | Longitud geográfica (grados decimales) |
| `total_area` | `float64` | `numeric_continuous` | 0.00% | 695 | Superficie total del inmueble (m²) |
| `rooms` | `int64` | `numeric_discrete` | 0.00% | 4 | Cantidad de habitaciones / ambientes |
| `floor` | `int64` | `numeric_discrete` | 0.00% | 22 | Piso en el que se ubica la unidad |
| `total_floors` | `int64` | `numeric_discrete` | 0.00% | 5 | Cantidad total de pisos del edificio |
| `building_year` | `int64` | `numeric_discrete` | 0.00% | 90 | Año de construcción del edificio |
| `building_type` | `str` | `categorical` | 0.00% | 6 | Material o tipología constructiva |
| `renovation` | `str` | `categorical` | 0.00% | 4 | Estado de renovación interior |
| `furnished` | `bool` | `boolean` | 0.00% | 2 | Indica si se alquila amoblado |
| `pets_allowed` | `bool` | `boolean` | 0.00% | 2 | Indica si se admiten mascotas |
| `deposit_months` | `int64` | `numeric_discrete` | 0.00% | 3 | Meses de depósito en garantía |
| `metro_station` | `str` | `categorical` | 0.00% | 104 | Estación de metro asignada al anuncio |
| `metro_line` | `str` | `categorical` | 0.00% | 12 | Línea de metro de la estación |
| `metro_distance_min` | `int64` | `numeric_continuous` | 0.00% | 852 | Distancia reportada al metro (en minutos) |
| `to_center_km` | `float64` | `numeric_continuous` | 0.00% | 4,266 | Distancia en línea recta al centro de Moscú (km) |
| `monthly_rent_rub` | `int64` | `target` | 0.00% | 1,576 | Variable objetivo: alquiler mensual (RUB) |
| `rent_per_sqm` | `int64` | `leakage_candidate` | 0.00% | 2,335 | Precio de alquiler por m² (RUB/m²) |
| `dq_flags` | `str` | `audit` | 98.31% | 1 | Banderas de auditoría generadas en 01_cleaning.py |

### 2.2. Chequeos de Integridad y Consistencia Lógica

- **Duplicados exactos (todas las columnas):** `0` filas.
- **Duplicados excluyendo `id`:** `0` filas.
- **Columnas constantes (varianza cero):** `Ninguna`.
- **Fallos de coerción numérica:** `Ninguno (0 fallos)`.
- **Inconsistencia vertical (`floor > total_floors`):** `0` filas.
- **Pisos no positivos (`floor <= 0`):** `0` filas.
- **Anomalía temporal (`building_year > año(date_posted)`):** `339` filas (1.70% del total).
- **Coordenadas `(lat, lon)` idénticas en registros distintos:** `1` grupo(s) / `2` filas.

## 3. Estadística Descriptiva

### 3.1. Variables Numéricas (paramétrica y no paramétrica)

```
          variable  n_valid  missing_pct  coerced_invalid      mean       std    cv      min       p01       p05       p25    median       p75        p95        p99        max       iqr       mad  skewness  kurtosis normaltest_p  outliers_1_5_iqr  outliers_1_5_pct  outliers_3_0_iqr
  monthly_rent_rub    20000          0.0                0 52432.855 30003.240 0.572 6800.000 11600.000 17300.000 32800.000 45800.000 64400.000 109705.000 161102.000 300700.000 31600.000 15000.000     1.820     5.565     0.00e+00               936              4.68               212
      rent_per_sqm    20000          0.0                0   945.461   480.537 0.508  148.000   246.000   329.000   644.000   845.000  1135.000   1873.000   2680.050   4264.000   491.000   234.000     1.593     4.020     0.00e+00              1001              5.00               233
        total_area    20000          0.0                0    55.387    12.740 0.230   22.100    33.199    37.700    45.700    53.800    63.300     79.300     89.200    104.600    17.600     8.700     0.577    -0.087     6.36e-54               174              0.87                 0
             rooms    20000          0.0                0     1.794     0.865 0.482    1.000     1.000     1.000     1.000     2.000     2.000      3.000      4.000      4.000     1.000     1.000     0.847    -0.115    1.34e-107               945              4.72                 0
             floor    20000          0.0                0     6.608     4.830 0.731    1.000     1.000     1.000     3.000     5.000     9.000     17.000     21.000     22.000     6.000     3.000     1.074     0.619    1.51e-161               570              2.85                 0
      total_floors    20000          0.0                0    12.191     5.692 0.467    5.000     5.000     5.000     9.000     9.000    17.000     22.000     22.000     22.000     8.000     4.000     0.437    -1.034    3.67e-290                 0              0.00                 0
     building_year    20000          0.0                0  1987.823    22.500 0.011 1935.000  1939.000  1954.000  1968.000  1988.000  2008.000   2021.000   2024.000   2024.000    40.000    20.000    -0.101    -1.141     0.00e+00                 0              0.00                 0
    deposit_months    20000          0.0                0     1.101     0.541 0.491    0.000     0.000     0.000     1.000     1.000     1.000      2.000      2.000      2.000     0.000     0.000     0.072     0.275     1.64e-04              6053             30.27              6053
metro_distance_min    20000          0.0                0   152.385   167.803 1.101    0.000     2.000     6.000    33.000   103.000   198.000    543.000    776.000    966.000   165.000    77.000     1.963     4.039     0.00e+00              1416              7.08               488
      to_center_km    20000          0.0                0    16.360    12.801 0.782    0.020     1.670     3.380     9.040    13.230    18.030     46.330     65.950     71.060     8.990     4.400     2.121     4.828     0.00e+00              1905              9.53              1102
               lat    20000          0.0                0    55.713     0.144 0.003   55.230    55.271    55.405    55.636    55.740    55.818     55.888     55.956     56.013     0.182     0.090    -0.980     0.908    1.60e-138               672              3.36                 0
               lon    20000          0.0                0    37.542     0.181 0.005   36.702    36.926    37.206    37.465    37.564    37.652     37.805     37.861     37.908     0.187     0.093    -1.224     2.969    1.15e-248               743              3.71               161
```

### 3.2. Variables Categóricas (cardinalidad, frecuencias y categorías raras)

```
     variable dtype  n_valid  missing_pct  n_unique                top_category  top_pct              least_category  least_pct  rare_categories_lt_1pct
        okrug   str    20000         0.00        11                        SVAO    12.94                       ZelAO       0.69                        1
     district   str    20000         0.00       129                Dorogomilovo     0.95                 Ivanovskoye       0.61                      129
building_type   str    20000         0.00         6                       panel    24.84                      stalin       4.88                        0
   renovation   str    20000         0.00         4                    cosmetic    45.17               no_renovation       9.93                        0
    furnished  bool    20000         0.00         2                        True    70.22                       False      29.79                        0
 pets_allowed  bool    20000         0.00         2                       False    69.72                        True      30.29                        0
   metro_line   str    20000         0.00        12         Kaluzhsko-Rizhskaya    14.14                         MCC       1.47                        0
metro_station   str    20000         0.00       104                    Sviblovo     1.93              Tekstilshchiki       0.54                       73
     dq_flags   str      339        98.31         1 building_year_after_posting   100.00 building_year_after_posting     100.00                        0
```

## 4. Análisis de la Variable Objetivo (`monthly_rent_rub`)

- **Rango y tendencia central:** de **6,800** a **300,700 RUB**, mediana **45,800 RUB** (P25 = 32,800; P75 = 64,400; IQR = 31,600) y media **52,433 RUB** (DE = 30,003).
- **Asimetría y cola derecha:** skewness = **1.820**, kurtosis = **5.565**; `936` observaciones (4.68%) superan el umbral de Tukey 1.5×IQR.
- **Efecto de `log1p`:** la asimetría cae a **-0.136** (curtosis = **0.216**), estabilizando la varianza condicional.
- **Recomendación de modelado:** usar $\ln(1+y)$ como respuesta con corrección de sesgo de retransformación (exp/ factor de Duan–smearing), o pérdidas robustas (Huber, MAE, Gamma/Tweedie).

## 5. Relaciones con el Objetivo, Asociaciones y Multicolinealidad

### 5.1. Correlaciones Numéricas con `monthly_rent_rub`

```
          variable  pearson_r  spearman_rho  abs_spearman
      rent_per_sqm     0.8933        0.8826        0.8826
      to_center_km    -0.5314       -0.6332        0.6332
metro_distance_min    -0.5123       -0.6329        0.6329
        total_area     0.4108        0.4376        0.4376
             rooms     0.3583        0.3785        0.3785
               lat     0.2738        0.2463        0.2463
               lon     0.2225        0.1926        0.1926
     building_year     0.0503        0.0663        0.0663
             floor    -0.0055       -0.0047        0.0047
    deposit_months     0.0011        0.0037        0.0037
      total_floors    -0.0024        0.0002        0.0002
```

### 5.2. Asociación Categórica → Objetivo (Kruskal‑Wallis y $\eta^2$)

```
     variable  n_unique  kruskal_H   p_value  eta_squared
     district       129   12289.91  0.00e+00       0.6044
        okrug        11    9853.63  0.00e+00       0.4916
   renovation         4     881.52 9.04e-191       0.0427
metro_station       104     593.91  4.47e-70       0.0426
building_type         6     671.05 8.93e-143       0.0289
   metro_line        12      74.99  1.36e-11       0.0084
    furnished         2      90.49  1.86e-21       0.0034
 pets_allowed         2      10.44  1.24e-03       0.0005
```

### 5.3. Asociación entre Categóricas (V de Cramér corregida)

```
     variable  okrug  district  building_type  renovation
        okrug 1.0000     0.997            0.0      0.0083
     district 0.9970     1.000            0.0      0.0000
building_type 0.0000     0.000            1.0      0.0000
   renovation 0.0083     0.000            0.0      1.0000
```

### 5.4. Multicolinealidad entre Predictores Numéricos (VIF, sin variables de fuga)

```
          variable   vif  r2_with_others
      to_center_km 9.046          0.8895
metro_distance_min 5.561          0.8202
        total_area 4.918          0.7967
             rooms 4.546          0.7800
               lon 2.295          0.5644
  monthly_rent_rub 2.131          0.5308
               lat 2.119          0.5280
      total_floors 1.545          0.3527
             floor 1.545          0.3526
     building_year 1.005          0.0051
    deposit_months 1.000          0.0004
```

- Número de condición de la matriz de correlación: **43.84468698275527** (inversa pseudoinversa usada: `False`).
- Predictores con **VIF > 5** (colinealidad problemática): ['to_center_km', 'metro_distance_min'].

## 6. Diagnóstico de Fuga de Información, Anomalías de Dominio y Sesgos

1. **Fuga directa (`rent_per_sqm`) — CRÍTICO:** r = 0.893 con el objetivo y equivale a `monthly_rent_rub / total_area` (±2%) en el **100.0%** de los registros. **Acción obligatoria: excluir del entrenamiento.**
2. **`id` y `dq_flags` — CRÍTICO:** identificador secuencial y columna de auditoría; **excluir ambas del conjunto de features**.
3. **Semántica de `metro_*` — ALTO RIESGO:**
   - Cada estación aparece en una mediana de **11 de 11 `okrug`**, lo cual es geográficamente imposible para una estación física.
   - `metro_distance_min` alcanza **966 min**, el **45.0%** supera 120 min, y su correlación con `to_center_km` es **r = 0.904**.
   - **Hipótesis:** asignación estación–ubicación sin restricción geográfica; `metro_distance_min` actúa como proxy ruidoso de `to_center_km` o está expresado en otra unidad.
   - **Recomendación:** excluir `metro_station`/`metro_line` (o ablación controlada) y evaluar si `metro_distance_min` aporta señal incremental sobre `to_center_km` (VIF elevado).
4. **Dimensión temporal (`date_posted`: 2021-01-01 → 2026-04-29):** más de 5 años de datos nominales en rublos sujetos a inflación.
   - **Split recomendado:** **Time‑Based Split** (entrenamiento histórico, validación intermedia, prueba en los últimos meses), y ajuste de transformadores **exclusivamente sobre `X_train`/`y_train`** dentro de un `Pipeline`/`ColumnTransformer`.

## 7. Conjunto de Features Recomendado

| Columna | Decisión | Justificación |
|---|---|---|
| `id` | ❌ EXCLUIR | Identificador sin valor predictivo |
| `date_posted` | ⚠️ DERIVAR | Usar para split temporal y `building_age`; no como feature cruda |
| `district` | ✅ INCLUIR | Predictora válida |
| `okrug` | ✅ INCLUIR | Predictora válida |
| `lat` | ✅ INCLUIR | Predictora válida |
| `lon` | ✅ INCLUIR | Predictora válida |
| `total_area` | ✅ INCLUIR | Predictora válida |
| `rooms` | ✅ INCLUIR | Predictora válida |
| `floor` | ✅ INCLUIR | Predictora válida |
| `total_floors` | ✅ INCLUIR | Predictora válida |
| `building_year` | ✅ INCLUIR | Predictora válida |
| `building_type` | ✅ INCLUIR | Predictora válida |
| `renovation` | ✅ INCLUIR | Predictora válida |
| `furnished` | ✅ INCLUIR | Predictora válida |
| `pets_allowed` | ✅ INCLUIR | Predictora válida |
| `deposit_months` | ✅ INCLUIR | Predictora válida |
| `metro_station` | ⚠️ EXCLUIR / ablación | Asignación geográfica inconsistente (ver §6.3) |
| `metro_line` | ⚠️ EXCLUIR / ablación | Asignación geográfica inconsistente (ver §6.3) |
| `metro_distance_min` | ⚠️ CONDICIONAL | Proxy de `to_center_km`; usar solo si aporta señal incremental (VIF) |
| `to_center_km` | ✅ INCLUIR | Predictora válida |
| `monthly_rent_rub` | **Objetivo** | Variable a predecir |
| `rent_per_sqm` | ❌ EXCLUIR | Fuga de información: derivada del objetivo |
| `dq_flags` | ❌ EXCLUIR | Columna de auditoría creada en limpieza |

## 8. Viabilidad para Machine Learning y Familias de Modelos Candidatas

| Familia | Ejemplos | Justificación (hallazgos del EDA) | Preprocesamiento |
|---|---|---|---|
| **1. Lineales regularizados (baseline interpretable)** | `Ridge`, `Lasso`, `ElasticNet`, `HuberRegressor` + `TransformedTargetRegressor(log1p)` | Elasticidades auditables; sensibles a colinealidad (`to_center_km` vs `metro_distance_min`, `total_area` vs `rooms`) y a la asimetría del objetivo | `log1p` del target, eliminar colineales (VIF>5), `OneHotEncoder` en categóricas de baja cardinalidad, `TargetEncoder` con CV en `district`, `RobustScaler`, interacciones `total_area * to_center_km` |
| **2. GBDT (familia principal)** | `LightGBM`, `CatBoost`, `XGBoost` | Capturan no linealidad (gradiente espacial `lat`/`lon`, `to_center_km`) e interacciones de orden superior (`total_area` × `okrug` × `renovation`); robustos a escalas dispares | Mínimo escalado; features de dominio (`floor_ratio = floor/total_floors`, `building_age`, coordenadas polares al centro); pérdida `RMSE` sobre `log1p(y)` o `Gamma`/`Huber` |
| **3. Bagging no paramétrico** | `RandomForestRegressor`, `ExtraTreesRegressor` | Comparación robusta frente a varianza y extremos; no extrapola tendencias temporales fuera del rango observado | Codificación ordinal/target de alta cardinalidad e indicadores de antigüedad/fecha |

## 9. Catálogo de Gráficos Generados (`reports/figures/`)

| Archivo | Finalidad Analítica |
|---|---|
| `reports/figures/01_target_distribution.png` | Evaluar asimetría del objetivo y si log1p estabiliza la varianza. |
| `reports/figures/02_numeric_distributions.png` | Multimodalidad, rangos, discretización y sesgo de predictores numéricos. |
| `reports/figures/03_categorical_analysis.png` | Poder discriminativo de categorías clave sobre la mediana del alquiler. |
| `reports/figures/04_correlation_matrices.png` | Relaciones lineales/monótonas con el objetivo y colinealidad entre predictores. |
| `reports/figures/05_key_bivariate_interactions.png` | No-linealidad del decaimiento del precio con to_center_km e interacción con total_area. |
| `reports/figures/06_leakage_and_metro_diagnostics.png` | Evidencia empírica de fuga en rent_per_sqm y escala anómala de metro_distance_min. |
| `reports/figures/07_temporal_trends.png` | Drift de precios 2021-2026, volumen mensual y estacionalidad para definir el split temporal. |
| `reports/figures/08_spatial_distribution.png` | Estructura monocéntrica de precios y coherencia espacial de los distritos. |

## 10. Limitaciones de Validez y Advertencias

- El EDA se basa exclusivamente en `rentals_clean.csv` y en el informe de limpieza; **no se verificó la fuente original de los datos**.
- `to_center_km` no documenta el punto de referencia ni la fórmula (haversine/euclidiana): la reproducibilidad del cálculo queda sin confirmar.
- `metro_distance_min` tiene unidades/semántica sin confirmar; sus resultados deben interpretarse con cautela.
- La calidad declarada (0 nulos, 0 duplicados, 0 normalizaciones aplicadas) es inusualmente perfecta para un scraping real: los datos podrían ser sintéticos o pre‑procesados.
- Los valores extremos se conservaron por diseño de la etapa 01; su legitimidad debe evaluarse con conocimiento de dominio antes de tratarlos.
- `dq_flags` es una columna de auditoría generada durante la limpieza: no debe emplearse como feature.