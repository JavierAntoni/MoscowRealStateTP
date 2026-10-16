# Informe de Preprocesamiento para Machine Learning — Alquileres en Moscú (Etapa 03)

| Metadato | Valor |
|---|---|
| Fecha de generación | `2026-10-16 15:51:36` |
| Archivo de entrada (solo lectura) | `data/rentals_clean.csv` |
| SHA-256 verificado (inmutable) | `e43378a0de55cdc2320b76f273ef47e1158a41e7fb0d06da0dc34666a5cc1a71` |
| Variable objetivo (`y`) | `monthly_rent_rub` (alquiler mensual en RUB) |
| Filas totales procesadas | `20000` |
| Dimensiones post-preprocesamiento (`linear`) | `170` columnas |
| Dimensiones post-preprocesamiento (`tree`) | `170` columnas |

## 1. Integridad de Entrada y Separación de Variables (`X` / `y`)

- **Verificación de inmutabilidad:** El archivo `data/rentals_clean.csv` se leyó en modo solo lectura comprobando que su hash SHA-256 coincide exactamente con la salida de `01_cleaning.py` y `02_eda.py` (`e43378a0de55cdc2320b76f273ef47e1158a41e7fb0d06da0dc34666a5cc1a71`).
- **Variable objetivo (`y`):** `monthly_rent_rub` fue separada antes de cualquier transformación. En el conjunto de entrenamiento presenta una asimetría de `1.821` en escala original y de `-0.133` bajo transformación `np.log1p`.

## 2. Variables Excluidas y Justificación Anti-Fuga (Data Leakage)

| Columna Excluida | Categoría de Exclusión | Justificación Basada en el EDA |
|---|---|---|
| `rent_per_sqm` | **Fuga directa del objetivo** | Equivale exactamente a `monthly_rent_rub / total_area` ($\pm 2\%$) en el 100.0% de los registros ($r = 0.8933$, Figura 06). No existe al momento de predecir un inmueble nuevo. |
| `id` | **Identificador único** | Secuencia alfanumérica (`R000001`..`R020000`) con cardinalidad 100% sin valor predictivo; se conserva por separado solo para trazabilidad de filas. |
| `dq_flags` | **Artefacto de auditoría** | Creada en `01_cleaning.py` (98.31% nula). Tanto el informe de limpieza como el EDA prohíben su uso como predictor. |
| `metro_station`, `metro_line` | **Corrupción semántica severa** | Cada estación aparece distribuida en una mediana de 11 de los 11 `okrug` de Moscú, lo cual es geográficamente imposible e introduce ruido espurio. |
| `metro_distance_min` | **Colinealidad severa y anomalía de escala** | Presenta $r = 0.904$ ($	ext{VIF} = 5.498$) con `to_center_km`, alcanza 966 minutos y supera 120 minutos en el 45.0% de las filas (Figura 06). |

## 3. Estrategia de División Temporal (Train / Validation / Test)

Dado que `date_posted` abarca desde `2021-01-01` hasta `2026-04-29` y la Figura 07 demuestra una tendencia inflacionaria nominal sostenida, se aplicó una **división cronológica estricta (70% / 15% / 15%)** ajustada a días completos para evitar fuga temporal del futuro hacia el pasado:

| Partición | Filas (`n`) | Proporción (%) | Fecha Inicial (`min`) | Fecha Final (`max`) |
|---|---:|---:|---|---|
| **Entrenamiento (`Train`)** | `14009` | `70.05%` | `2021-01-01` | `2024-09-06` |
| **Validación (`Validation`)** | `2991` | `14.95%` | `2024-09-07` | `2025-06-29` |
| **Prueba (`Test`)** | `3000` | `15.00%` | `2025-06-30` | `2026-04-29` |

- **Control de registros relacionados:** Se verificaron las coordenadas exactas `(lat, lon)` entre particiones (`Train ∩ Val = 0`, `Train ∩ Test = 1`, `Val ∩ Test = 0`).

## 4. Ingeniería de Variables y Transformaciones por Familia de Modelos

1. **Ingeniería de dominio sin estado (`DomainFeatureEngineer`):**
   - Extracción temporal desde `date_posted`: `posting_year`, `posting_month_sin`, `posting_month_cos` (eliminando luego la cadena `date_posted`).
   - Resolución de la anomalía `building_year > posting_year` (339 filas): creación de `is_presale_or_new` (0/1) y `building_age = max(0, posting_year - building_year)`.
   - Ratios estructurales: `area_per_room = total_area / rooms`, `floor_ratio = floor / total_floors`, `is_first_floor` e `is_top_floor`.
2. **Columnas por rama dentro del `ColumnTransformer`:**
   - **Numéricas (15):** `total_area, rooms, floor, total_floors, building_year, deposit_months, to_center_km, lat, lon, posting_year, posting_month_sin, posting_month_cos, building_age, area_per_room, floor_ratio`
   - **Binarias (5):** `furnished, pets_allowed, is_presale_or_new, is_first_floor, is_top_floor`
   - **Categóricas (4):** `okrug, district, building_type, renovation`
3. **Escalado diferenciado según el modelo:**
   - **Pipeline `linear` (`Ridge`, `Lasso`, `ElasticNet`, `HuberRegressor`):** Imputación por mediana + **`RobustScaler`** en las 15 variables numéricas (mitiga la asimetría y cola pesada de `to_center_km` y `floor` observadas en la Figura 02), imputación por moda en binarias (`{0, 1}` sin escalar) y `OneHotEncoder(handle_unknown="ignore", drop="if_binary")` en categóricas.
   - **Pipeline `tree` (`LightGBM`, `XGBoost`, `CatBoost`, `RandomForest`, `ExtraTrees`):** Imputación por mediana **sin escalado** en las 15 variables numéricas (los árboles son invariantes a escalados monótonos y conservan las unidades físicas originales), imputación por moda en binarias y `OneHotEncoder(handle_unknown="ignore")` en categóricas.
4. **Garantía anti-fuga en el ajuste:** Todos los estimadores (`SimpleImputer`, `RobustScaler`, `OneHotEncoder`) se ajustaron **exclusivamente sobre `X_train`**, aplicando únicamente `.transform()` sobre `X_val` y `X_test`.
