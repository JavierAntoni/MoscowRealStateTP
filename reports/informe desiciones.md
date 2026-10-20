# Bitácora de Decisiones y Arquitectura del Modelo

## 1. Inventario de Variables

| Variable Original | Acción | Justificación de Negocio/Estadística |
|---|---|---|
| `id` | Descartar | Identificador único, sin información inmobiliaria predictiva útil. |
| `date_posted` | Transformar | Deriva año, estacionalidad mensual y referencia temporal del anuncio. |
| `district` | Transformar | Target Encoding controlado; captura diferencias de alquiler entre distritos. |
| `okrug` | Descartar | Redundancia con `district`: V de Cramér = 0,997. |
| `lat` | Incluir | Captura gradientes geográficos residuales. |
| `lon` | Incluir | Complementa la ubicación con el gradiente este-oeste. |
| `total_area` | Transformar | Splines + PCA representan su relación no lineal con el alquiler. |
| `rooms` | Incluir | Describe la capacidad y distribución del departamento. |
| `floor` | Transformar | Se deriva `floor_ratio`, relativo a las plantas del edificio. |
| `total_floors` | Incluir | Describe la altura y tipología edilicia. |
| `building_year` | Transformar | Se deriva `building_age` para representar la antigüedad. |
| `building_type` | Transformar | One-Hot Encoding; distingue tipologías arquitectónicas. |
| `renovation` | Transformar | One-Hot Encoding; refleja el estado y la calidad interior. |
| `furnished` | Incluir | Captura la asociación del amoblamiento con el alquiler. |
| `pets_allowed` | Incluir | Representa una condición contractual relevante. |
| `deposit_months` | Incluir | Incorpora las condiciones de garantía del alquiler. |
| `metro_station` | Descartar | Presenta asignaciones geográficas inconsistentes. |
| `metro_line` | Descartar | Presenta inconsistencias geográficas similares a las estaciones. |
| `metro_distance_min` | Transformar | Splines + PCA; la ablación mostró mejora predictiva. |
| `to_center_km` | Transformar | Splines + PCA capturan una relación no lineal con la distancia. |
| `monthly_rent_rub` | Transformar | Es el objetivo; `log1p` reduce la asimetría y comprime valores altos. |
| `rent_per_sqm` | Descartar | Fuga de información: se deriva del alquiler objetivo y la superficie. |
| `dq_flags` | Descartar | Variable de auditoría, no una característica válida para tasar. |

**Variables derivadas:** `posting_year`, `posting_month_sin/cos`, `building_age`, `is_presale_or_new`, `area_per_room`, `floor_ratio`, `is_first_floor` e `is_top_floor`. El preprocesador final utiliza **57 predictores transformados**.

## 2. Anatomía de la Fórmula Ganadora

### La fórmula completa en criollo

- **Intercepto (`b0 = 10.744228`):** punto de partida del alquiler en escala logarítmica, cuando los predictores están en sus valores de referencia y las categorías en sus bases.
- A ese punto de partida se suman los aportes de **57 predictores**, incluidas variables inmobiliarias, categorías codificadas y curvas spline.
- El resultado vuelve a rublos mediante la transformación inversa y la corrección de Duan (**× 1,003500**).

### Términos interpretables

| Término | Qué es en criollo | Qué aporta al alquiler |
|---|---|---:|
| `furnished` | Departamento amoblado | +8,15 % |
| `pets_allowed` | Se permiten mascotas | −2,62 % |
| `lat` | Posición norte-sur | +22,36 % por grado |
| `lon` | Posición este-oeste | −6,71 % por grado |
| `rooms` | Cantidad de habitaciones | +0,19 % |
| `building_age` | Antigüedad del edificio | +0,01 % |
| `floor_ratio` | Piso relativo a la altura del edificio | +0,11 % |
| `deposit_months` | Meses de depósito exigidos | −0,02 % |
| `total_floors` | Cantidad total de plantas | Aproximadamente 0 % |
| `renovation_designer` | Renovación de diseño | +36,83 % |
| `renovation_euro` | Renovación estilo europeo | +15,79 % |
| `renovation_no_renovation` | Sin renovación | −10,15 % |
| `building_type_stalin` | Edificio de estilo estalinista | +13,63 % |
| `building_type_khrushchev` | Edificio tipo Khrushchev | −19,25 % |
| `building_type_panel` | Edificio prefabricado tipo panel | −12,06 % |
| `building_type_monolith` | Edificio monolítico | +5,63 % |
| `building_type_modern_panel` | Edificio de panel moderno | −4,43 % |

*Los porcentajes son efectos ceteris paribus respecto de las referencias del modelo. Las categorías base omitidas —edificio de ladrillo (`brick`) y renovación cosmética (`cosmetic`)— quedan representadas en la codificación junto con el intercepto.*

### Bloques que no se interpretan término a término

| Bloque | Qué es en criollo | Importancia estadística |
|---|---|---|
| `target_enc district` | Codificación del distrito según información del alquiler, ajustada para evitar fuga | **#1: +0,258** de efecto estandarizado |
| `spline total_area` | Tramos que representan la relación curva entre superficie y precio | `pca0`, **#2: −0,183** |
| `spline to_center_km` | Tramos que representan cómo cambia el alquiler con la distancia al centro | `pca3`, **#3: +0,127** |
| `spline metro_distance_min` | Tramos que resumen la relación entre accesibilidad y alquiler | No se dispone aquí del puesto y efecto estandarizado exactos |
| `time` | Componente temporal que captura patrones de evolución del mercado | `pca0`, **#9: −0,072** |

**Importante:** un componente PCA individual no se traduce directamente a rublos ni a porcentajes; combina información transformada de varios tramos de una curva.

### Frase para el Tribunal

> “El modelo parte de un intercepto en escala logarítmica, suma los aportes de 57 predictores —incluidas categorías y curvas spline— y transforma el resultado nuevamente a rublos con la corrección de Duan, buscando una estimación precisa y económicamente interpretable.”

## 3. El Modelo Ganador

`LinearRegression` (OLS) obtuvo el menor RMSE de validación: **0,085614 en escala logarítmica**. Ridge, Lasso y ElasticNet rindieron ligeramente peor; las penalizaciones óptimas de Lasso y ElasticNet fueron mínimas. OLS ofrece una solución simple y auditable, pero el teorema de Gauss-Markov solo respalda sus propiedades bajo determinados supuestos; la elección se basa principalmente en el rendimiento observado.

## 4. Blindaje: Decisiones Polémicas

| Decisión | Defensa ante el tribunal |
|---|---|
| **Navaja de Ockham: ¿por qué OLS y no Random Forest/XGBoost?** | Se priorizó una arquitectura auditable, con coeficientes analizables y relaciones no lineales capturadas mediante transformaciones. No se afirma que OLS sea universalmente superior a los modelos de árboles. |
| **Outliers: ¿por qué no borrar 936 moderados y 212 severos?** | Son observaciones de la muestra, no necesariamente errores. `log1p` comprime los precios altos sin eliminarlos; filtrarlos habría descartado parte de la realidad del mercado observado. |
| **Variable sucia: ¿por qué mantener `metro_distance_min` con valores de hasta 966?** | La ablación mostró mejora predictiva al conservarla transformada. Se usa como descriptor continuo de accesibilidad, no como minutos reales de caminata; su semántica sigue siendo una limitación. |