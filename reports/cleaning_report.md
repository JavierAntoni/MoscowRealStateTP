# Informe de limpieza - alquileres en Moscu (etapa 01)

| Item | Valor |
|---|---|
| Generado | 2026-10-16 03:55:03 |
| Fecha de referencia (fechas futuras) | 2026-10-16 |
| Entrada (solo lectura) | `/home/claude/moscow_rentals/data/rentals.csv` (3069209 bytes) |
| SHA-256 entrada (verificado sin cambios) | `92884e2696c3f998b98f432006ca7e95bf47ca2b984f55c72e7d7f2a7154dfdf` |
| Salida limpia | `/home/claude/moscow_rentals/data/rentals_clean.csv` |
| SHA-256 salida | `e43378a0de55cdc2320b76f273ef47e1158a41e7fb0d06da0dc34666a5cc1a71` |
| Deduplicar 'iguales salvo id' | si |

## 1. Conteos

| Concepto | Filas |
|---|---|
| Filas iniciales (filas de datos leidas) | 20000 |
| Filas eliminadas | 0 |
| Filas finales | 20000 |

Verificacion: 20000 - 0 = 20000 = 20000 -> **OK**.

## 2. Operaciones ejecutadas

1. Lectura de todo el CSV como texto (sin inferencia de tipos ni de nulos).
2. Normalizacion de texto: recorte y colapso de espacios; tokens de nulo (`nan`, `None`, `N/A`, `-`, vacio...) -> ausente real.
3. Unificacion de variantes que solo difieren en mayusculas (district, okrug, metro_station, metro_line) y normalizacion de building_type/renovation.
4. Eliminacion de filas totalmente vacias.
5. Validacion y conversion de tipos (booleanos, numericos, enteros, fecha ISO `YYYY-MM-DD`). Lo no convertible pasa a ausente y el registro se conserva.
6. Eliminacion de duplicados REALES (ver seccion 3).
7. Correccion de lat/lon intercambiadas (solo si el par invertido cae en Moscu).
8. Valores fisicamente imposibles (area <= 0, distancias < 0, ...) -> ausente.
9. Senalamiento de anomalias dudosas en `dq_flags` (se conservan sin modificar).
10. Chequeos de consistencia agregados (solo informativos).

| Normalizacion | Celdas |
|---|---|
| Celdas con espacios sobrantes/no separables recortadas | 0 |
| Celdas con token de nulo ('nan', 'None', 'N/A', '-'...) -> ausente | 0 |
| Valores categoricos normalizados (mayusculas/separadores) | 0 |

## 3. Registros eliminados

_Ninguno._

**Ningun registro se elimino por su precio, ni por criterios estadisticos (IQR, residuos, Cook, percentiles).** `monthly_rent_rub` no interviene en ninguna decision de conservar/eliminar filas; ante duplicados se conserva la primera aparicion en el archivo.

## 4. Valores convertidos a ausente (registro conservado)

_Ninguno._

## 5. Correcciones objetivas aplicadas

_Ninguno._

## 6. Anomalias detectadas y CONSERVADAS (columna `dq_flags`)

| Codigo | Filas | Descripcion | Ejemplos |
|---|---|---|---|
| building_year_after_posting | 339 | building_year posterior al anio de date_posted (puede ser obra nueva en preventa) | R000066 (fila 67); R000084 (fila 85); R000189 (fila 190); R000235 (fila 236); R000299 (fila 300) |

`dq_flags` es solo para auditoria/EDA: **no debe usarse como feature**.

## 7. Chequeos de consistencia agregados (informativos)

| Chequeo | Resultado |
|---|---|
| district con mas de un okrug | 0 |
| metro_station con mas de una metro_line | 0 |
| okrugs distintos por estacion (mediana) / okrugs totales | 11 / 11 (104 estaciones) |
| rent_per_sqm ~ monthly_rent_rub / total_area (+-2%) | 100.0% de 20000 filas comparables |
| corr(monthly_rent_rub, rent_per_sqm) | 0.893 |
| corr(metro_distance_min, to_center_km) | 0.904 |
| metro_distance_min > 120 min | 45.0% |
| Registros con identico (lat, lon) pero contenido distinto (CONSERVADOS) | 1 grupos / 2 filas ['R003839, R018015'] |
| Misma unidad fisica (lat, lon, area, piso, pisos) con otra fecha/precio (posible re-publicacion; CONSERVADOS) | 0 grupos / 0 filas |
| Rango de date_posted | 2021-01-01 a 2026-04-29 |

## 8. Valores faltantes por columna

| Columna | Ausentes tras normalizar (todas las filas leidas) | Ausentes en el CSV final |
|---|---|---|
| id | 0 | 0 |
| date_posted | 0 | 0 |
| district | 0 | 0 |
| okrug | 0 | 0 |
| lat | 0 | 0 |
| lon | 0 | 0 |
| total_area | 0 | 0 |
| rooms | 0 | 0 |
| floor | 0 | 0 |
| total_floors | 0 | 0 |
| building_year | 0 | 0 |
| building_type | 0 | 0 |
| renovation | 0 | 0 |
| furnished | 0 | 0 |
| pets_allowed | 0 | 0 |
| deposit_months | 0 | 0 |
| metro_station | 0 | 0 |
| metro_line | 0 | 0 |
| metro_distance_min | 0 | 0 |
| to_center_km | 0 | 0 |
| monthly_rent_rub | 0 | 0 |
| rent_per_sqm | 0 | 0 |

## 9. Rangos observados (descriptivo; NO se filtro nada)

| Columna | n validos | min | mediana | max |
|---|---|---|---|---|
| lat | 20000 | 55.2296 | 55.7404 | 56.0125 |
| lon | 20000 | 36.7024 | 37.5639 | 37.9081 |
| total_area | 20000 | 22.1 | 53.8 | 104.6 |
| to_center_km | 20000 | 0.02 | 13.23 | 71.06 |
| rooms | 20000 | 1 | 2 | 4 |
| floor | 20000 | 1 | 5 | 22 |
| total_floors | 20000 | 5 | 9 | 22 |
| building_year | 20000 | 1935 | 1988 | 2024 |
| deposit_months | 20000 | 0 | 1 | 2 |
| metro_distance_min | 20000 | 0 | 103 | 966 |
| monthly_rent_rub | 20000 | 6800 | 45800 | 300700 |
| rent_per_sqm | 20000 | 148 | 845 | 4264 |

Los valores extremos se conservan: no se aplico ningun filtro de atipicos.

## 10. Decisiones pendientes (para el EDA / etapas posteriores)

- **Fuga de informacion (`rent_per_sqm`)**: coincide con `monthly_rent_rub / total_area` (tolerancia +-2%) en el 100.0% de las filas (corr = 0.893 con el precio). Es una columna derivada de la variable objetivo; se conserva aqui, pero debe excluirse de las features si el objetivo es `monthly_rent_rub`.
- **`building_year` posterior a `date_posted`** en 339 filas. Podrian ser obras nuevas en preventa o un error; sin evidencia no se modifico. Decidir tras el EDA.
- **Semantica de `metro_*`**: cada estacion aparece en una mediana de 11 de 11 okrugs, lo que sugiere que la asignacion estacion-ubicacion no refleja la cercania real; ademas corr(metro_distance_min, to_center_km) = 0.90 y 45.0% de las filas supera 120 min. Verificar con la fuente el significado/unidad antes de usar `metro_station`, `metro_line` y `metro_distance_min`.
- **Mismo punto geografico** (lat y lon identicas) en 2 filas que difieren en otras columnas (area, piso, ...): NO son duplicados y se conservaron.
- **Valores extremos** (precio, area, distancias): no se tocaron; evaluar en el EDA si son legitimos y como tratarlos.
- **Esquema de validacion**: `date_posted` abarca 2021-01-01 a 2026-04-29; decidir si conviene validacion temporal.
- **Escalado, imputacion, codificacion y seleccion de variables**: fuera del alcance de esta etapa.

## 11. Garantias

- El archivo original no se modifica (hash comprobado antes y despues).
- Las salidas se escriben de forma atomica (temporal + reemplazo).
- Sin escalado, imputacion, codificacion ni seleccion de variables; sin dataset de entrenamiento.
