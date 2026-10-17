# Reporte de ablación y diagnóstico (Etapa 03/04)

Ridge sobre `log1p(monthly_rent_rub)`; alpha elegido solo con *valid*. Test no interviene en ninguna decisión.

## 1. Ablación acumulativa

| Variante | α | RMSE valid | RMSE test | Sesgo valid | Sesgo test | MAPE test % | ρ(\|r\|,ŷ) test |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0. Baseline (v1) | 0.0001 | 0.1463 | 0.1507 | +0.068 | +0.081 | 10.77 | +0.09 |
| c. Control: tendencia lineal | 10.0 | 0.1546 | 0.1653 | -0.084 | -0.106 | 15.48 | +0.13 |
| 1. + tiempo (spline, const.) | 0.0001 | 0.1297 | 0.1276 | -0.002 | +0.012 | 9.99 | +0.19 |
| 2. + spline total_area | 0.1 | 0.1270 | 0.1246 | +0.002 | +0.014 | 9.64 | +0.18 |
| 3. + metro log/spline (FINAL) | 0.0001 | 0.1015 | 0.1025 | +0.011 | +0.021 | 8.09 | +0.07 |
| c. Control: FINAL + Box-Cox λ=-0.2 | 0.0001 | 0.1051 | 0.1055 | +0.012 | +0.023 | 8.30 | +0.03 |

λ de Box-Cox por máxima verosimilitud (solo train): **0.072** (≈ log). Un λ más agresivo (−0.2) reduce ρ (ver fila de control) a cambio de algo más de RMSE: es un intercambio de precisión por homocedasticidad, no una mejora neta.

## 2. Sesgo por segmento — test (modelo final vs baseline)

**Distancia al centro (quintiles de train)**

| Segmento | n | RMSE baseline | RMSE final | Desv. resid. baseline | Desv. resid. final |
|---|---:|---:|---:|---:|---:|
| (-inf, 8.06] | 422 | 0.1704 | 0.1057 | 0.1533 | 0.1036 |
| (8.06, 11.63] | 369 | 0.1682 | 0.1114 | 0.1450 | 0.1073 |
| (11.63, 14.72] | 399 | 0.1414 | 0.0960 | 0.1184 | 0.0939 |
| (14.72, 20.52] | 411 | 0.1449 | 0.1082 | 0.1209 | 0.1075 |
| (20.52, inf] | 392 | 0.1233 | 0.0898 | 0.0866 | 0.0863 |

**Cuantil de ŷ (quintiles de train)**

| Segmento | n | RMSE baseline | RMSE final | Desv. resid. baseline | Desv. resid. final |
|---|---:|---:|---:|---:|---:|
| (-inf, 10.298] | 345 | 0.1229 | 0.0898 | 0.0944 | 0.0872 |
| (10.298, 10.599] | 390 | 0.1428 | 0.0975 | 0.1141 | 0.0934 |
| (10.599, 10.844] | 381 | 0.1424 | 0.0974 | 0.1210 | 0.0961 |
| (10.844, 11.139] | 416 | 0.1650 | 0.1136 | 0.1418 | 0.1129 |
| (11.139, inf] | 461 | 0.1757 | 0.1090 | 0.1571 | 0.1053 |

## 3. Sensibilidad «sin propiedades de lujo» (solo diagnóstico)

Se excluye el 5 % con mayor ŷ. **No** se propone como cambio del modelo: filtrar por una función del objetivo introduce sesgo de selección y no resuelve la causa.

| Partición | Modelo | RMSE (todas) | RMSE (sin top 5 %) | ρ (todas) | ρ (sin top 5 %) |
|---|---|---:|---:|---:|---:|
| valid | baseline | 0.1463 | 0.1444 | +0.15 | +0.13 |
| valid | final | 0.1015 | 0.1016 | +0.05 | +0.05 |
| test | baseline | 0.1507 | 0.1498 | +0.09 | +0.08 |
| test | final | 0.1025 | 0.1028 | +0.07 | +0.08 |

![antes/después](figures/07_before_after.png)