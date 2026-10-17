# Diagnóstico completo — modelo ganador: **LinearRegression** (59 features)

Split: train n=16012 (80.1 %) / valid n=1995 (10.0 %) / test n=1993 (10.0 %)

Modelo ajustado SOLO con *train*; *valid* y *test* son datos no vistos (el 05 reentrena con train+valid, por eso sus cifras de valid/test pueden diferir ligeramente).

## 1. Métricas por partición

| Métrica | train | valid | test |
|---|---:|---:|---:|
| MSE_log | 0.0097 | 0.0103 | 0.0105 |
| RMSE_log | 0.0984 | 0.1015 | 0.1023 |
| MAE_log | 0.0782 | 0.0807 | 0.0821 |
| MedAE_log | 0.0667 | 0.0672 | 0.0720 |
| R2_log | 0.9672 | 0.9647 | 0.9636 |
| R2_ajustado_log | 0.9670 | 0.9636 | 0.9625 |
| Sesgo_log (media real-pred) | 3.146e-15 | 0.0108 | 0.0209 |
| Desv_residuos_log | 0.0984 | 0.1009 | 0.1002 |
| Asimetria_residuos | -0.2478 | -0.2890 | -0.2872 |
| Exceso_curtosis_residuos | 0.2984 | 0.3956 | 0.2487 |
| Max_|z|_residuo | 4.7495 | 4.7907 | 4.1893 |
| MSE_RUB | 37,062,960 | 42,380,034 | 44,476,262 |
| RMSE_RUB | 6,088 | 6,510 | 6,669 |
| MAE_RUB | 4,144 | 4,521 | 4,583 |
| R2_RUB | 0.9580 | 0.9566 | 0.9530 |
| Sesgo_RUB (media real-pred) | 49 | 684 | 1,281 |
| MAPE_% | 7.9110 | 8.0369 | 8.0736 |
| MSPE (≈MAPE², error % al cuadrado) [%²] | 102.7813 | 105.3745 | 102.8326 |
| RMSPE_% (raíz del MSPE) | 10.1381 | 10.2652 | 10.1406 |
| sMAPE_% | 7.8052 | 8.0080 | 8.1240 |
| Aciertos_±10% | 69.9163 | 68.3208 | 68.1887 |
| Aciertos_±20% | 95.4347 | 95.7895 | 95.6347 |
| Spearman(|resid|, ŷ) [heteroced.] | 0.0765 | 0.0576 | 0.0742 |
| Jarque-Bera p-valor | 3.383e-49 | 1.401e-09 | 8.581e-08 |
| Breusch-Pagan p-valor | 9.218e-39 | 0.0304 | 0.0006038 |
| Durbin-Watson (≈2 = sin autocorr.) | 1.9854 | 2.0741 | 1.9429 |
| n | 16,012 | 1,995 | 1,993 |

> **MAPE²:** el cuadrado literal del MAPE no tiene interpretación estándar; se reporta el **MSPE** (media del error porcentual al cuadrado, en %²) y su raíz RMSPE, que son su equivalente correcto.

## 2. Semáforo de calidad (reglas prácticas, NO estándares universales)

| Criterio | valid | test | Regla | Estado |
|---|---:|---:|---|---|
| Sobreajuste: RMSE_log (valid | test) / train | 1.0309 | 1.0398 | ≤1.10 OK, ≤1.25 atención | **OK** |
| Sesgo medio (log) | 0.0108 | 0.0209 | |sesgo| ≤0.02 OK, ≤0.05 atención | **ATENCIÓN** |
| R² (log) | 0.9647 | 0.9636 | ≥0.90 OK, ≥0.80 atención | **OK** |
| MAPE % | 8.0369 | 8.0736 | ≤10 OK, ≤15 atención | **OK** |
| Asimetría de residuos | -0.2890 | -0.2872 | |skew| ≤0.5 OK, ≤1 atención | **OK** |
| Exceso de curtosis | 0.3956 | 0.2487 | |kurt| ≤1 OK, ≤3 atención | **OK** |
| Heterocedasticidad Spearman(|r|,ŷ) | 0.0576 | 0.0742 | |ρ| ≤0.10 OK, ≤0.25 atención | **OK** |
| Autocorrelación (Durbin-Watson) | 0.0741 | -0.0571 | |DW−2| ≤0.5 OK, ≤1 atención | **OK** |

**Cómo leerlo:** con n ≈ 2 000 los tests de Jarque-Bera y Breusch-Pagan rechazan casi siempre (son muy sensibles); guíate por magnitud (skew, curtosis, ρ) y por las gráficas, no solo por el p-valor.

Valid y test deberían dar métricas parecidas entre sí y cercanas a train; una brecha grande entre train y valid/test indica sobreajuste, y un sesgo creciente por mes indica deriva temporal.
