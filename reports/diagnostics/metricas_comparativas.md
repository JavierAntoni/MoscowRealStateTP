# Diagnóstico completo — modelo ganador: **HuberRegressor** (57 features)

Split: train n=16012 (80.1 %) / valid n=1995 (10.0 %) / test n=1993 (10.0 %)

Modelo ajustado SOLO con *train*; *valid* y *test* son datos no vistos (el 05 reentrena con train+valid, por eso sus cifras de valid/test pueden diferir ligeramente).

## 1. Métricas por partición

| Métrica | train | valid | test |
|---|---:|---:|---:|
| MSE_log | 0.0097 | 0.0103 | 0.0103 |
| RMSE_log | 0.0985 | 0.1013 | 0.1014 |
| MAE_log | 0.0781 | 0.0804 | 0.0811 |
| MedAE_log | 0.0664 | 0.0668 | 0.0694 |
| R2_log | 0.9671 | 0.9648 | 0.9643 |
| R2_ajustado_log | 0.9670 | 0.9638 | 0.9632 |
| Sesgo_log (media real-pred) | -0.0022 | 0.0056 | 0.0157 |
| Desv_residuos_log | 0.0985 | 0.1012 | 0.1002 |
| Asimetria_residuos | -0.2621 | -0.3027 | -0.3128 |
| Exceso_curtosis_residuos | 0.3487 | 0.4275 | 0.3042 |
| Max_|z|_residuo | 4.8353 | 4.8400 | 4.2287 |
| MSE_RUB | 37,178,889 | 42,368,621 | 43,867,970 |
| RMSE_RUB | 6,097 | 6,509 | 6,623 |
| MAE_RUB | 4,141 | 4,517 | 4,554 |
| R2_RUB | 0.9579 | 0.9566 | 0.9536 |
| Sesgo_RUB (media real-pred) | 43 | 512 | 1,107 |
| MAPE_% | 7.9036 | 8.0600 | 8.0431 |
| MSPE (≈MAPE², error % al cuadrado) [%²] | 103.0716 | 106.7931 | 103.1415 |
| RMSPE_% (raíz del MSPE) | 10.1524 | 10.3341 | 10.1559 |
| sMAPE_% | 7.7960 | 8.0045 | 8.0662 |
| Aciertos_±10% | 69.8851 | 68.2707 | 68.4897 |
| Aciertos_±20% | 95.4222 | 95.5890 | 95.7351 |
| Spearman(|resid|, ŷ) [heteroced.] | 0.0770 | 0.0543 | 0.0766 |
| Jarque-Bera p-valor | 3.856e-58 | 1.219e-10 | 1.877e-09 |
| Breusch-Pagan p-valor | 1.123e-45 | 0.0048 | 6.641e-05 |
| Durbin-Watson (≈2 = sin autocorr.) | 1.9837 | 2.0910 | 1.9817 |
| n | 16,012 | 1,995 | 1,993 |

> **MAPE²:** el cuadrado literal del MAPE no tiene interpretación estándar; se reporta el **MSPE** (media del error porcentual al cuadrado, en %²) y su raíz RMSPE, que son su equivalente correcto.

## 2. Semáforo de calidad (reglas prácticas, NO estándares universales)

| Criterio | valid | test | Regla | Estado |
|---|---:|---:|---|---|
| Sobreajuste: RMSE_log (valid | test) / train | 1.0285 | 1.0293 | ≤1.10 OK, ≤1.25 atención | **OK** |
| Sesgo medio (log) | 0.0056 | 0.0157 | |sesgo| ≤0.02 OK, ≤0.05 atención | **OK** |
| R² (log) | 0.9648 | 0.9643 | ≥0.90 OK, ≥0.80 atención | **OK** |
| MAPE % | 8.0600 | 8.0431 | ≤10 OK, ≤15 atención | **OK** |
| Asimetría de residuos | -0.3027 | -0.3128 | |skew| ≤0.5 OK, ≤1 atención | **OK** |
| Exceso de curtosis | 0.4275 | 0.3042 | |kurt| ≤1 OK, ≤3 atención | **OK** |
| Heterocedasticidad Spearman(|r|,ŷ) | 0.0543 | 0.0766 | |ρ| ≤0.10 OK, ≤0.25 atención | **OK** |
| Autocorrelación (Durbin-Watson) | 0.0910 | -0.0183 | |DW−2| ≤0.5 OK, ≤1 atención | **OK** |

**Cómo leerlo:** con n ≈ 2 000 los tests de Jarque-Bera y Breusch-Pagan rechazan casi siempre (son muy sensibles); guíate por magnitud (skew, curtosis, ρ) y por las gráficas, no solo por el p-valor.

Valid y test deberían dar métricas parecidas entre sí y cercanas a train; una brecha grande entre train y valid/test indica sobreajuste, y un sesgo creciente por mes indica deriva temporal.
