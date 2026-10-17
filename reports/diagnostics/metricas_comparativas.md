# Diagnóstico completo — modelo ganador: **LinearRegression** (57 features)

Split: train n=16012 (80.1 %) / valid n=1995 (10.0 %) / test n=1993 (10.0 %)

Modelo ajustado SOLO con *train*; *valid* y *test* son datos no vistos (el 05 reentrena con train+valid, por eso sus cifras de valid/test pueden diferir ligeramente).

## 1. Métricas por partición

| Métrica | train | valid | test |
|---|---:|---:|---:|
| MSE_log | 0.0070 | 0.0073 | 0.0077 |
| RMSE_log | 0.0836 | 0.0856 | 0.0879 |
| MAE_log | 0.0677 | 0.0692 | 0.0720 |
| MedAE_log | 0.0588 | 0.0583 | 0.0640 |
| R2_log | 0.9763 | 0.9749 | 0.9731 |
| R2_ajustado_log | 0.9762 | 0.9741 | 0.9723 |
| Sesgo_log (media real-pred) | 9.377e-16 | 0.0126 | 0.0233 |
| Desv_residuos_log | 0.0836 | 0.0847 | 0.0848 |
| Asimetria_residuos | -0.0239 | -0.0847 | -0.0317 |
| Exceso_curtosis_residuos | -0.2625 | -0.1796 | -0.4619 |
| Max_|z|_residuo | 3.7589 | 3.4360 | 3.0684 |
| MSE_RUB | 24,506,818 | 28,227,108 | 30,614,110 |
| RMSE_RUB | 4,950 | 5,313 | 5,533 |
| MAE_RUB | 3,488 | 3,774 | 3,916 |
| R2_RUB | 0.9722 | 0.9711 | 0.9676 |
| Sesgo_RUB (media real-pred) | -7 | 683 | 1,314 |
| MAPE_% | 6.8184 | 6.8375 | 7.0110 |
| MSPE (≈MAPE², error % al cuadrado) [%²] | 71.7382 | 71.9462 | 72.4478 |
| RMSPE_% (raíz del MSPE) | 8.4698 | 8.4821 | 8.5116 |
| sMAPE_% | 6.7690 | 6.8662 | 7.1209 |
| Aciertos_±10% | 75.4746 | 74.9373 | 74.4104 |
| Aciertos_±20% | 98.3450 | 98.7469 | 99.0968 |
| Spearman(|resid|, ŷ) [heteroced.] | -0.0025 | -0.0240 | 0.0012 |
| Jarque-Bera p-valor | 4.849e-11 | 0.0793 | 0.0001203 |
| Breusch-Pagan p-valor | 0.0650 | 0.7214 | 0.2244 |
| Durbin-Watson (≈2 = sin autocorr.) | 1.9844 | 1.9936 | 1.8661 |
| n | 16,012 | 1,995 | 1,993 |

> **MAPE²:** el cuadrado literal del MAPE no tiene interpretación estándar; se reporta el **MSPE** (media del error porcentual al cuadrado, en %²) y su raíz RMSPE, que son su equivalente correcto.

## 2. Semáforo de calidad (reglas prácticas, NO estándares universales)

| Criterio | valid | test | Regla | Estado |
|---|---:|---:|---|---|
| Sobreajuste: RMSE_log (valid | test) / train | 1.0237 | 1.0511 | ≤1.10 OK, ≤1.25 atención | **OK** |
| Sesgo medio (log) | 0.0126 | 0.0233 | |sesgo| ≤0.02 OK, ≤0.05 atención | **ATENCIÓN** |
| R² (log) | 0.9749 | 0.9731 | ≥0.90 OK, ≥0.80 atención | **OK** |
| MAPE % | 6.8375 | 7.0110 | ≤10 OK, ≤15 atención | **OK** |
| Asimetría de residuos | -0.0847 | -0.0317 | |skew| ≤0.5 OK, ≤1 atención | **OK** |
| Exceso de curtosis | -0.1796 | -0.4619 | |kurt| ≤1 OK, ≤3 atención | **OK** |
| Heterocedasticidad Spearman(|r|,ŷ) | -0.0240 | 0.0012 | |ρ| ≤0.10 OK, ≤0.25 atención | **OK** |
| Autocorrelación (Durbin-Watson) | -0.0064 | -0.1339 | |DW−2| ≤0.5 OK, ≤1 atención | **OK** |

**Cómo leerlo:** con n ≈ 2 000 los tests de Jarque-Bera y Breusch-Pagan rechazan casi siempre (son muy sensibles); guíate por magnitud (skew, curtosis, ρ) y por las gráficas, no solo por el p-valor.

Valid y test deberían dar métricas parecidas entre sí y cercanas a train; una brecha grande entre train y valid/test indica sobreajuste, y un sesgo creciente por mes indica deriva temporal.
