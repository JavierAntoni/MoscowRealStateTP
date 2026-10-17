# Diagnóstico de residuos — Paso 5 (entrenamiento + MLflow)

- Modelo reentrenado: **LinearRegression** (allowlist lineal).
- Conjunto de ajuste: `train`.
- Smearing de Duan: `S = 1.003500`.
- Fecha (UTC): 2026-10-17T21:43:22+00:00.

## Escala log (homocedasticidad)
- Residuos log — media 0.0126, std 0.0847, skew -0.085, kurtosis -0.177 (valid).
- Q-Q: desviación máxima |z| = 3.44 (valid), 3.07 (test). Ningún residuo supera |z| = 4: sin outliers extremos en el Q-Q.
- Exceso de curtosis: valid -0.177 (colas cercanas a la normal), test -0.460 (colas cercanas a la normal).
- Sesgo medio en log (real − predicho): valid +0.0126, test +0.0233. Sesgo positivo creciente tras el corte de train: deriva de mercado (el spline temporal mantiene constante el último nivel); `--fit-on train_valid` con el preprocesador train+valid la reduce.

## Escala RUB (impacto de negocio)
- La retransformación a RUB reintroduce el abanico: Spearman(|residuo|, predicho) pasa de -0.024 (log) a 0.453 (RUB). Coherente con lo visto en la etapa 04: en log hay relativa homocedasticidad y en RUB la varianza crece con el nivel de precio.
- Cola alta (>= 200,000 RUB, valid): n=7 filas (0.35%), sesgo medio (real - predicho) = -1,656 RUB, MAPE = 7.90%.
- Sin subestimación neta de la cola alta en valid (sesgo <= 0). Conviene verificarlo también en test.
- Sesgo global (real - predicho) en valid: 683 RUB; MAPE valid = 6.84%.

## Comparación con las gráficas previas (etapa 04)
- Predicciones previas de valid (`predictions_valid.csv`): abanico Spearman = 0.453, sesgo cola alta = -1,656 RUB, MAPE = 6.84%.
- Deltas (05 − 04): abanico +0.000, sesgo cola -0 RUB. Un sesgo de cola menos positivo indica mejoría en la subestimación.

- Recuerde: el objetivo de modelado es `log1p(monthly_rent_rub)`; en RUB la heterocedasticidad reaparece por la retransformación. El diagnóstico decisivo de homocedasticidad es el de escala log; el de RUB mide impacto de negocio.
