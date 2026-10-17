# Diagnóstico de residuos — Paso 5 (entrenamiento + MLflow)

- Modelo reentrenado: **LinearRegression** (allowlist lineal).
- Conjunto de ajuste: `train`.
- Smearing de Duan: `S = 1.004815`.
- Fecha (UTC): 2026-10-17T13:18:17+00:00.

## Escala log (homocedasticidad)
- Residuos log — media 0.0108, std 0.1009, skew -0.289, kurtosis 0.400 (valid).
- Q-Q: desviación máxima |z| = 4.79 (valid), 4.19 (test). Valores altos confirman las colas pesadas observadas en la etapa 04.
- Kurtosis > 0 implica colas más pesadas que una normal en escala log: el Q-Q se separa en los extremos (patrón previo: colas pesadas).

## Escala RUB (impacto de negocio)
- La retransformación a RUB reintroduce el abanico: Spearman(|residuo|, predicho) pasa de 0.058 (log) a 0.501 (RUB). Coherente con lo visto en la etapa 04: en log hay relativa homocedasticidad y en RUB la varianza crece con el nivel de precio.
- Cola alta (>= 200,000 RUB, valid): n=7 filas (0.35%), sesgo medio (real - predicho) = 12,150 RUB, MAPE = 6.15%.
- **Subestimación persistente de la cola alta**: con familia lineal las predicciones quedan por debajo del valor real en promedio 12,150 RUB. Esperado: la restricción lineal acota el rango predicho; el alivio viene de splines + smearing, no de más datos.
- Sesgo global (real - predicho) en valid: 684 RUB; MAPE valid = 8.04%.

## Comparación con las gráficas previas (etapa 04)
- Predicciones previas de valid (`predictions_valid.csv`): abanico Spearman = 0.977, sesgo cola alta = 226,031 RUB, MAPE = 99.97%.
- Deltas (05 − 04): abanico -0.476, sesgo cola -213,880 RUB. Un sesgo de cola menos positivo indica mejoría en la subestimación.

- Recuerde: el objetivo de modelado es `log1p(monthly_rent_rub)`; en RUB la heterocedasticidad reaparece por la retransformación. El diagnóstico decisivo de homocedasticidad es el de escala log; el de RUB mide impacto de negocio.
