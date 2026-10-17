# Diagnóstico de residuos — Paso 5 (entrenamiento + MLflow)

- Modelo reentrenado: **HuberRegressor** (allowlist lineal).
- Conjunto de ajuste: `train`.
- Smearing de Duan: `S = 1.002622`.
- Fecha (UTC): 2026-10-17T16:30:20+00:00.

## Escala log (homocedasticidad)
- Residuos log — media 0.0056, std 0.1012, skew -0.303, kurtosis 0.432 (valid).
- Q-Q: desviación máxima |z| = 4.84 (valid), 4.23 (test). Valores altos confirman las colas pesadas observadas en la etapa 04.
- Kurtosis > 0 implica colas más pesadas que una normal en escala log: el Q-Q se separa en los extremos (patrón previo: colas pesadas).

## Escala RUB (impacto de negocio)
- La retransformación a RUB reintroduce el abanico: Spearman(|residuo|, predicho) pasa de 0.054 (log) a 0.500 (RUB). Coherente con lo visto en la etapa 04: en log hay relativa homocedasticidad y en RUB la varianza crece con el nivel de precio.
- Cola alta (>= 200,000 RUB, valid): n=7 filas (0.35%), sesgo medio (real - predicho) = 11,584 RUB, MAPE = 5.93%.
- **Subestimación persistente de la cola alta**: con familia lineal las predicciones quedan por debajo del valor real en promedio 11,584 RUB. Esperado: la restricción lineal acota el rango predicho; el alivio viene de splines + smearing, no de más datos.
- Sesgo global (real - predicho) en valid: 512 RUB; MAPE valid = 8.06%.

## Comparación con las gráficas previas (etapa 04)
- Predicciones previas de valid (`predictions_valid.csv`): abanico Spearman = 0.977, sesgo cola alta = 226,031 RUB, MAPE = 99.97%.
- Deltas (05 − 04): abanico -0.477, sesgo cola -214,447 RUB. Un sesgo de cola menos positivo indica mejoría en la subestimación.

- Recuerde: el objetivo de modelado es `log1p(monthly_rent_rub)`; en RUB la heterocedasticidad reaparece por la retransformación. El diagnóstico decisivo de homocedasticidad es el de escala log; el de RUB mide impacto de negocio.
